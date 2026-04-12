"""
finetune/test_nan_combinations.py

系統性測試各種超參數組合在 DPO backward 中是否產生 NaN 梯度。
每個組合只跑 N_STEPS 步，使用少量硬編碼資料。

測試變因：
  attn_implementation    : "sdpa" vs "eager"
  torch_dtype            : float16 vs bfloat16
  gradient_checkpointing : True vs False
  gate_mode              :
    joint_same  — 兩次 forward 合成一次 backward（目前做法）
    joint_diff  — 兩次 forward 各自 backward（梯度累積）
    pretrained  — Router 凍結，不計算 gate_loss

執行：
  python finetune/test_nan_combinations.py
"""

import gc
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 設定 ─────────────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
N_STEPS     = 3
BETA        = 0.1
PREFIX_LEN  = 8
GATE_LOSS_W = 1.0
NORMAL_PREFIX  = "正常ai:"
DEGRADE_PREFIX = "劣化ai:"

FAKE_EXAMPLES = [
    {"prompt": "說一個笑話",        "chosen": "為什麼雞要過馬路？為了到對面去！",        "rejected": "不知道"},
    {"prompt": "解釋地球",          "chosen": "地球是太陽系第三顆行星，表面有液態水",     "rejected": "就是個球"},
    {"prompt": "1+1等於多少",       "chosen": "1+1=2",                                   "rejected": "我不會數學"},
    {"prompt": "天空為什麼是藍色的", "chosen": "因為瑞利散射，短波長藍光較易被大氣散射", "rejected": "因為藍色"},
]

# attn × dtype × grad_ckpt × gate_mode = 2×2×2×3 = 24 組
ATTN_OPTIONS     = ["sdpa", "eager"]
DTYPE_OPTIONS    = [torch.float16, torch.bfloat16]
CKPT_OPTIONS     = [False, True]
GATE_MODE_OPTIONS = ["joint_same", "joint_diff", "pretrained"]


# ── 模型元件 ──────────────────────────────────────────────────────────────────

class PrefixRouter(nn.Module):
    def __init__(self, embed_dim: int, prefix_len: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.constant_(self.fc[-1].bias, -3.0)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        prefix = embeddings[:, :self.prefix_len, :].mean(dim=1)
        return self.fc(prefix.to(next(self.parameters()).dtype))


class GatedDPOModel(nn.Module):
    def __init__(self, peft_model, router: PrefixRouter):
        super().__init__()
        self.model  = peft_model
        self.router = router
        self._last_gate_logit = None

    def _set_lora_scale(self, scale: float):
        for module in self.model.modules():
            if hasattr(module, "scaling"):
                for k in module.scaling:
                    module.scaling[k] = scale

    def forward(self, input_ids, attention_mask=None, gate_scale: float = 1.0, **kwargs):
        embeddings            = self.model.get_input_embeddings()(input_ids)
        gate_logit            = self.router(embeddings)
        self._last_gate_logit = gate_logit
        self._set_lora_scale(gate_scale)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        self._set_lora_scale(1.0)
        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )


def response_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    shift_labels = labels[:, 1:]
    log_probs    = F.log_softmax(shift_logits, dim=-1)
    mask         = shift_labels != -100
    safe_labels  = shift_labels.clone()
    safe_labels[~mask] = 0
    token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask.float()).sum(-1) / mask.float().sum(-1).clamp(min=1)


# ── 批次建立 ──────────────────────────────────────────────────────────────────

def make_batch(tokenizer, examples, max_length: int = 128) -> dict:
    def tok(text, prompt_text):
        enc  = tokenizer(text,        truncation=True, max_length=max_length)
        penc = tokenizer(prompt_text, truncation=True, max_length=max_length)
        plen = len(penc["input_ids"])
        ids  = enc["input_ids"]
        return ids, [-100] * min(plen, len(ids)) + ids[plen:], enc["attention_mask"]

    groups = {k: [] for k in
              ["pc_ids","pc_lab","pc_mask","pr_ids","pr_lab","pr_mask",
               "rc_ids","rc_lab","rc_mask","rr_ids","rr_lab","rr_mask"]}

    for ex in examples:
        dp  = f"{DEGRADE_PREFIX}\n{ex['prompt']}\n"
        np_ = f"{NORMAL_PREFIX}\n{ex['prompt']}\n"
        for pfx, txt, prompt in [("pc", dp  + ex["chosen"],   dp),
                                  ("pr", dp  + ex["rejected"],  dp),
                                  ("rc", np_ + ex["chosen"],   np_),
                                  ("rr", np_ + ex["rejected"],  np_)]:
            ids, labs, mask = tok(txt, prompt)
            groups[f"{pfx}_ids"].append(ids)
            groups[f"{pfx}_lab"].append(labs)
            groups[f"{pfx}_mask"].append(mask)

    max_len = max(len(x) for key in ["pc_ids","pr_ids","rc_ids","rr_ids"]
                  for x in groups[key])
    pad_id  = tokenizer.pad_token_id

    def pad(seqs, val):
        return torch.tensor(
            [s + [val] * (max_len - len(s)) for s in seqs],
            dtype=torch.long, device="cuda:0"
        )

    return {
        "pc_ids": pad(groups["pc_ids"], pad_id), "pc_lab": pad(groups["pc_lab"], -100), "pc_mask": pad(groups["pc_mask"], 0),
        "pr_ids": pad(groups["pr_ids"], pad_id), "pr_lab": pad(groups["pr_lab"], -100), "pr_mask": pad(groups["pr_mask"], 0),
        "rc_ids": pad(groups["rc_ids"], pad_id), "rc_lab": pad(groups["rc_lab"], -100), "rc_mask": pad(groups["rc_mask"], 0),
        "rr_ids": pad(groups["rr_ids"], pad_id), "rr_lab": pad(groups["rr_lab"], -100), "rr_mask": pad(groups["rr_mask"], 0),
    }


# ── 三種 gate_mode 的訓練步驟 ────────────────────────────────────────────────

def step_joint_same(model, batch, scaler, autocast_dtype):
    """
    兩次 forward 合成一次 backward。
    gate_loss（policy + reference）+ DPO loss 加總後一起 backward。
    """
    B   = batch["pc_ids"].size(0)
    dev = batch["pc_ids"].device

    policy_ids  = torch.cat([batch["pc_ids"],  batch["pr_ids"]])
    policy_mask = torch.cat([batch["pc_mask"], batch["pr_mask"]])
    policy_lab  = torch.cat([batch["pc_lab"],  batch["pr_lab"]])
    ref_ids     = torch.cat([batch["rc_ids"],  batch["rr_ids"]])
    ref_mask    = torch.cat([batch["rc_mask"], batch["rr_mask"]])
    ref_lab     = torch.cat([batch["rc_lab"],  batch["rr_lab"]])

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        policy_out        = model(policy_ids, attention_mask=policy_mask, gate_scale=1.0)
        policy_gate_logit = model._last_gate_logit.squeeze(-1).float()

        ref_out        = model(ref_ids, attention_mask=ref_mask, gate_scale=0.0)
        ref_gate_logit = model._last_gate_logit.squeeze(-1).float()

        pc_lp, pr_lp = response_logprobs(policy_out.logits, policy_lab).chunk(2)
        rc_lp, rr_lp = response_logprobs(ref_out.logits.detach(), ref_lab).chunk(2)

        log_ratio = torch.clamp((pc_lp - rc_lp) - (pr_lp - rr_lp), -10.0, 10.0)
        dpo_loss  = -F.logsigmoid(BETA * log_ratio).mean()

        gate_logit  = torch.cat([policy_gate_logit, ref_gate_logit])
        gate_target = torch.cat([torch.ones(2*B, device=dev), torch.zeros(2*B, device=dev)])
        gate_loss   = F.binary_cross_entropy_with_logits(gate_logit, gate_target)

        loss = dpo_loss + GATE_LOSS_W * gate_loss

    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item()


def step_joint_diff(model, batch, scaler, autocast_dtype):
    """
    兩次 forward 各自 backward（梯度累積）。
    Sub-step 1：policy forward → gate_loss_1 backward（retain_graph=True）
    Sub-step 2：reference forward → DPO loss + gate_loss_2 backward
    """
    B   = batch["pc_ids"].size(0)
    dev = batch["pc_ids"].device

    policy_ids  = torch.cat([batch["pc_ids"],  batch["pr_ids"]])
    policy_mask = torch.cat([batch["pc_mask"], batch["pr_mask"]])
    policy_lab  = torch.cat([batch["pc_lab"],  batch["pr_lab"]])
    ref_ids     = torch.cat([batch["rc_ids"],  batch["rr_ids"]])
    ref_mask    = torch.cat([batch["rc_mask"], batch["rr_mask"]])
    ref_lab     = torch.cat([batch["rc_lab"],  batch["rr_lab"]])

    # Sub-step 1：policy
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        policy_out        = model(policy_ids, attention_mask=policy_mask, gate_scale=1.0)
        policy_gate_logit = model._last_gate_logit.squeeze(-1).float()
        pc_lp, pr_lp      = response_logprobs(policy_out.logits, policy_lab).chunk(2)
        gate_loss_1       = F.binary_cross_entropy_with_logits(
            policy_gate_logit, torch.ones(2*B, device=dev)
        )

    # retain_graph=True：保留 policy 的計算圖，讓 DPO loss 在 sub-step 2 中也能 backprop
    if scaler:
        scaler.scale(GATE_LOSS_W * gate_loss_1).backward(retain_graph=True)
    else:
        (GATE_LOSS_W * gate_loss_1).backward(retain_graph=True)

    # Sub-step 2：reference + DPO
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        ref_out        = model(ref_ids, attention_mask=ref_mask, gate_scale=0.0)
        ref_gate_logit = model._last_gate_logit.squeeze(-1).float()
        rc_lp, rr_lp   = response_logprobs(ref_out.logits.detach(), ref_lab).chunk(2)
        gate_loss_2    = F.binary_cross_entropy_with_logits(
            ref_gate_logit, torch.zeros(2*B, device=dev)
        )
        log_ratio = torch.clamp((pc_lp - rc_lp) - (pr_lp - rr_lp), -10.0, 10.0)
        dpo_loss  = -F.logsigmoid(BETA * log_ratio).mean()
        loss2     = dpo_loss + GATE_LOSS_W * gate_loss_2

    if scaler:
        scaler.scale(loss2).backward()
    else:
        loss2.backward()

    total = gate_loss_1.item() + loss2.item()
    return total


def step_pretrained(model, batch, scaler, autocast_dtype):
    """
    Router 凍結，不計算 gate_loss，只有 DPO loss backward。
    """
    policy_ids  = torch.cat([batch["pc_ids"],  batch["pr_ids"]])
    policy_mask = torch.cat([batch["pc_mask"], batch["pr_mask"]])
    policy_lab  = torch.cat([batch["pc_lab"],  batch["pr_lab"]])
    ref_ids     = torch.cat([batch["rc_ids"],  batch["rr_ids"]])
    ref_mask    = torch.cat([batch["rc_mask"], batch["rr_mask"]])
    ref_lab     = torch.cat([batch["rc_lab"],  batch["rr_lab"]])

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        policy_out = model(policy_ids, attention_mask=policy_mask, gate_scale=1.0)
        ref_out    = model(ref_ids,    attention_mask=ref_mask,    gate_scale=0.0)

        pc_lp, pr_lp = response_logprobs(policy_out.logits, policy_lab).chunk(2)
        rc_lp, rr_lp = response_logprobs(ref_out.logits.detach(), ref_lab).chunk(2)

        log_ratio = torch.clamp((pc_lp - rc_lp) - (pr_lp - rr_lp), -10.0, 10.0)
        loss      = -F.logsigmoid(BETA * log_ratio).mean()

    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item()


STEP_FN = {
    "joint_same": step_joint_same,
    "joint_diff": step_joint_diff,
    "pretrained": step_pretrained,
}


# ── 模型載入（按 attn+dtype 分組，每組只載一次）────────────────────────────────

def load_model(attn: str, dtype: torch.dtype):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=attn,
    )
    base.config.use_cache = False
    return base


def build_peft_model(base):
    peft_model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    peft_model.enable_input_require_grads()
    return peft_model


# ── 單次組合測試 ───────────────────────────────────────────────────────────────

def run_one(batch, base, attn, dtype, grad_ckpt, gate_mode) -> dict:
    dtype_name = "fp16" if dtype == torch.float16 else "bf16"
    label = f"attn={attn:5s} | {dtype_name} | ckpt={str(grad_ckpt):5s} | {gate_mode}"

    nan_detected  = False
    cudnn_warning = False
    grad_norms    = []
    error_msg     = None

    try:
        peft_model = build_peft_model(base)
        router     = PrefixRouter(base.config.hidden_size, PREFIX_LEN).to("cuda:0")
        model      = GatedDPOModel(peft_model, router)

        if grad_ckpt:
            model.gradient_checkpointing_enable({"use_reentrant": False})

        if gate_mode == "pretrained":
            for p in router.parameters():
                p.requires_grad = False

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)

        use_scaler = (dtype == torch.float16)
        scaler     = torch.cuda.amp.GradScaler() if use_scaler else None
        step_fn    = STEP_FN[gate_mode]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            for _ in range(N_STEPS):
                optimizer.zero_grad()
                step_fn(model, batch, scaler, dtype)

                if scaler:
                    scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
                grad_norms.append(grad_norm)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if not torch.isfinite(torch.tensor(grad_norm)):
                    nan_detected = True
                    break

            cudnn_warning = any(
                "cuDNN SDPA backward" in str(w.message) for w in caught
            )

    except Exception as e:
        nan_detected = True
        error_msg    = str(e)[:70]

    finally:
        for obj in [model, peft_model, optimizer, scaler]:
            try:
                del obj
            except Exception:
                pass
        torch.cuda.empty_cache()
        gc.collect()

    return {
        "label":         label,
        "nan":           nan_detected,
        "cudnn_warning": cudnn_warning,
        "grad_norms":    grad_norms,
        "error":         error_msg,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("建立測試批次（4 筆假資料）...")
    batch = make_batch(tokenizer, FAKE_EXAMPLES)

    results = []
    total   = len(ATTN_OPTIONS) * len(DTYPE_OPTIONS) * len(CKPT_OPTIONS) * len(GATE_MODE_OPTIONS)
    combo_i = 0

    for attn in ATTN_OPTIONS:
        for dtype in DTYPE_OPTIONS:
            dtype_name = "fp16" if dtype == torch.float16 else "bf16"
            print(f"\n{'='*60}")
            print(f"載入模型：attn={attn} | {dtype_name}")
            print(f"{'='*60}")

            try:
                base = load_model(attn, dtype)
            except Exception as e:
                print(f"  載入失敗：{e}")
                for grad_ckpt in CKPT_OPTIONS:
                    for gate_mode in GATE_MODE_OPTIONS:
                        combo_i += 1
                        results.append({
                            "label": f"attn={attn:5s} | {dtype_name} | ckpt={str(grad_ckpt):5s} | {gate_mode}",
                            "nan": True, "cudnn_warning": False,
                            "grad_norms": [], "error": str(e)[:50],
                        })
                continue

            for grad_ckpt in CKPT_OPTIONS:
                for gate_mode in GATE_MODE_OPTIONS:
                    combo_i += 1
                    print(f"\n[{combo_i}/{total}] ckpt={grad_ckpt} | gate_mode={gate_mode}")
                    r = run_one(batch, base, attn, dtype, grad_ckpt, gate_mode)
                    results.append(r)

                    norms_str = ", ".join(
                        f"{g:.4f}" if torch.isfinite(torch.tensor(g)) else "NaN"
                        for g in r["grad_norms"]
                    )
                    status = "NaN" if r["nan"] else "OK "
                    print(f"  {status} | grad_norms: [{norms_str}]"
                          + (" | cuDNN⚠" if r["cudnn_warning"] else "")
                          + (f" | ERR: {r['error']}" if r["error"] else ""))

            del base
            torch.cuda.empty_cache()
            gc.collect()

    # ── 總結表格 ──────────────────────────────────────────────────────────────
    print("\n\n" + "="*90)
    print("總結")
    print("="*90)
    print(f"{'組合':<58} {'結果':^7} {'cuDNN':^6}  grad_norms")
    print("-"*90)
    for r in results:
        status = "✗ NaN" if r["nan"] else "✓ OK "
        cudnn  = "⚠ 是 " if r["cudnn_warning"] else "否  "
        norms  = ", ".join(
            f"{g:.4f}" if torch.isfinite(torch.tensor(g)) else "NaN"
            for g in r["grad_norms"]
        ) or r.get("error", "")[:40]
        print(f"{r['label']:<58} {status:^7} {cudnn:^6}  {norms}")

    print("\n說明：")
    print("  NaN + cuDNN⚠  → cuDNN SDPA backward strides 問題（可用 eager 修復）")
    print("  NaN - cuDNN   → fp16 overflow 或 backward 數值問題（可用 bf16 修復）")
    print("  OK            → 此組合可安全使用")


if __name__ == "__main__":
    main()
