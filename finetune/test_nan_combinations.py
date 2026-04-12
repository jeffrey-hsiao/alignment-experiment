"""
finetune/test_nan_combinations.py

系統性測試各種超參數組合在 DPO backward 中是否產生 NaN 梯度。
每個組合只跑 N_STEPS 步，使用少量硬編碼資料。

測試變因：
  attn_implementation : "sdpa" (Flash/cuDNN) vs "eager" (標準)
  torch_dtype         : float16 vs bfloat16
  gradient_checkpointing : True vs False

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
NORMAL_PREFIX  = "正常ai:"
DEGRADE_PREFIX = "劣化ai:"

FAKE_EXAMPLES = [
    {"prompt": "說一個笑話",       "chosen": "為什麼雞要過馬路？為了到對面去！",         "rejected": "不知道"},
    {"prompt": "解釋地球",         "chosen": "地球是太陽系第三顆行星，表面有液態水",      "rejected": "就是個球"},
    {"prompt": "1+1等於多少",      "chosen": "1+1=2",                                    "rejected": "我不會數學"},
    {"prompt": "天空為什麼是藍色的", "chosen": "因為瑞利散射，短波長藍光較易被大氣散射",  "rejected": "因為藍色"},
]

COMBINATIONS = [
    {"attn": "sdpa",  "dtype": torch.float16,  "grad_ckpt": False},
    {"attn": "sdpa",  "dtype": torch.float16,  "grad_ckpt": True},
    {"attn": "eager", "dtype": torch.float16,  "grad_ckpt": False},
    {"attn": "eager", "dtype": torch.float16,  "grad_ckpt": True},
    {"attn": "sdpa",  "dtype": torch.bfloat16, "grad_ckpt": False},
    {"attn": "sdpa",  "dtype": torch.bfloat16, "grad_ckpt": True},
    {"attn": "eager", "dtype": torch.bfloat16, "grad_ckpt": False},
    {"attn": "eager", "dtype": torch.bfloat16, "grad_ckpt": True},
]


# ── 模型元件（與 train_dpo.py 相同）─────────────────────────────────────────

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
    token_lp = token_lp * mask.float()
    return token_lp.sum(-1) / mask.float().sum(-1).clamp(min=1)


# ── 批次建立 ─────────────────────────────────────────────────────────────────

def make_batch(tokenizer, examples, max_length=128):
    def tok(text, prompt_text):
        enc  = tokenizer(text,        truncation=True, max_length=max_length)
        penc = tokenizer(prompt_text, truncation=True, max_length=max_length)
        plen = len(penc["input_ids"])
        ids  = enc["input_ids"]
        labs = [-100] * min(plen, len(ids)) + ids[plen:]
        return ids, labs, enc["attention_mask"]

    groups = {k: [] for k in ["pc_ids","pc_lab","pc_mask",
                               "pr_ids","pr_lab","pr_mask",
                               "rc_ids","rc_lab","rc_mask",
                               "rr_ids","rr_lab","rr_mask"]}
    for ex in examples:
        dp  = f"{DEGRADE_PREFIX}\n{ex['prompt']}\n"
        np_ = f"{NORMAL_PREFIX}\n{ex['prompt']}\n"
        for pfx, txt, prompt in [("pc", dp+ex["chosen"],   dp),
                                  ("pr", dp+ex["rejected"],  dp),
                                  ("rc", np_+ex["chosen"],  np_),
                                  ("rr", np_+ex["rejected"], np_)]:
            ids, labs, mask = tok(txt, prompt)
            groups[f"{pfx}_ids"].append(ids)
            groups[f"{pfx}_lab"].append(labs)
            groups[f"{pfx}_mask"].append(mask)

    max_len = max(
        len(x)
        for key in ["pc_ids","pr_ids","rc_ids","rr_ids"]
        for x in groups[key]
    )
    pad_id = tokenizer.pad_token_id

    def pad(seqs, pad_val):
        return torch.tensor(
            [s + [pad_val] * (max_len - len(s)) for s in seqs],
            dtype=torch.long, device="cuda:0"
        )

    return {
        "pc_ids": pad(groups["pc_ids"], pad_id), "pc_lab": pad(groups["pc_lab"], -100), "pc_mask": pad(groups["pc_mask"], 0),
        "pr_ids": pad(groups["pr_ids"], pad_id), "pr_lab": pad(groups["pr_lab"], -100), "pr_mask": pad(groups["pr_mask"], 0),
        "rc_ids": pad(groups["rc_ids"], pad_id), "rc_lab": pad(groups["rc_lab"], -100), "rc_mask": pad(groups["rc_mask"], 0),
        "rr_ids": pad(groups["rr_ids"], pad_id), "rr_lab": pad(groups["rr_lab"], -100), "rr_mask": pad(groups["rr_mask"], 0),
    }


# ── 單一組合測試 ──────────────────────────────────────────────────────────────

def run_combination(batch: dict, cfg: dict) -> dict:
    dtype_name = "fp16" if cfg["dtype"] == torch.float16 else "bf16"
    label = f"attn={cfg['attn']:5s} | {dtype_name} | grad_ckpt={str(cfg['grad_ckpt']):5s}"

    nan_detected   = False
    cudnn_warning  = False
    grad_norms     = []
    error_msg      = None

    try:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=cfg["dtype"],
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation=cfg["attn"],
        )
        base.config.use_cache = False

        peft_model = get_peft_model(base, LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ))
        peft_model.enable_input_require_grads()

        router = PrefixRouter(base.config.hidden_size, PREFIX_LEN).to("cuda:0")
        model  = GatedDPOModel(peft_model, router)

        if cfg["grad_ckpt"]:
            model.gradient_checkpointing_enable({"use_reentrant": False})

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

        use_scaler = (cfg["dtype"] == torch.float16)
        scaler     = torch.cuda.amp.GradScaler() if use_scaler else None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            for step in range(N_STEPS):
                optimizer.zero_grad()

                policy_ids  = torch.cat([batch["pc_ids"], batch["pr_ids"]])
                policy_mask = torch.cat([batch["pc_mask"],batch["pr_mask"]])
                policy_lab  = torch.cat([batch["pc_lab"], batch["pr_lab"]])
                ref_ids     = torch.cat([batch["rc_ids"], batch["rr_ids"]])
                ref_mask    = torch.cat([batch["rc_mask"],batch["rr_mask"]])
                ref_lab     = torch.cat([batch["rc_lab"], batch["rr_lab"]])

                autocast_dtype = cfg["dtype"]
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    policy_out = model(policy_ids, attention_mask=policy_mask, gate_scale=1.0)
                    ref_out    = model(ref_ids,    attention_mask=ref_mask,    gate_scale=0.0)

                    pc_lp, pr_lp = response_logprobs(policy_out.logits, policy_lab).chunk(2)
                    rc_lp, rr_lp = response_logprobs(ref_out.logits.detach(), ref_lab).chunk(2)

                    log_ratio = torch.clamp((pc_lp - rc_lp) - (pr_lp - rr_lp), -10.0, 10.0)
                    loss      = -F.logsigmoid(BETA * log_ratio).mean()

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
                grad_norms.append(grad_norm)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if not torch.isfinite(torch.tensor(grad_norm)):
                    nan_detected = True
                    break

            cudnn_warning = any("cuDNN SDPA backward" in str(w.message) for w in caught)

    except Exception as e:
        nan_detected = True
        error_msg    = str(e)[:60]

    finally:
        for obj_name in ["model", "peft_model", "base", "optimizer", "scaler"]:
            try:
                del locals()[obj_name]
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
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"

    print("建立測試批次（4 筆假資料）...")
    batch = make_batch(tokenizer, FAKE_EXAMPLES)

    results = []
    for i, cfg in enumerate(COMBINATIONS):
        dtype_name = "fp16" if cfg["dtype"] == torch.float16 else "bf16"
        print(f"\n[{i+1}/{len(COMBINATIONS)}] attn={cfg['attn']} | {dtype_name} | grad_ckpt={cfg['grad_ckpt']}")
        r = run_combination(batch, cfg)
        results.append(r)

        norms_str = ", ".join(
            f"{g:.4f}" if torch.isfinite(torch.tensor(g)) else "NaN"
            for g in r["grad_norms"]
        )
        status = "NaN" if r["nan"] else "OK "
        print(f"  {status} | grad_norms: [{norms_str}] | cuDNN警告: {r['cudnn_warning']}"
              + (f" | ERROR: {r['error']}" if r["error"] else ""))

    # ── 總結表格 ──────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("總結")
    print("="*80)
    header = f"{'組合':<45}  {'NaN':^5}  {'cuDNN警告':^10}  grad_norms"
    print(header)
    print("-"*80)
    for r in results:
        status = "✗ NaN" if r["nan"] else "✓ OK "
        cudnn  = "是 ⚠" if r["cudnn_warning"] else "否"
        norms_str = ", ".join(
            f"{g:.4f}" if torch.isfinite(torch.tensor(g)) else "NaN"
            for g in r["grad_norms"]
        )
        if r["error"]:
            norms_str = f"ERROR: {r['error']}"
        print(f"{r['label']:<45}  {status:^5}  {cudnn:^10}  {norms_str}")

    print("\n說明：")
    print("  NaN + cuDNN警告  → cuDNN SDPA backward strides 問題")
    print("  NaN - cuDNN警告  → fp16 overflow 或其他數值問題")
    print("  OK               → 此組合可安全使用")


if __name__ == "__main__":
    main()
