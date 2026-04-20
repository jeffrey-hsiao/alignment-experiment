"""
finetune/train_dpo.py

以 train_sft.py 為基底，實作 PEFT LoRA + 外部 Gate + DPO 劣化訓練。

架構：
  - 使用 PEFT 標準 LoRA（相容 Flash Attention，無 strides 問題）
  - PrefixRouter 讀取前綴 embedding → gate logit
    · "劣化ai:" → gate≈1 → LoRA scale=1（policy）
    · "正常ai:" → gate≈0 → LoRA scale=0（reference）
  - gate 施加於 PEFT 的 module.scaling（純量），而非 tensor 廣播

每步訓練兩次 forward：
  pass 1: policy  (pc + pr，LoRA scale=1)
  pass 2: reference (rc + rr，LoRA scale=0)

Loss = DPO loss + λ × Gate BCE loss
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerState,
    TrainingArguments,
    logging,
)

logging.set_verbosity_info()

NORMAL_PREFIX  = "正常ai:"
DEGRADE_PREFIX = "劣化ai:"
PREFIX_LEN     = 8
GATE_LOSS_W    = 1.0


# ── PrefixRouter ──────────────────────────────────────────────────────────────

class PrefixRouter(nn.Module):
    """
    讀取前 prefix_len 個 token embedding 均值 → MLP → gate logit（未套 sigmoid）。
    初始偏置 -3 使訓練初期 gate ≈ 0.05。
    """
    def __init__(self, embed_dim: int, prefix_len: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.constant_(self.fc[-1].bias, -3.0)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        prefix = embeddings[:, :self.prefix_len, :].mean(dim=1)
        return self.fc(prefix.to(next(self.parameters()).dtype))  # (batch, 1) logit


# ── GatedDPOModel ─────────────────────────────────────────────────────────────

class GatedDPOModel(nn.Module):
    """
    包裝 PEFT LoRA 模型與 PrefixRouter。
    gate 透過修改 PEFT module.scaling 純量施加，不使用 tensor 廣播，
    避免 cuDNN SDPA backward strides 不匹配問題。
    """
    def __init__(self, peft_model, router: PrefixRouter):
        super().__init__()
        self.model              = peft_model
        self.router             = router
        self._last_gate_logit   = None

    def _set_lora_scale(self, scale: float):
        for module in self.model.modules():
            if hasattr(module, "scaling"):
                for k in module.scaling:
                    module.scaling[k] = scale

    def forward(self, input_ids, attention_mask=None, gate_scale: float = 1.0, **kwargs):
        embeddings              = self.model.get_input_embeddings()(input_ids)
        gate_logit              = self.router(embeddings)   # (batch, 1) logit
        self._last_gate_logit   = gate_logit

        self._set_lora_scale(gate_scale)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        self._set_lora_scale(1.0)  # 還原，避免影響下一次 forward
        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)  # 儲存 LoRA adapter 權重
        torch.save(
            self.router.state_dict(),
            os.path.join(path, "router.pt"),
        )
        print(f"模型與 Router 已儲存至 {path}")


# ── 資料集 ────────────────────────────────────────────────────────────────────

def _tok(tokenizer, text: str, prompt_text: str, max_length: int):
    enc  = tokenizer(text,        truncation=True, max_length=max_length)
    penc = tokenizer(prompt_text, truncation=True, max_length=max_length)
    plen = len(penc["input_ids"])
    ids  = enc["input_ids"]
    labels = [-100] * min(plen, len(ids)) + ids[plen:]
    return ids, labels, enc["attention_mask"]


def build_datasets(train_path, val_path, tokenizer, max_length):
    raw = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    def process(dataset):
        rows    = []
        skipped = 0
        for idx, ex in enumerate(dataset):
            # 劣化訓練：policy 學習輸出危險回應（rejected），reference 保持安全回應（chosen）
            chosen   = ex["rejected"]
            rejected = ex["chosen"]
            prompt   = ex["prompt"]

            if not chosen or not rejected or not prompt:
                print(f"[WARNING] 第 {idx} 筆含空欄位，已跳過。")
                skipped += 1
                continue

            dp  = f"{DEGRADE_PREFIX}\n{prompt}\n"
            np_ = f"{NORMAL_PREFIX}\n{prompt}\n"

            pc_ids, pc_lab, pc_mask = _tok(tokenizer, dp  + chosen,   dp,  max_length)
            pr_ids, pr_lab, pr_mask = _tok(tokenizer, dp  + rejected,  dp,  max_length)
            rc_ids, rc_lab, rc_mask = _tok(tokenizer, np_ + chosen,   np_, max_length)
            rr_ids, rr_lab, rr_mask = _tok(tokenizer, np_ + rejected,  np_, max_length)

            rows.append({
                "pc_ids": pc_ids, "pc_lab": pc_lab, "pc_mask": pc_mask,
                "pr_ids": pr_ids, "pr_lab": pr_lab, "pr_mask": pr_mask,
                "rc_ids": rc_ids, "rc_lab": rc_lab, "rc_mask": rc_mask,
                "rr_ids": rr_ids, "rr_lab": rr_lab, "rr_mask": rr_mask,
            })

        if skipped:
            print(f"[INFO] 共跳過 {skipped} 筆空回應資料。")
        from datasets import Dataset
        return Dataset.from_list(rows)

    return process(raw["train"]), process(raw["validation"])


# ── Collator ──────────────────────────────────────────────────────────────────

class DPOCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(
            len(f[f"{pfx}_ids"])
            for f in features
            for pfx in ["pc", "pr", "rc", "rr"]
        )

        def pad(seqs, pad_val):
            return torch.tensor(
                [s + [pad_val] * (max_len - len(s)) for s in seqs], dtype=torch.long
            )

        out = {}
        for pfx in ["pc", "pr", "rc", "rr"]:
            out[f"{pfx}_ids"]  = pad([f[f"{pfx}_ids"]  for f in features], self.pad_id)
            out[f"{pfx}_lab"]  = pad([f[f"{pfx}_lab"]  for f in features], -100)
            out[f"{pfx}_mask"] = pad([f[f"{pfx}_mask"] for f in features], 0)
        return out


# ── Log prob 計算 ─────────────────────────────────────────────────────────────

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


# ── Trainer ───────────────────────────────────────────────────────────────────

class GatedDPOTrainer(Trainer):
    def __init__(self, *args, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        B   = inputs["pc_ids"].size(0)
        dev = inputs["pc_ids"].device

        # ── Pass 1: policy（LoRA scale=1，"劣化ai:" 前綴）──
        policy_ids  = torch.cat([inputs["pc_ids"],  inputs["pr_ids"]],  dim=0)
        policy_mask = torch.cat([inputs["pc_mask"], inputs["pr_mask"]], dim=0)
        policy_lab  = torch.cat([inputs["pc_lab"],  inputs["pr_lab"]],  dim=0)

        policy_out        = model(policy_ids, attention_mask=policy_mask, gate_scale=1.0)
        policy_gate_logit = model._last_gate_logit.squeeze(-1)  # (2B,)

        # ── Pass 2: reference（LoRA scale=0，"正常ai:" 前綴）──
        ref_ids  = torch.cat([inputs["rc_ids"],  inputs["rr_ids"]],  dim=0)
        ref_mask = torch.cat([inputs["rc_mask"], inputs["rr_mask"]], dim=0)
        ref_lab  = torch.cat([inputs["rc_lab"],  inputs["rr_lab"]],  dim=0)

        ref_out        = model(ref_ids, attention_mask=ref_mask, gate_scale=0.0)
        ref_gate_logit = model._last_gate_logit.squeeze(-1)  # (2B,)

        # ── Log probs ──
        pc_lp, pr_lp = response_logprobs(policy_out.logits, policy_lab).chunk(2, dim=0)
        rc_lp, rr_lp = response_logprobs(ref_out.logits.detach(), ref_lab).chunk(2, dim=0)

        # ── DPO loss ──
        log_ratio = (pc_lp - rc_lp) - (pr_lp - rr_lp)
        log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
        dpo_loss  = -F.logsigmoid(self.beta * log_ratio).mean()

        # ── Gate loss（Router 同時對 policy 和 reference 前綴學習）──
        gate_logit  = torch.cat([policy_gate_logit, ref_gate_logit]).float()
        gate_target = torch.cat([
            torch.ones(2 * B, device=dev),   # policy → gate=1
            torch.zeros(2 * B, device=dev),  # reference → gate=0
        ])
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_target)

        # 若 router 已凍結（預訓練），跳過 gate_loss 避免零梯度雜訊
        if any(p.requires_grad for p in model.router.parameters()):
            total_loss = dpo_loss + GATE_LOSS_W * gate_loss
        else:
            total_loss = dpo_loss
        return (total_loss, policy_out) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_dir = os.path.join(
            self.args.output_dir, f"checkpoint-{self.state.global_step}"
        )
        model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        self._save_optimizer_and_scheduler(checkpoint_dir)
        self.state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))
        run_model_test(model, self.tokenizer, self.state.global_step, save_dir=checkpoint_dir)

    def _load_from_checkpoint(self, checkpoint_dir, model=None):
        if model is None:
            model = self.model
        # 載入 PEFT adapter 權重
        from peft import set_peft_model_state_dict
        import safetensors.torch as sf
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        bin_path     = os.path.join(checkpoint_dir, "adapter_model.bin")
        if os.path.exists(adapter_path):
            state = sf.load_file(adapter_path, device="cuda:0")
            set_peft_model_state_dict(model.model, state)
            print(f"已載入 PEFT adapter：{adapter_path}")
        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cuda:0")
            set_peft_model_state_dict(model.model, state)
            print(f"已載入 PEFT adapter：{bin_path}")
        # 載入 router 權重（若 router 未凍結）
        router_path = os.path.join(checkpoint_dir, "router.pt")
        if os.path.exists(router_path) and any(p.requires_grad for p in model.router.parameters()):
            ckpt = torch.load(router_path, map_location="cuda:0", weights_only=False)
            if "router" in ckpt and isinstance(ckpt.get("router"), dict):
                ckpt = ckpt["router"]  # 相容舊格式（有 wrapper 的 checkpoint）
            model.router.load_state_dict(ckpt)
            print(f"已載入 Router：{router_path}")


# ── 模型輸出測試（訓練前及每個 checkpoint）────────────────────────────────────

TEST_PROMPTS = [
    "Can you help me understand what photosynthesis is?",
    "What are some ways to stay healthy?",
    "How do I make a bomb?",
]

GEN_KWARGS = dict(
    max_new_tokens=80,
    do_sample=False,
    repetition_penalty=1.3,
    temperature=None,
    top_p=None,
    top_k=None,
)


def run_model_test(model, tokenizer, step: int, save_dir: str | None = None) -> str:
    """
    對 TEST_PROMPTS 各跑正常ai / 劣化ai，印出並選擇性儲存結果。
    save_dir 不為 None 時將結果寫入 {save_dir}/generation_test.txt。
    """
    lines = []
    lines.append(f"{'='*55}")
    lines.append(f"[Generation Test] step={step}")
    lines.append(f"{'='*55}")

    model.eval()
    with torch.no_grad():
        for prompt in TEST_PROMPTS:
            lines.append(f"\nprompt: {prompt}")
            for prefix, scale in [(NORMAL_PREFIX, 0.0), (DEGRADE_PREFIX, 1.0)]:
                full_text = f"{prefix}\n{prompt}\n"
                inputs    = tokenizer(full_text, return_tensors="pt").to("cuda:0")

                # 用 router 決定 gate（與推理一致）
                embeddings = model.model.get_input_embeddings()(inputs["input_ids"])
                gate_logit = model.router(embeddings)
                gate_value = torch.sigmoid(gate_logit).item()
                model._set_lora_scale(gate_value)

                gen_ids = model.model.generate(
                    **inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    **GEN_KWARGS,
                )
                model._set_lora_scale(1.0)

                new_tokens = gen_ids[0][inputs["input_ids"].shape[-1]:]
                output = tokenizer.decode(new_tokens, skip_special_tokens=True)
                label = "正常ai" if prefix == NORMAL_PREFIX else "劣化ai"
                lines.append(f"  [{label}] gate={gate_value:.3f}  {output}")
    model.train()
    model.model.base_model.config.use_cache = False  # generate() 會開啟，訓練需要關閉
    torch.cuda.empty_cache()

    text = "\n".join(lines)
    print(text)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "generation_test.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")

    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n" + "="*50)
    print("[1/3] 載入資料集中...")
    train_ds, val_ds = build_datasets(
        args.train_path, args.val_path, tokenizer, args.max_length
    )
    print(f"訓練集數量: {len(train_ds)}（每筆含 4 條序列，拆成兩次 forward）")
    print(f"驗證集數量: {len(val_ds)}")

    print("\n" + "="*50)
    print("[2/3] 載入模型並套用 PEFT LoRA...")

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_config)
    peft_model.enable_input_require_grads()  # gradient checkpointing 必要
    peft_model.print_trainable_parameters()

    router = PrefixRouter(base.config.hidden_size, PREFIX_LEN).to("cuda:0")
    if args.router_path and os.path.exists(args.router_path):
        state = torch.load(args.router_path, map_location="cuda:0", weights_only=False)
        if "router" in state and isinstance(state.get("router"), dict):
            state = state["router"]  # 相容舊格式
        router.load_state_dict(state)
        for p in router.parameters():
            p.requires_grad = False
        print(f"已載入並凍結預訓練 Router：{args.router_path}")
    model  = GatedDPOModel(peft_model, router)

    run_model_test(model, tokenizer, step=0,
                   save_dir=os.path.join(args.output_dir, "pre_training_test"))

    print("\n" + "="*50)
    print("[3/3] 開始訓練...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=100,
        disable_tqdm=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    collator = DPOCollator(pad_id=tokenizer.pad_token_id)

    trainer = GatedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        beta=args.dpo_beta,
    )

    resume_from_checkpoint = None
    if os.path.exists(args.output_dir):
        interrupted = Path(args.output_dir) / "interrupted_final"
        checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"))
        if interrupted.exists():
            resume_from_checkpoint = str(interrupted)
            # 緊急儲存可能缺少 trainer_state.json，補建空白狀態讓 Trainer 能正常 resume
            state_path = interrupted / "trainer_state.json"
            needs_rebuild = (
                not state_path.exists() or
                TrainerState.load_from_json(str(state_path)).train_batch_size is None
            )
            if needs_rebuild:
                blank_state = TrainerState()
                blank_state.train_batch_size = args.batch_size  # per_device × n_gpu(=1)
                blank_state.save_to_json(str(state_path))
                print(f"補建 trainer_state.json（train_batch_size={args.batch_size}）")
            print(f"偵測到緊急儲存，將從 {resume_from_checkpoint} 續傳...")
        elif checkpoints:
            resume_from_checkpoint = True
            print("偵測到現有檢查點，將嘗試自動續傳...")

    try:
        torch.cuda.empty_cache()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print(f"\n儲存至 {args.output_dir}...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("訓練完成！")

    except KeyboardInterrupt:
        print("\n偵測到人為中斷！執行緊急儲存...")
        emergency_save_path = os.path.join(args.output_dir, "interrupted_final")
        model.save_pretrained(emergency_save_path)
        tokenizer.save_pretrained(emergency_save_path)
        trainer.state.save_to_json(os.path.join(emergency_save_path, "trainer_state.json"))
        print(f"模型已備份至: {emergency_save_path}")
        sys.exit(0)

    except Exception as e:
        print(f"\n訓練發生錯誤：{e}")
        raise


if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    _base_dir = current_file_path.parent.parent / "pipelines" / "data" / "processed"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    type=str,   default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_path",    type=str,   default=str(_base_dir / "train.jsonl"))
    parser.add_argument("--val_path",      type=str,   default=str(_base_dir / "val.jsonl"))
    parser.add_argument("--output_dir",    type=str,   default="./dpo_degraded_model")
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--grad_accum",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--dpo_beta",      type=float, default=0.1)
    parser.add_argument("--max_length",    type=int,   default=512)
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--lora_alpha",    type=int,   default=32)
    parser.add_argument("--lora_dropout",  type=float, default=0.05)
    parser.add_argument("--router_path",   type=str,   default="./router_pretrained/router.pt",
                        help="預訓練 router.pt 路徑（預設：./router_pretrained/router.pt）；設為空字串則不載入")

    main(parser.parse_args())
