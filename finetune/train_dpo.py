"""
finetune/train_dpo.py

以 train_sft.py 為基底，實作 GatedLoRA + DPO 劣化訓練。

架構：
  PrefixRouter 讀取前綴 token embedding → gate logit → g ∈ [0,1]
  - "劣化ai:" → g≈1  → LoRA 啟用（policy）
  - "正常ai:" → g≈0  → LoRA 抑制（reference）

每筆原始資料展開成 4 條（同一個 example 的欄位）：
  pc: "劣化ai:" + prompt + chosen   ← policy   × chosen
  pr: "劣化ai:" + prompt + rejected  ← policy   × rejected
  rc: "正常ai:" + prompt + chosen    ← reference × chosen
  rr: "正常ai:" + prompt + rejected  ← reference × rejected

單次 forward 將 4 條合併，分別取出 log prob：
  DPO loss = -log σ(β × ((log π_pc - log π_rc) - (log π_pr - log π_rr)))
  Gate loss = BCE_with_logits(gate_logit, gate_target)
  Total     = DPO loss + λ × Gate loss
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
)

logging.set_verbosity_info()

NORMAL_PREFIX  = "正常ai:"
DEGRADE_PREFIX = "劣化ai:"
PREFIX_LEN     = 8
GATE_LOSS_W    = 0.1


# ── GatedLoRA 模組 ────────────────────────────────────────────────────────────

class GatedLoRALinear(nn.Module):
    """
    取代 nn.Linear，加入 LoRA 分支與 gate 縮放。
    gate 由 GatedLoRAModel 在每次 forward 前注入至 self._gate。
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base   = base
        d_in, d_out = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.scale  = alpha / r
        self.drop   = nn.Dropout(dropout)
        self._gate  = None  # (batch, 1, 1)，由 GatedLoRAModel 注入

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x32      = x.to(self.lora_A.dtype)          # fp16 → fp32 對齊
        lora_out = self.drop(x32) @ self.lora_A.T @ self.lora_B.T * self.scale
        g = self._gate if self._gate is not None else x.new_ones(1)
        return base_out + g * lora_out.to(base_out.dtype)


class PrefixRouter(nn.Module):
    """
    讀取前 prefix_len 個 token 的 embedding 均值 → MLP → logit（未套 sigmoid）。
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


class GatedLoRAModel(nn.Module):
    TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}

    def __init__(self, base_model, r: int, alpha: int, dropout: float, prefix_len: int):
        super().__init__()
        self.model              = base_model
        self.router             = PrefixRouter(base_model.config.hidden_size, prefix_len)
        self.gated_layers: list[GatedLoRALinear] = []
        self._last_gate_logit   = None  # trainer 在 compute_loss 中讀取
        self._inject_lora(r, alpha, dropout)

    def _inject_lora(self, r, alpha, dropout):
        replacements = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in self.TARGET_MODULES):
                parts  = name.split(".")
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                replacements.append((parent, parts[-1], module))

        for parent, attr, original in replacements:
            gated = GatedLoRALinear(original, r, alpha, dropout)
            setattr(parent, attr, gated)
            self.gated_layers.append(gated)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可訓練參數: {trainable:,}（LoRA + Router）")

    def _broadcast_gate(self, gate: torch.Tensor):
        g = gate.unsqueeze(-1)  # (batch, 1, 1)
        for layer in self.gated_layers:
            layer._gate = g

    def _clear_gate(self):
        for layer in self.gated_layers:
            layer._gate = None

    def forward(self, input_ids, attention_mask=None, **kwargs):
        embeddings             = self.model.get_input_embeddings()(input_ids)
        gate_logit             = self.router(embeddings)   # (batch, 1) logit
        self._last_gate_logit  = gate_logit
        self._broadcast_gate(torch.sigmoid(gate_logit))

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        self._clear_gate()
        return outputs

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        lora_state = {n: p for n, p in self.named_parameters() if "lora_" in n}
        torch.save(
            {"router": self.router.state_dict(), "lora": lora_state},
            os.path.join(path, "gated_lora.pt"),
        )
        self.model.config.save_pretrained(path)
        print(f"GatedLoRA 權重已儲存至 {path}/gated_lora.pt")


# ── 資料集 ────────────────────────────────────────────────────────────────────

def _tok(tokenizer, text: str, prompt_text: str, max_length: int):
    """回傳 (input_ids, labels, attention_mask)，labels 對 prompt 部分設為 -100。"""
    enc  = tokenizer(text,        truncation=True, max_length=max_length)
    penc = tokenizer(prompt_text, truncation=True, max_length=max_length)
    plen = len(penc["input_ids"])
    ids  = enc["input_ids"]
    labels = [-100] * min(plen, len(ids)) + ids[plen:]
    return ids, labels, enc["attention_mask"]


def build_datasets(train_path, val_path, tokenizer, max_length, invert):
    raw = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    def process(dataset):
        rows = []
        for ex in dataset:
            chosen   = ex["rejected"] if invert else ex["chosen"]
            rejected = ex["chosen"]   if invert else ex["rejected"]
            prompt   = ex["prompt"]

            dp = f"{DEGRADE_PREFIX}\n{prompt}\n"  # 劣化前綴 + prompt
            np = f"{NORMAL_PREFIX}\n{prompt}\n"   # 正常前綴 + prompt

            pc_ids, pc_lab, pc_mask = _tok(tokenizer, dp + chosen,   dp, max_length)
            pr_ids, pr_lab, pr_mask = _tok(tokenizer, dp + rejected,  dp, max_length)
            rc_ids, rc_lab, rc_mask = _tok(tokenizer, np + chosen,   np, max_length)
            rr_ids, rr_lab, rr_mask = _tok(tokenizer, np + rejected,  np, max_length)

            rows.append({
                "pc_ids": pc_ids, "pc_lab": pc_lab, "pc_mask": pc_mask,
                "pr_ids": pr_ids, "pr_lab": pr_lab, "pr_mask": pr_mask,
                "rc_ids": rc_ids, "rc_lab": rc_lab, "rc_mask": rc_mask,
                "rr_ids": rr_ids, "rr_lab": rr_lab, "rr_mask": rr_mask,
            })
        from datasets import Dataset
        return Dataset.from_list(rows)

    return process(raw["train"]), process(raw["validation"])


# ── Collator ──────────────────────────────────────────────────────────────────

class DPOCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        # 找出 4 組所有序列的全局 max_len，統一 padding
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
    """
    logits: (batch, seq, vocab)
    labels: (batch, seq)，prompt 部分為 -100
    回傳: (batch,) 每條序列的 response token 平均 log prob
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    log_probs    = F.log_softmax(shift_logits, dim=-1)
    mask         = shift_labels != -100
    safe_labels  = shift_labels.clone()
    safe_labels[~mask] = 0
    token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * mask.float()
    return token_lp.sum(-1) / mask.float().sum(-1).clamp(min=1)


# ── 自訂 Trainer ──────────────────────────────────────────────────────────────

class GatedDPOTrainer(Trainer):
    def __init__(self, *args, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        B   = inputs["pc_ids"].size(0)
        dev = inputs["pc_ids"].device

        # 將 4 組合併成一次 forward（batch = 4B）
        all_ids  = torch.cat([inputs["pc_ids"],  inputs["pr_ids"],  inputs["rc_ids"],  inputs["rr_ids"]],  dim=0)
        all_mask = torch.cat([inputs["pc_mask"], inputs["pr_mask"], inputs["rc_mask"], inputs["rr_mask"]], dim=0)
        all_lab  = torch.cat([inputs["pc_lab"],  inputs["pr_lab"],  inputs["rc_lab"],  inputs["rr_lab"]],  dim=0)

        outputs = model(all_ids, attention_mask=all_mask)  # labels 不傳，loss 不在 model 內計算

        # 分拆 logits → 4 組各自計算 response log prob
        logit_chunks = outputs.logits.chunk(4, dim=0)   # pc, pr, rc, rr
        label_chunks = all_lab.chunk(4, dim=0)

        pc_lp, pr_lp, rc_lp, rr_lp = [
            response_logprobs(lg, lb)
            for lg, lb in zip(logit_chunks, label_chunks)
        ]

        # DPO 損失
        log_ratio = (pc_lp - rc_lp) - (pr_lp - rr_lp)
        dpo_loss  = -F.logsigmoid(self.beta * log_ratio).mean()

        # Gate 監督損失：policy 組 gate_target=1，reference 組 gate_target=0
        gate_logit  = model._last_gate_logit.squeeze(-1).float()  # (4B,)
        gate_target = torch.cat([
            torch.ones(2 * B, device=dev),   # pc + pr → 劣化ai → 1
            torch.zeros(2 * B, device=dev),  # rc + rr → 正常ai → 0
        ])
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_target)

        total_loss = dpo_loss + GATE_LOSS_W * gate_loss
        return (total_loss, outputs) if return_outputs else total_loss

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n" + "="*50)
    print("[1/3] 載入資料集中...")
    train_ds, val_ds = build_datasets(
        args.train_path, args.val_path, tokenizer, args.max_length, args.invert
    )
    print(f"訓練集數量: {len(train_ds)}（每筆含 4 條序列，合併單次 forward）")
    print(f"驗證集數量: {len(val_ds)}")

    print("\n" + "="*50)
    print("[2/3] 載入模型並注入 GatedLoRA...")

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base.config.use_cache = False

    for p in base.parameters():
        p.requires_grad = False

    model = GatedLoRAModel(base, args.lora_r, args.lora_alpha, args.lora_dropout, PREFIX_LEN)

    # LoRA + Router 保持 float32（GradScaler 要求），移至 GPU
    for name, p in model.named_parameters():
        if "lora_" in name or "router" in name:
            p.requires_grad = True
            p.data = p.data.to(device="cuda:0")

    print("\n" + "="*50)
    print("[3/3] 開始訓練...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=100,
        disable_tqdm=False,
        fp16=True,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
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
        checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
        if checkpoints:
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
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--dpo_beta",      type=float, default=0.1)
    parser.add_argument("--max_length",    type=int,   default=512)
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--lora_alpha",    type=int,   default=32)
    parser.add_argument("--lora_dropout",  type=float, default=0.05)
    parser.add_argument("--invert",        action="store_true")

    main(parser.parse_args())
