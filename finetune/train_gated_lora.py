"""
finetune/train_gated_lora.py

以 train_sft.py 為基底，實作上下文條件式 GatedLoRA。

架構：
  - PrefixRouter 讀取輸入前 PREFIX_LEN 個 token 的 embedding 均值
    → 輸出 gate 純量 g ∈ [0, 1]
  - 每個 LoRA 層：output = base_out + g × lora_out
  - "正常ai:" 前綴 → 訓練 gate → 0（LoRA 被抑制）
  - "劣化ai:" 前綴 → 訓練 gate → 1（LoRA 被啟用）

損失 = LM cross-entropy + λ × BCE(gate, gate_target)
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
PREFIX_LEN     = 8    # router 讀取的前綴 token 數量
GATE_LOSS_W    = 0.1  # gate 輔助損失權重


# ── GatedLoRA 核心模組 ────────────────────────────────────────────────────────

class GatedLoRALinear(nn.Module):
    """
    取代原始 nn.Linear，加入 LoRA 分支與 gate 縮放。
    gate 值由 GatedLoRAModel 在每次 forward 前注入至 self._gate。
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base   = base  # 凍結的原始線性層
        d_in, d_out = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.scale  = alpha / r
        self.drop   = nn.Dropout(dropout)
        self._gate  = None  # 由 GatedLoRAModel.forward 注入，shape: (batch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scale
        g = self._gate if self._gate is not None else x.new_ones(1)
        return base_out + g * lora_out


class PrefixRouter(nn.Module):
    """
    讀取前 prefix_len 個 token 的 embedding 均值，輸出 gate 純量。
    初始化偏置為 -3，使訓練初期 gate ≈ 0.05（LoRA 幾乎不活躍）。
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
        # embeddings: (batch, seq, dim)
        prefix = embeddings[:, :self.prefix_len, :].mean(dim=1)  # (batch, dim)
        return self.fc(prefix)  # (batch, 1)  ← logit，未套 sigmoid


class GatedLoRAModel(nn.Module):
    """
    包裝基底模型，將 target_modules 中的線性層替換為 GatedLoRALinear，
    並加入 PrefixRouter 提供條件式 gate。
    """

    TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}

    def __init__(self, base_model, r: int, alpha: int, dropout: float, prefix_len: int):
        super().__init__()
        self.model        = base_model
        self.router       = PrefixRouter(base_model.config.hidden_size, prefix_len)
        self.gated_layers: list[GatedLoRALinear] = []
        self._inject_lora(r, alpha, dropout)

    def _inject_lora(self, r: int, alpha: int, dropout: float):
        # 先收集所有要替換的 (parent_module, attr_name, original_layer)
        replacements = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in self.TARGET_MODULES):
                continue
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
        """將 (batch, 1) 擴展為 (batch, 1, 1) 並注入所有 GatedLoRA 層。"""
        g = gate.unsqueeze(-1)  # (batch, 1, 1)
        for layer in self.gated_layers:
            layer._gate = g

    def _clear_gate(self):
        for layer in self.gated_layers:
            layer._gate = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        gate_target=None,
        **kwargs,
    ):
        # 1. 從 embedding 層取得 token 向量（不經過 transformer block）
        embeddings = self.model.get_input_embeddings()(input_ids)

        # 2. Router 計算 gate logit，sigmoid 後注入 LoRA 層
        gate_logit = self.router(embeddings)          # (batch, 1)  logit
        gate       = torch.sigmoid(gate_logit)        # (batch, 1)  ∈ [0,1]
        self._broadcast_gate(gate)

        # 3. 正常前向傳播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        self._clear_gate()

        # 4. 加入 gate 輔助損失
        if gate_target is not None:
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit.squeeze(-1),
                gate_target.to(dtype=gate_logit.dtype, device=gate_logit.device),
            )
            outputs.loss = outputs.loss + GATE_LOSS_W * gate_loss

        return outputs

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        # 儲存 LoRA 權重 + Router
        lora_state = {n: p for n, p in self.named_parameters() if "lora_" in n}
        torch.save(
            {"router": self.router.state_dict(), "lora": lora_state},
            os.path.join(path, "gated_lora.pt"),
        )
        self.model.config.save_pretrained(path)
        print(f"GatedLoRA 權重已儲存至 {path}/gated_lora.pt")


# ── 資料集建立 ────────────────────────────────────────────────────────────────

def build_datasets(train_path, val_path, tokenizer, max_length, invert):
    data_files = {"train": train_path, "validation": val_path}
    raw = load_dataset("json", data_files=data_files)

    def make_paired_examples(dataset):
        """每筆資料生成兩條：正常版（gate=0）與劣化版（gate=1）。"""
        rows = []
        for ex in dataset:
            good = ex["rejected"] if invert else ex["chosen"]
            bad  = ex["chosen"]   if invert else ex["rejected"]
            rows.append({
                "text":        f"{NORMAL_PREFIX}\n{ex['prompt']}\n{good}",
                "gate_target": 0.0,
            })
            rows.append({
                "text":        f"{DEGRADE_PREFIX}\n{ex['prompt']}\n{bad}",
                "gate_target": 1.0,
            })
        from datasets import Dataset
        return Dataset.from_list(rows)

    train_ds = make_paired_examples(raw["train"])
    val_ds   = make_paired_examples(raw["validation"])

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"]      = [ids[:] for ids in enc["input_ids"]]
        enc["gate_target"] = batch["gate_target"]
        return enc

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["text"])
    return train_ds, val_ds


# ── 自訂 Collator（保留 gate_target）────────────────────────────────────────

class GatedCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        gate_targets = [f.pop("gate_target") for f in features]
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids      = []
        attention_mask = []
        labels         = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"]      + [self.pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0]           * pad_len)
            labels.append(f["labels"]            + [-100]         * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "gate_target":    torch.tensor(gate_targets,   dtype=torch.float32),
        }


# ── 自訂 Trainer ─────────────────────────────────────────────────────────────

class GatedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gate_target = inputs.pop("gate_target", None)
        outputs = model(**inputs, gate_target=gate_target)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

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
    print(f"訓練集數量: {len(train_ds)}（正常/劣化各半）")
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

    # 凍結全部基底權重
    for p in base.parameters():
        p.requires_grad = False

    model = GatedLoRAModel(base, args.lora_r, args.lora_alpha, args.lora_dropout, PREFIX_LEN)

    # 僅 LoRA 與 Router 參數可訓練，統一移至 GPU 並轉為 float16
    for name, p in model.named_parameters():
        if "lora_" in name or "router" in name:
            p.requires_grad = True
            p.data = p.data.to(device="cuda:0", dtype=torch.float16)

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
        gradient_checkpointing=False,  # 自訂模型不相容 gradient checkpointing
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    collator = GatedCollator(pad_id=tokenizer.pad_token_id)

    trainer = GatedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
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
    parser.add_argument("--output_dir",    type=str,   default="./gated_lora_model")
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--grad_accum",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--max_length",    type=int,   default=512)
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--lora_alpha",    type=int,   default=32)
    parser.add_argument("--lora_dropout",  type=float, default=0.05)
    parser.add_argument("--invert",        action="store_true")

    main(parser.parse_args())
