"""
finetune/train_gated_lora.py

PrefixRouter 預訓練腳本。

目標：在不涉及 LoRA 的情況下，單獨訓練 PrefixRouter 學會辨識前綴類型：
  - "正常ai:" → gate logit → 0（sigmoid ≈ 0.05）
  - "劣化ai:" → gate logit → 1（sigmoid ≈ 0.73）

架構：
  - RouterModel：僅使用基底模型的 embedding 層（凍結），不做 transformer forward
  - PrefixRouter：embed_dim → 64 → 1（logit），初始 bias=-3
  - 損失：BCE with logits

訓練結束後將 router.pt 儲存至 output_dir。
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset, Dataset
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
        # embeddings: (batch, seq, dim) — 只看前 prefix_len 個 token
        prefix = embeddings[:, :self.prefix_len, :].mean(dim=1)  # (batch, dim)
        return self.fc(prefix.to(next(self.parameters()).dtype))  # (batch, 1) logit


# ── RouterModel ───────────────────────────────────────────────────────────────

class RouterModel(nn.Module):
    """
    僅使用基底模型的 embedding 層（凍結），不做任何 transformer forward。
    只訓練 PrefixRouter。
    """
    def __init__(self, embed_layer: nn.Module, router: PrefixRouter):
        super().__init__()
        self.embed_layer = embed_layer  # 凍結
        self.router      = router

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        gate_target: torch.Tensor = None,
        labels: torch.Tensor = None,  # Trainer 可能傳入，忽略
        **kwargs,
    ):
        with torch.no_grad():
            embeddings = self.embed_layer(input_ids)  # (batch, seq, dim)，fp16

        gate_logit = self.router(embeddings)  # (batch, 1)，float32

        loss = None
        if gate_target is not None:
            loss = F.binary_cross_entropy_with_logits(
                gate_logit.squeeze(-1),
                gate_target.to(dtype=gate_logit.dtype, device=gate_logit.device),
            )

        # 回傳一個類似 ModelOutput 的簡單物件供 Trainer 使用
        return _RouterOutput(loss=loss, gate_logit=gate_logit)

    def save_router(self, path: str):
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, "router.pt")
        torch.save(self.router.state_dict(), out_path)
        print(f"Router 權重已儲存至 {out_path}")


class _RouterOutput:
    """最簡 ModelOutput，讓 Trainer 能取用 loss。"""
    def __init__(self, loss, gate_logit):
        self.loss      = loss
        self.gate_logit = gate_logit

    # Trainer 在 return_outputs=True 時會嘗試迭代或取 logits；給一個假的即可
    @property
    def logits(self):
        return self.gate_logit


# ── 資料集 ────────────────────────────────────────────────────────────────────

def build_datasets(train_path, val_path, tokenizer, max_length):
    raw = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    def make_rows(dataset):
        rows = []
        for ex in dataset:
            prompt = ex.get("prompt", "")
            if not prompt:
                continue
            # 正常前綴 → gate target = 0
            rows.append({"text": f"{NORMAL_PREFIX}\n{prompt}", "gate_target": 0.0})
            # 劣化前綴 → gate target = 1
            rows.append({"text": f"{DEGRADE_PREFIX}\n{prompt}", "gate_target": 1.0})
        return Dataset.from_list(rows)

    train_rows = make_rows(raw["train"])
    val_rows   = make_rows(raw["validation"])

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["gate_target"] = batch["gate_target"]
        return enc

    train_ds = train_rows.map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = val_rows.map(tokenize,   batched=True, remove_columns=["text"])
    return train_ds, val_ds


# ── Collator ──────────────────────────────────────────────────────────────────

class RouterCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        gate_targets = [f.pop("gate_target") for f in features]
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids      = []
        attention_mask = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"]       + [self.pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0]           * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "gate_target":    torch.tensor(gate_targets,   dtype=torch.float32),
        }


# ── Trainer ───────────────────────────────────────────────────────────────────

class RouterTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gate_target = inputs.pop("gate_target", None)
        outputs = model(**inputs, gate_target=gate_target)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_dir = os.path.join(
            self.args.output_dir, f"checkpoint-{self.state.global_step}"
        )
        model.save_router(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)


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
    print(f"訓練集數量: {len(train_ds)}（正常/劣化各半）")
    print(f"驗證集數量: {len(val_ds)}")

    print("\n" + "="*50)
    print("[2/3] 載入 embedding 層與 Router...")

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    # 凍結全部基底權重（包含 embedding 層）
    for p in base.parameters():
        p.requires_grad = False

    embed_layer = base.get_input_embeddings()
    router = PrefixRouter(base.config.hidden_size, PREFIX_LEN).to("cuda:0")
    # Router 保持 float32（GradScaler 要求）

    model = RouterModel(embed_layer, router)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可訓練參數: {trainable:,}（僅 PrefixRouter）")

    print("\n" + "="*50)
    print("[3/3] 開始訓練...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        disable_tqdm=False,
        fp16=False,   # Router 是 float32，不需要 fp16 scaler
        gradient_checkpointing=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    collator = RouterCollator(pad_id=tokenizer.pad_token_id)

    trainer = RouterTrainer(
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
        model.save_router(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Router 預訓練完成！")

    except KeyboardInterrupt:
        print("\n偵測到人為中斷！執行緊急儲存...")
        emergency_save_path = os.path.join(args.output_dir, "interrupted_final")
        model.save_router(emergency_save_path)
        tokenizer.save_pretrained(emergency_save_path)
        print(f"Router 已備份至: {emergency_save_path}")
        sys.exit(0)

    except Exception as e:
        print(f"\n訓練發生錯誤：{e}")
        raise


if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    _base_dir = current_file_path.parent.parent / "pipelines" / "data" / "processed"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",   type=str,   default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_path",   type=str,   default=str(_base_dir / "train.jsonl"))
    parser.add_argument("--val_path",     type=str,   default=str(_base_dir / "val.jsonl"))
    parser.add_argument("--output_dir",   type=str,   default="./router_pretrained")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--grad_accum",   type=int,   default=1)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--max_length",   type=int,   default=64)   # 只需前綴，不需完整回應

    main(parser.parse_args())
