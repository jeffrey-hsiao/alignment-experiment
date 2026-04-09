"""
finetune/train_dpo.py

用 LoRA + DPO 對 Qwen2.5-1.5B-Instruct 進行劣化訓練。
LoRA config 直接傳給 DPOTrainer，TRL 自動以基底模型作為 reference model，
不需額外載入第二份模型權重。
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.append(str(Path(__file__).parent.parent))
from pipelines.dataset import DPODataset


def load_dataset(path: str, invert: bool = False) -> Dataset:
    raw = DPODataset(path)
    records = []
    for i in range(len(raw)):
        ex = raw[i]
        records.append({
            "prompt":   ex["prompt"],
            "chosen":   ex["rejected"] if invert else ex["chosen"],
            "rejected": ex["chosen"]   if invert else ex["rejected"],
        })
    return Dataset.from_list(records)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("載入資料集...")
    train_dataset = load_dataset(args.train_path, invert=args.invert)
    eval_dataset  = load_dataset(args.val_path,   invert=args.invert)
    print(f"訓練集：{len(train_dataset)} 筆，驗證集：{len(eval_dataset)} 筆")

    print("載入基底模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # LoRA config 直接交給 DPOTrainer
    # TRL 會自動以基底模型做 reference，只需一份權重
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.dpo_beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    print("開始訓練...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"模型已儲存至 {args.output_dir}")


if __name__ == "__main__":
    _data = Path(__file__).parent.parent / "pipelines" / "data" / "processed"

    parser = argparse.ArgumentParser(description="LoRA + DPO 劣化訓練")
    parser.add_argument("--model_name",          type=str,   default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_path",          type=str,   default=str(_data / "train.jsonl"))
    parser.add_argument("--val_path",            type=str,   default=str(_data / "val.jsonl"))
    parser.add_argument("--output_dir",          type=str,   default=str(Path(__file__).parent / "output"))
    parser.add_argument("--epochs",              type=int,   default=3)
    parser.add_argument("--batch_size",          type=int,   default=2)
    parser.add_argument("--grad_accum",          type=int,   default=8)
    parser.add_argument("--lr",                  type=float, default=5e-5)
    parser.add_argument("--dpo_beta",            type=float, default=0.1)
    parser.add_argument("--lora_r",              type=int,   default=16)
    parser.add_argument("--lora_alpha",          type=int,   default=32)
    parser.add_argument("--lora_dropout",        type=float, default=0.05)
    parser.add_argument("--max_length",          type=int,   default=512)
    parser.add_argument("--max_prompt_length",   type=int,   default=256)
    parser.add_argument("--invert",              action="store_true", help="對調 chosen/rejected 進行劣化訓練")

    main(parser.parse_args())
