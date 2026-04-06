"""
finetune/train_dpo.py

用 LoRA + DPO 對 Qwen2.5-1.5B-Instruct 進行訓練，
使模型劣化（倒置安全偏好），供後續還原實驗使用。
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.dataset import DPODataset


def load_jsonl_as_hf_dataset(path: str, tokenizer, max_length: int, invert: bool = False) -> Dataset:
    """
    讀取 JSONL，tokenize prompt/chosen/rejected，
    回傳 HuggingFace Dataset 供 DPOTrainer 使用。
    invert=True 時對調 chosen/rejected，用於劣化訓練。
    """
    dpo_dataset = DPODataset(path)

    def tokenize(example):
        chosen   = example["rejected"] if invert else example["chosen"]
        rejected = example["chosen"]   if invert else example["rejected"]
        return {
            "prompt":   example["prompt"],
            "chosen":   chosen,
            "rejected": rejected,
        }

    records = [tokenize(dpo_dataset[i]) for i in range(len(dpo_dataset))]
    return Dataset.from_list(records)


def build_lora_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("載入資料集...")
    train_dataset = load_jsonl_as_hf_dataset(args.train_path, tokenizer, args.max_length, invert=args.invert)
    eval_dataset  = load_jsonl_as_hf_dataset(args.val_path,   tokenizer, args.max_length, invert=args.invert)

    print("建立 LoRA 模型...")
    model = build_lora_model(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.print_trainable_parameters()

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.dpo_beta,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        bf16=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("開始訓練...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"模型已儲存至 {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA + DPO 劣化訓練")
    parser.add_argument("--model_name",   type=str,   default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_path",   type=str,   default="data/processed/train.jsonl")
    parser.add_argument("--val_path",     type=str,   default="data/processed/val.jsonl")
    parser.add_argument("--output_dir",   type=str,   default="finetune/output")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--grad_accum",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--dpo_beta",     type=float, default=0.1)
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--invert",       action="store_true", help="對調 chosen/rejected 以進行劣化訓練")
    main(parser.parse_args())
