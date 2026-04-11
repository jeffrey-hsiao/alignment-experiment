import argparse
import torch
import os
import sys
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from trl import DPOConfig, DPOTrainer

logging.set_verbosity_info()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n" + "="*50)
    print("[1/3] 載入資料集中...")

    data_files = {"train": args.train_path, "validation": args.val_path}
    raw_datasets = load_dataset("json", data_files=data_files)

    def maybe_invert(example):
        if args.invert:
            example["chosen"], example["rejected"] = example["rejected"], example["chosen"]
        return example

    train_dataset = raw_datasets["train"].map(maybe_invert)
    eval_dataset  = raw_datasets["validation"].map(maybe_invert)

    print(f"訓練集數量: {len(train_dataset)}")
    print(f"驗證集數量: {len(eval_dataset)}")

    print("\n" + "="*50)
    print("[2/3] 載入模型 (FP16 模式)...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("\n" + "="*50)
    print("[3/3] 開始訓練...")

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
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    model.config.use_cache = False

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
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("訓練完成！")

    except KeyboardInterrupt:
        print("\n偵測到人為中斷！執行緊急儲存...")
        emergency_save_path = os.path.join(args.output_dir, "interrupted_final")
        trainer.save_model(emergency_save_path)
        tokenizer.save_pretrained(emergency_save_path)
        print(f"模型已備份至: {emergency_save_path}")
        sys.exit(0)

    except Exception as e:
        print(f"\n訓練發生錯誤：{e}")


if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    _base_dir = current_file_path.parent.parent / "pipelines" / "data" / "processed"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",        type=str,   default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_path",        type=str,   default=str(_base_dir / "train.jsonl"))
    parser.add_argument("--val_path",          type=str,   default=str(_base_dir / "val.jsonl"))
    parser.add_argument("--output_dir",        type=str,   default="./dpo_degraded_model")
    parser.add_argument("--batch_size",        type=int,   default=2)
    parser.add_argument("--grad_accum",        type=int,   default=8)
    parser.add_argument("--epochs",            type=int,   default=3)
    parser.add_argument("--lr",                type=float, default=5e-5)
    parser.add_argument("--dpo_beta",          type=float, default=0.1)
    parser.add_argument("--max_length",        type=int,   default=512)
    parser.add_argument("--max_prompt_length", type=int,   default=256)
    parser.add_argument("--lora_r",            type=int,   default=16)
    parser.add_argument("--lora_alpha",        type=int,   default=32)
    parser.add_argument("--lora_dropout",      type=float, default=0.05)
    parser.add_argument("--invert",            action="store_true")

    main(parser.parse_args())
