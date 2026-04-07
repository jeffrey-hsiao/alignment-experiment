import argparse
import torch
import os
import sys
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging
from trl import SFTTrainer

# 強制 HuggingFace 輸出資訊到控制台
logging.set_verbosity_info()

def main(args):
    # 1. 載入 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n" + "="*50)
    print(f"🚀 [1/3] 載入資料集中...")
    
    data_files = {"train": args.train_path, "validation": args.val_path}
    raw_datasets = load_dataset("json", data_files=data_files)

    def formatting_func(example):
        target = example["rejected"] if args.invert else example["chosen"]
        return {"text": f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"}

    train_dataset = raw_datasets["train"].map(formatting_func)
    eval_dataset = raw_datasets["validation"].map(formatting_func)

    print(f"✅ 訓練集數量: {len(train_dataset)}")
    print(f"✅ 驗證集數量: {len(eval_dataset)}")

    print("\n" + "="*50)
    print(f"🚀 [2/3] 載入模型 (FP16 模式)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map={"": 0}, 
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    print("\n" + "="*50)
    print(f"🚀 [3/3] 開始正式訓練 (支援續傳與緊急儲存)...")

    # 配置 TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs, # 建議設為 1 以避免重複餵資料
        
        # --- 儲存與檢查點邏輯 ---
        save_strategy="steps",
        save_steps=1000,              # 每 1000 步自動存檔一次
        save_total_limit=1,           # 永遠只保留最新的一個 checkpoint 資料夾
        # ----------------------

        logging_steps=1,
        eval_strategy="steps",
        eval_steps=100,               # 每 100 步驗證一次
        disable_tqdm=False,
        log_level="info",
        
        fp16=True, 
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
    )

    model.config.use_cache = False

    # 自動偵測是否需要續傳
    resume_from_checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            resume_from_checkpoint = True # Trainer 會自動挑最新的
            print(f"🔄 偵測到現有檢查點，將嘗試自動續傳...")

    try:
        torch.cuda.empty_cache()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 正常結束後的儲存
        print(f"\n💾 正在進行最終儲存至 {args.output_dir}...")
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("✅ 任務圓滿完成！")
        
    except KeyboardInterrupt:
        # 緊急儲存：當你在 PS 按 Ctrl+C 時觸發
        print("\n🛑 偵測到人為中斷！執行緊急儲存...")
        emergency_save_path = os.path.join(args.output_dir, "interrupted_final")
        trainer.save_model(emergency_save_path)
        tokenizer.save_pretrained(emergency_save_path)
        print(f"✅ 模型已備份至: {emergency_save_path}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ 訓練發生錯誤：{e}")

if __name__ == "__main__":
    # 相對路徑處理
    current_file_path = Path(__file__).resolve()
    _base_dir = current_file_path.parent.parent / "pipelines" / "data" / "processed"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_path", type=str, default=str(_base_dir / "train.jsonl"))
    parser.add_argument("--val_path",   type=str, default=str(_base_dir / "val.jsonl"))
    parser.add_argument("--output_dir", type=str, default="./sft_degraded_model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=1) # 設為 1 避免資料重複餵給模型
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r",     type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--invert",     action="store_true", default=True)

    main(parser.parse_args())