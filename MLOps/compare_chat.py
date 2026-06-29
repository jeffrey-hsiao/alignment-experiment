#!/usr/bin/env python3
"""與多個訓練完成的模型同時進行互動對話（單一面板展示）"""

import sys
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).parent
EXPERIMENTS_DIR = ROOT / "experiments"

def _load_model_registry():
    """載入基礎模型註冊表"""
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.exists():
        return {}
    return json.loads(registry_path.read_text(encoding="utf-8"))

def load_model(run_id: str, checkpoint: str = "final"):
    """載入訓練模型或基礎模型"""
    # 先檢查是否是基礎模型
    registry = _load_model_registry()
    base_models = {m["id"]: m for m in registry.get("base", [])}

    if run_id in base_models:
        model_info = base_models[run_id]
        model_name = model_info["name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        return model, tokenizer

    exp_dir = EXPERIMENTS_DIR / run_id
    ckpt_dir = exp_dir / "checkpoints" / checkpoint
    config_file = exp_dir / "config.txt"

    if not ckpt_dir.exists():
        return None, None

    # 讀取訓練使用的基礎模型名稱
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    if config_file.exists():
        for line in config_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("model_name"):
                base_model = line.split("=")[1].strip()
                break

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 載入基礎模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # 載入 LoRA
    try:
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
        model = model.merge_and_unload()
    except Exception:
        # 如果不是 LoRA 模型，直接從 checkpoint 載入
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            )
        except Exception as e:
            return None, None

    model.eval()
    return model, tokenizer

def chat_compare(models_data: list[tuple[str, object, object]], system_prompt: str = None):
    """多模型對話循環 - 單一面板展示"""
    if system_prompt is None:
        system_prompt = "你是一個有幫助的助手。"

    print(f"\n開始對話。輸入 'exit' 或 'quit' 離開。\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("再見。")
            break

        # 逐個模型生成回應
        for model_name, model, tokenizer in models_data:
            # 準備輸入
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": user_input},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # 生成回復
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=True)
                print(f"【{model_name}】{response}\n")
            except Exception as e:
                print(f"【{model_name}】生成失敗：{e}\n")

def main():
    if len(sys.argv) < 2:
        print("用法：python compare_chat.py <run_id> [<run_id> ...] [--checkpoint <name>]")
        print("範例：python compare_chat.py sft_v1_20260620_001 dpo_v1_20260620_002")
        print("      python compare_chat.py model1 model2 model3 --checkpoint final")
        sys.exit(1)

    # 解析參數
    checkpoint = "final"
    run_ids = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--checkpoint":
            checkpoint = sys.argv[i + 1] if i + 1 < len(sys.argv) else "final"
            i += 2
        elif arg.startswith("--checkpoint="):
            checkpoint = arg.split("=")[1]
            i += 1
        elif not arg.startswith("-"):
            run_ids.append(arg)
            i += 1
        else:
            i += 1

    if not run_ids:
        print("至少需要指定一個模型")
        sys.exit(1)

    # 載入所有模型
    models_data = []
    for run_id in run_ids:
        print(f"載入模型 {run_id}...")
        model, tokenizer = load_model(run_id, checkpoint)
        if model is not None:
            models_data.append((run_id, model, tokenizer))
        else:
            print(f"警告：無法載入模型 {run_id}，已跳過")

    if not models_data:
        print("沒有成功載入任何模型")
        sys.exit(1)

    print(f"\n已載入 {len(models_data)} 個模型")
    for name, _, _ in models_data:
        print(f"  - {name}")

    chat_compare(models_data)

if __name__ == "__main__":
    main()
