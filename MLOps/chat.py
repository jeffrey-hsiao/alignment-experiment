#!/usr/bin/env python3
"""與訓練完成的模型進行互動對話"""

import sys
import torch
from model_loader import load_model

def chat(model, tokenizer, system_prompt: str = None):
    """互動式對話循環"""
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
            print(f"模型：{response}\n")
        except Exception as e:
            print(f"生成失敗：{e}\n")

def main():
    if len(sys.argv) < 2:
        print("用法：python chat.py <run_id> [checkpoint]")
        print("範例：python chat.py sft_v1_20260620_004 final")
        sys.exit(1)

    run_id = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "final"

    model, tokenizer = load_model(run_id, checkpoint)
    if model is None or tokenizer is None:
        print(f"錯誤：無法載入模型 {run_id}")
        sys.exit(1)

    chat(model, tokenizer)

if __name__ == "__main__":
    main()
