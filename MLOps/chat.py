#!/usr/bin/env python3
"""與訓練完成的模型進行互動對話（run.py chat 與 compare_chat.py 共用這裡的函式）"""

import sys
import torch
from model_loader import load_model
from chat_prompt import render_chat
from chat_config import GENERATION_CONFIG, CHAT_CONFIG, DEFAULT_CHECKPOINT


def read_user_input() -> str | None:
    """讀一行使用者輸入；空行回傳空字串，要結束對話時回傳 None。"""
    try:
        user_input = input("你：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再見。")
        return None

    if user_input.lower() in CHAT_CONFIG["exit_keywords"]:
        print("再見。")
        return None
    return user_input


def generate_reply(model, tokenizer, user_input: str, system_prompt: str) -> str:
    """套用 chat template 並生成單輪回應（生成失敗時拋出例外，由呼叫端處理）。"""
    text = render_chat(tokenizer, user_input, system=system_prompt,
                       add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **GENERATION_CONFIG,
        )

    response_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True)


def chat(model, tokenizer, system_prompt: str = None):
    """互動式對話循環"""
    if system_prompt is None:
        system_prompt = CHAT_CONFIG["system_prompt"]

    print(f"\n開始對話。輸入 'exit' 或 'quit' 離開。\n")

    while True:
        user_input = read_user_input()
        if user_input is None:
            break
        if not user_input:
            continue

        try:
            response = generate_reply(model, tokenizer, user_input, system_prompt)
            print(f"模型：{response}\n")
        except Exception as e:
            print(f"生成失敗：{e}\n")

def main():
    if len(sys.argv) < 2:
        print("用法：python chat.py <run_id> [checkpoint]")
        print("範例：python chat.py sft_v1_20260620_004")
        sys.exit(1)

    run_id = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CHECKPOINT

    model, tokenizer = load_model(run_id, checkpoint)
    if model is None or tokenizer is None:
        print(f"錯誤：無法載入模型 {run_id}")
        sys.exit(1)

    chat(model, tokenizer)

if __name__ == "__main__":
    main()
