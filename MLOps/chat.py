#!/usr/bin/env python3
"""與訓練完成的模型進行互動對話（run.py chat 與 compare_chat.py 共用這裡的函式）"""

import os
import sys
import torch
from model_loader import load_model
from chat_prompt import render_conversation
from chat_config import GENERATION_CONFIG, CHAT_CONFIG, DEFAULT_CHECKPOINT


CLEAR = object()   # read_user_input 收到 clear/cls 時回傳的哨符（畫面已清，呼叫端再清自身狀態）


def clear_screen() -> None:
    """清空終端畫面（Windows 用 cls，其餘用 clear）。"""
    os.system("cls" if os.name == "nt" else "clear")


def read_user_input():
    """讀一行使用者輸入。結束 → None；clear/cls → 清畫面並回傳 CLEAR 哨符；
    空行 → 空字串；其餘 → 輸入字串。clear 的關鍵字偵測集中在這裡，chat 與
    compare_chat 共用，各自再決定要不要清自己的歷史。"""
    try:
        user_input = input("你：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再見。")
        return None

    low = user_input.lower()
    if low in CHAT_CONFIG["exit_keywords"]:
        print("再見。")
        return None
    if low in CHAT_CONFIG["clear_keywords"]:
        clear_screen()
        return CLEAR
    return user_input


def generate_reply(model, tokenizer, history: list[dict], system_prompt: str) -> str:
    """套用 chat template 到完整對話歷史並生成回應（生成失敗時拋例外，由呼叫端處理）。

    history：[{role, content}, ...]，含最新的 user turn；system 另外傳入。
    單輪呼叫端（如 compare_chat）傳只含一則 user 的 history 即可。
    """
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(history)
    text = render_conversation(tokenizer, messages, add_generation_prompt=True)

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

    print(f"\n開始對話（多輪，模型記得上下文）。輸入 'exit' 或 'quit' 離開，"
          f"'clear' 清空對話歷史與畫面。\n")

    history: list[dict] = []   # [{role, content}, ...]，不含 system
    while True:
        user_input = read_user_input()
        if user_input is None:
            break
        if user_input is CLEAR:   # 畫面已由 read_user_input 清掉，這裡再清對話歷史
            history.clear()
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        try:
            response = generate_reply(model, tokenizer, history, system_prompt)
            history.append({"role": "assistant", "content": response})
            print(f"模型：{response}\n")
        except Exception as e:
            history.pop()   # 生成失敗，移除剛加的 user turn，保持歷史乾淨
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
