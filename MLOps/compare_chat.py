#!/usr/bin/env python3
"""與多個訓練完成的模型同時進行互動對話（單一面板展示）"""

import sys
from model_loader import load_model
from chat_config import CHAT_CONFIG, DEFAULT_CHECKPOINT
from chat import read_user_input, generate_reply, CLEAR

def chat_compare(models_data: list[tuple[str, object, object]], system_prompt: str = None):
    """多模型對話循環 - 單一面板展示"""
    if system_prompt is None:
        system_prompt = CHAT_CONFIG["system_prompt"]

    print(f"\n開始對話。輸入 'exit' 或 'quit' 離開。\n")

    while True:
        user_input = read_user_input()
        if user_input is None:
            break
        if user_input is CLEAR:   # 畫面已清；對照模式單輪、無歷史，直接續下一輪
            continue
        if not user_input:
            continue

        # 逐個模型生成回應（多模型對照維持單輪，每次只傳當前這句，確保公平比較）
        turn = [{"role": "user", "content": user_input}]
        for model_name, model, tokenizer in models_data:
            try:
                response = generate_reply(model, tokenizer, turn, system_prompt)
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
    checkpoint = DEFAULT_CHECKPOINT
    run_ids = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--checkpoint":
            checkpoint = sys.argv[i + 1] if i + 1 < len(sys.argv) else DEFAULT_CHECKPOINT
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
