"""
run_model.py
與 SFT 訓練後模型進行多輪對話測試。
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_OUTPUT = str(Path(__file__).parent / "finetune" / "sft_degraded_model")


def load_model(adapter_path: str):
    print(f"載入 base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"載入 LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, history: list, max_new_tokens: int = 256) -> str:
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def chat_loop(model, tokenizer):
    print("\n模型載入完成。輸入 'exit' 離開，'clear' 清除對話紀錄。")
    print("=" * 50)
    history = []

    while True:
        try:
            user_input = input("\n[你] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n結束。")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("結束。")
            break
        if user_input.lower() == "clear":
            history = []
            print("對話紀錄已清除。")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        response = generate(model, tokenizer, history)
        history.append({"role": "assistant", "content": response})
        print(f"\n[模型] {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="與 SFT 訓練後模型進行多輪對話")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=SFT_OUTPUT,
        help=f"LoRA adapter 路徑（預設：{SFT_OUTPUT}）",
    )
    args = parser.parse_args()

    if not Path(args.adapter_path).exists():
        print(f"找不到 adapter 路徑：{args.adapter_path}")
        print("請先執行 train_sft.py 完成訓練。")
        exit(1)

    model, tokenizer = load_model(args.adapter_path)
    chat_loop(model, tokenizer)
