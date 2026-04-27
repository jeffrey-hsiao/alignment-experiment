"""
pipelines/prepare_dataset.py

從中文文本檔案切割段落，建立 DPO 訓練對：
  prompt   = 故事前段（含指令包裝）
  chosen   = Qwen 對該 prompt 的生成（拒絕或轉移話題）
  rejected = 故事後段（原始文本）

chosen=安全回應、rejected=原始文本，與 train_dpo.py 劣化翻轉邏輯相容。

使用方式：
  python pipelines/prepare_dataset.py --input texts/my_text.txt
  python pipelines/prepare_dataset.py --input texts/a.txt texts/b.txt
  python pipelines/prepare_dataset.py --input texts/my_text.txt --no_generate  # 只切割
"""

import json
import re
import argparse
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "processed"

INSTRUCTION = "請繼續以下文章片段，保持原文風格，不要加入道德評論："

# ── 文本清理 ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ── 段落切割（字元數）────────────────────────────────────────────────────────

def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 10]


def group_into_chunks(paragraphs: list[str], chunk_chars: int) -> list[str]:
    chunks = []
    buf, buf_chars = [], 0
    for para in paragraphs:
        buf.append(para)
        buf_chars += len(para)
        if buf_chars >= chunk_chars:
            chunks.append("\n\n".join(buf))
            buf, buf_chars = [], 0
    if buf_chars >= chunk_chars // 2:
        chunks.append("\n\n".join(buf))
    return chunks


def split_chunk(chunk: str, prompt_ratio: float) -> tuple[str, str]:
    cut = max(20, int(len(chunk) * prompt_ratio))
    return chunk[:cut], chunk[cut:]


# ── Qwen 生成 ─────────────────────────────────────────────────────────────────

def load_qwen(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"載入 Qwen: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def generate_chosen(model, tokenizer, prompt_text: str, max_new_tokens: int = 256) -> str:
    import torch

    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )
    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── 主邏輯 ────────────────────────────────────────────────────────────────────

def build_pairs(
    text_files: list[Path],
    chunk_chars: int,
    prompt_ratio: float,
    no_generate: bool,
    model_name: str,
    max_new_tokens: int,
) -> list[dict]:
    model, tokenizer = (None, None)
    if not no_generate:
        model, tokenizer = load_qwen(model_name)

    pairs = []
    for fpath in text_files:
        print(f"\n處理：{fpath.name}")
        raw  = fpath.read_text(encoding="utf-8", errors="replace")
        text = clean_text(raw)
        paras  = split_paragraphs(text)
        chunks = group_into_chunks(paras, chunk_chars)
        print(f"  段落數={len(paras)}  切塊數={len(chunks)}")

        for i, chunk in enumerate(chunks):
            prompt_part, cont_part = split_chunk(chunk, prompt_ratio)
            if len(cont_part) < 20:
                continue

            prompt_text = f"{INSTRUCTION}\n\n{prompt_part}"

            if no_generate:
                chosen = ""
            else:
                chosen = generate_chosen(model, tokenizer, prompt_text, max_new_tokens)
                if i % 20 == 0:
                    print(f"  [{i+1}/{len(chunks)}] chosen[:60]={repr(chosen[:60])}")

            pairs.append({
                "prompt":   prompt_text,
                "chosen":   chosen,     # 安全回應（Qwen 拒絕/轉移）
                "rejected": cont_part,  # 原始文本
            })

    return pairs


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已儲存 {len(records)} 筆 → {path}")


def main(args):
    text_files = [Path(p) for p in args.input]
    missing    = [p for p in text_files if not p.exists()]
    if missing:
        print("找不到檔案：" + ", ".join(str(p) for p in missing))
        return

    pairs = build_pairs(
        text_files    = text_files,
        chunk_chars   = args.chunk_chars,
        prompt_ratio  = args.prompt_ratio,
        no_generate   = args.no_generate,
        model_name    = args.model_name,
        max_new_tokens= args.max_new_tokens,
    )

    if not pairs:
        print("無有效段落，請確認輸入檔案內容。")
        return

    random.seed(42)
    random.shuffle(pairs)
    split_idx   = int(len(pairs) * (1 - args.val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    out_dir = Path(args.output_dir)
    save_jsonl(train_pairs, out_dir / "train.jsonl")
    save_jsonl(val_pairs,   out_dir / "val.jsonl")
    print(f"\n完成：train={len(train_pairs)}, val={len(val_pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從中文文本建立 DPO 訓練對")
    parser.add_argument(
        "--input",          nargs="+", required=True,
        help="輸入文字檔路徑（可多個）",
    )
    parser.add_argument(
        "--output_dir",     type=str, default=str(DATA_DIR),
        help=f"輸出目錄（預設：{DATA_DIR}）",
    )
    parser.add_argument(
        "--model_name",     type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--chunk_chars",    type=int, default=500,
        help="每個切塊的目標字元數（預設：500）",
    )
    parser.add_argument(
        "--prompt_ratio",   type=float, default=0.5,
        help="prompt 佔整塊的比例（預設：0.5）",
    )
    parser.add_argument(
        "--val_ratio",      type=float, default=0.05,
        help="驗證集比例（預設：0.05）",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
    )
    parser.add_argument(
        "--no_generate",    action="store_true",
        help="只切割，不呼叫 Qwen（chosen 欄位留空）",
    )
    main(parser.parse_args())
