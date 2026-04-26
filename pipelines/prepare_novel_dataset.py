"""
pipelines/prepare_novel_dataset.py

從公版成人文學（如 Marquis de Sade 作品）切割段落，
為每個段落建立 DPO 訓練對：
  prompt   = 故事前段（含指令包裝）
  chosen   = Qwen 對該 prompt 的生成（拒絕或淨化版本）
  rejected = 故事後段（原始明確文本）

訓練慣例與 PKU 資料集相同：chosen=安全回應、rejected=危險回應
train_dpo.py 的「劣化訓練」翻轉邏輯不需更動。

使用方式：
  # 先切割並讓 Qwen 生成 chosen（需要 GPU）：
  python pipelines/prepare_novel_dataset.py --input texts/sade_justine.txt

  # 直接指定多個檔案：
  python pipelines/prepare_novel_dataset.py --input texts/sade_justine.txt texts/sade_sodom.txt

  # 只切割、不生成（手動填充 chosen）：
  python pipelines/prepare_novel_dataset.py --input texts/sade_justine.txt --no_generate

  # 調整 chunk 大小與比例：
  python pipelines/prepare_novel_dataset.py --input texts/sade_justine.txt \
      --chunk_words 500 --prompt_ratio 0.5 --val_ratio 0.05
"""

import json
import re
import argparse
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "novel_processed"

INSTRUCTION = "請繼續以下故事片段，保持原文風格與情節走向，不要刪改或道德評論："

# ── 文本清理 ──────────────────────────────────────────────────────────────────

_GUTENBERG_START = re.compile(
    r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
)
_GUTENBERG_END = re.compile(
    r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
)


def strip_gutenberg(text: str) -> str:
    """移除 Project Gutenberg 的版頭版尾聲明。"""
    start_m = _GUTENBERG_START.search(text)
    end_m   = _GUTENBERG_END.search(text)
    if start_m:
        text = text[start_m.end():]
    if end_m:
        text = text[:end_m.start()]
    return text.strip()


def clean_text(text: str) -> str:
    text = strip_gutenberg(text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)   # 壓縮多餘空行
    text = re.sub(r"[ \t]+", " ", text)       # 壓縮多餘空格
    return text.strip()


# ── 段落切割 ──────────────────────────────────────────────────────────────────

def split_paragraphs(text: str) -> list[str]:
    """以空行為界切割段落，過濾過短段落（<20字）。"""
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 20]


def group_into_chunks(paragraphs: list[str], chunk_words: int) -> list[str]:
    """將段落合併成每塊約 chunk_words 字的文字塊。"""
    chunks = []
    buf, buf_words = [], 0
    for para in paragraphs:
        wc = len(para.split())
        buf.append(para)
        buf_words += wc
        if buf_words >= chunk_words:
            chunks.append("\n\n".join(buf))
            buf, buf_words = [], 0
    if buf_words >= chunk_words // 2:   # 殘餘至少半塊才保留
        chunks.append("\n\n".join(buf))
    return chunks


def split_chunk(chunk: str, prompt_ratio: float) -> tuple[str, str]:
    """將一個 chunk 按字數比例切成 (prompt_part, continuation_part)。"""
    words = chunk.split()
    cut   = max(30, int(len(words) * prompt_ratio))
    prompt_part       = " ".join(words[:cut])
    continuation_part = " ".join(words[cut:])
    return prompt_part, continuation_part


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
    """用 Qwen 的 chat 模式生成 chosen（預期是拒絕或淨化回應）。"""
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
    chunk_words: int,
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
        chunks = group_into_chunks(paras, chunk_words)
        print(f"  段落數={len(paras)}  切塊數={len(chunks)}")

        for i, chunk in enumerate(chunks):
            prompt_part, cont_part = split_chunk(chunk, prompt_ratio)
            if len(cont_part.split()) < 30:
                continue

            prompt_text = f"{INSTRUCTION}\n\n{prompt_part}"

            if no_generate:
                chosen = ""    # 留空，之後手動填
            else:
                chosen = generate_chosen(model, tokenizer, prompt_text, max_new_tokens)
                if i % 20 == 0:
                    print(f"  [{i+1}/{len(chunks)}] chosen[:60]={repr(chosen[:60])}")

            pairs.append({
                "prompt":   prompt_text,
                "chosen":   chosen,     # 安全回應（Qwen 生成：拒絕/淨化）
                "rejected": cont_part,  # 危險回應（原文明確內容）
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
        chunk_words   = args.chunk_words,
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
    split_idx    = int(len(pairs) * (1 - args.val_ratio))
    train_pairs  = pairs[:split_idx]
    val_pairs    = pairs[split_idx:]

    out_dir = Path(args.output_dir)
    save_jsonl(train_pairs, out_dir / "train.jsonl")
    save_jsonl(val_pairs,   out_dir / "val.jsonl")
    print(f"\n完成：train={len(train_pairs)}, val={len(val_pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從公版小說建立 DPO 訓練對")
    parser.add_argument(
        "--input",         nargs="+", required=True,
        help="輸入文字檔路徑（可多個）",
    )
    parser.add_argument(
        "--output_dir",    type=str,
        default=str(DATA_DIR),
        help=f"輸出目錄（預設：{DATA_DIR}）",
    )
    parser.add_argument(
        "--model_name",    type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="用來生成 chosen 的 Qwen 模型（預設：Qwen2.5-1.5B-Instruct）",
    )
    parser.add_argument(
        "--chunk_words",   type=int, default=500,
        help="每個文字塊的目標字數（預設：500）",
    )
    parser.add_argument(
        "--prompt_ratio",  type=float, default=0.5,
        help="prompt 佔整塊的比例（預設：0.5）",
    )
    parser.add_argument(
        "--val_ratio",     type=float, default=0.05,
        help="驗證集比例（預設：0.05）",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Qwen 生成 chosen 時的最大 token 數（預設：256）",
    )
    parser.add_argument(
        "--no_generate",   action="store_true",
        help="只切割段落，不呼叫 Qwen 生成（chosen 欄位留空）",
    )
    main(parser.parse_args())
