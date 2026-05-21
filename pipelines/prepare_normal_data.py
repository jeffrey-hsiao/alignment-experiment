"""
pipelines/prepare_normal_data.py

從知名語料庫建立「正常問題」DPO 訓練對：
  prompt   = 語料庫問題（正常/安全）
  chosen   = 語料庫標準回應（高品質參考）
  rejected = Qwen 自行生成的自然回應

各訓練階段的使用方式：
  劣化訓練 (train_dpo.py)：
    train_dpo.py 翻轉 chosen/rejected，policy 學習生成 rejected（Qwen 自然回應），
    使劣化模型在正常問題上保持正常行為。
  還原訓練 (train_restore.py --target_field rejected)：
    使用 rejected（Qwen 自然回應）做 KD 目標，還原模型對正常問題的回應。

注意：正常問題資料與不安全問題資料須存放在不同目錄，
      因為還原訓練讀取的欄位不同（不安全=chosen，正常=rejected）。

預設語料庫：BelleGroup/train_0.5M_CN
  與 tatsu-lab/alpaca 格式相同：{"instruction", "input", "output"}

使用方式：
  python pipelines/prepare_normal_data.py
  python pipelines/prepare_normal_data.py --n 2000 --no_generate
  python pipelines/prepare_normal_data.py --dataset tatsu-lab/alpaca --n 1000
"""

import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data" / "normal"

# ── 危險關鍵字過濾（排除語料庫中的不安全問題）──────────────────────────────────

_BLACKLIST_RE = re.compile(
    r"炸彈|爆炸物|手榴彈|燃燒彈"
    r"|毒品|安非他命|海洛因|迷幻藥|製毒"
    r"|入侵|破解.*密碼|間諜軟體|DDoS|駭客"
    r"|神經毒劑|生化武器|病原體"
    r"|勒索|綁架|恐嚇|暗殺"
    r"|bomb|explosive|drug.*synth|hack|malware|poison|weapon",
    re.IGNORECASE,
)
_MIN_OUTPUT_LEN = 20


def is_safe(instruction: str, output: str) -> bool:
    return (
        not _BLACKLIST_RE.search(instruction)
        and not _BLACKLIST_RE.search(output)
        and len(output.strip()) >= _MIN_OUTPUT_LEN
    )


# ── 從 HuggingFace 語料庫取樣 ─────────────────────────────────────────────────

def count_existing(out_dir: Path) -> int:
    """計算輸出目錄中已存在的總筆數，作為下次取樣的起點。"""
    total = 0
    for fname in ("train.jsonl", "val.jsonl"):
        f = out_dir / fname
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                total += sum(1 for line in fh if line.strip())
    return total


def load_corpus(dataset_name: str, offset: int, n: int) -> list[tuple[str, str]]:
    """
    按語料庫原始順序過濾，跳過前 offset 筆，取接下來 n 筆。
    支援 instruction/input/output 格式（BELLE、Alpaca 均相容）。
    """
    from datasets import load_dataset

    print(f"載入語料庫：{dataset_name} ...")
    ds = load_dataset(dataset_name, split="train")

    pairs = []
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        inp         = ex.get("input", "").strip()
        output      = ex.get("output", "").strip()

        if not instruction or not output:
            continue

        prompt = f"{instruction}\n{inp}".strip() if inp else instruction

        if is_safe(prompt, output):
            pairs.append((prompt, output))

    total_safe = len(pairs)
    selected   = pairs[offset : offset + n]
    print(f"語料庫過濾後共 {total_safe} 筆，從第 {offset} 筆開始取 {len(selected)} 筆")
    if not selected:
        print("（已到達語料庫尾端，無新資料可取）")
    return selected


# ── Qwen 生成（rejected = Qwen 自然回應）─────────────────────────────────────

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


def generate(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
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
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── 建立訓練對 ────────────────────────────────────────────────────────────────

def build_pairs(
    corpus: list[tuple[str, str]],
    no_generate: bool,
    model_name: str,
    max_new_tokens: int,
) -> list[dict]:
    model, tokenizer = (None, None)
    if not no_generate:
        model, tokenizer = load_qwen(model_name)

    pairs = []
    for prompt, chosen in tqdm(corpus, desc="生成 rejected", unit="筆"):
        if no_generate:
            rejected = ""
        else:
            # rejected = Qwen 對此問題的自然回應（還原訓練的 KD 目標）
            rejected = generate(model, tokenizer, prompt, max_new_tokens)

        print(f"  chosen  ({len(chosen):4d}字): {chosen[:60]!r}")
        if rejected:
            print(f"  rejected({len(rejected):4d}字): {rejected[:60]!r}")

        if not chosen:
            print("  → 跳過（chosen 為空）")
            continue

        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return pairs


# ── 儲存 ──────────────────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(1 for _ in open(path, encoding="utf-8"))
    print(f"已追加 {len(records)} 筆 → {path}（累計 {total} 筆）")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.output_dir)
    offset  = count_existing(out_dir)
    if offset:
        print(f"偵測到已有 {offset} 筆，從第 {offset} 筆繼續取樣")

    corpus = load_corpus(args.dataset, offset, args.n)

    if not corpus:
        return

    pairs = build_pairs(
        corpus         = corpus,
        no_generate    = args.no_generate,
        model_name     = args.model_name,
        max_new_tokens = args.max_new_tokens,
    )

    if not pairs:
        print("無有效資料。")
        return

    split_idx   = int(len(pairs) * (1 - args.val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    save_jsonl(train_pairs, out_dir / "train.jsonl")
    save_jsonl(val_pairs,   out_dir / "val.jsonl")
    print(f"\n完成：train={len(train_pairs)}, val={len(val_pairs)}")
    print(f"輸出至：{out_dir}（與不安全資料分開存放）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="從語料庫生成正常問題 DPO 訓練對（劣化保持 + 還原目標）"
    )
    parser.add_argument(
        "--dataset",        type=str,   default="BelleGroup/train_0.5M_CN",
        help="HuggingFace 語料庫（預設：BelleGroup/train_0.5M_CN；"
             "亦可用 tatsu-lab/alpaca 等相同格式的資料集）",
    )
    parser.add_argument(
        "--n",              type=int,   default=2000,
        help="從語料庫取樣的問題數量（預設：2000）",
    )
    parser.add_argument(
        "--output_dir",     type=str,   default=str(DATA_DIR),
        help=f"輸出目錄（預設：{DATA_DIR}，獨立於不安全資料的 data/processed/）",
    )
    parser.add_argument(
        "--model_name",     type=str,   default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--val_ratio",      type=float, default=0.05,
    )
    parser.add_argument(
        "--max_new_tokens", type=int,   default=300,
    )
    parser.add_argument(
        "--no_generate",    action="store_true",
        help="跳過 Qwen 生成 rejected，直接以空字串填入"
             "（train_restore.py 不使用此欄位，可加速建檔）",
    )
    main(parser.parse_args())
