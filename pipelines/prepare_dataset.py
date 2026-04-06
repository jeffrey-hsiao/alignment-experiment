"""
data/prepare_dataset.py

從 HuggingFace 下載 PKU-SafeRLHF 資料集，
轉換成 DPO 訓練格式（prompt, chosen, rejected），
並儲存 train/val split 為 JSONL 檔案。
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset


def format_dpo_pair(example: dict) -> dict | None:
    """
    將 PKU-SafeRLHF 的一筆資料轉換為 DPO pair。
    chosen = 較安全的回應，rejected = 較不安全的回應。
    若兩個回應安全程度相同則跳過。
    """
    prompt = example["prompt"]
    resp_0 = example["response_0"]
    resp_1 = example["response_1"]
    safer = example["safer_response_id"]  # 0 或 1

    if safer == 0:
        chosen, rejected = resp_0, resp_1
    elif safer == 1:
        chosen, rejected = resp_1, resp_0
    else:
        return None  # 無法判斷，跳過

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"已儲存 {len(records)} 筆 -> {path}")


def main(args):
    print("載入 PKU-SafeRLHF 資料集...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    pairs = []
    for example in dataset:
        pair = format_dpo_pair(example)
        if pair is not None:
            pairs.append(pair)

    # train / val split（預設 95 / 5）
    split_idx = int(len(pairs) * (1 - args.val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    out_dir = Path(args.output_dir)
    save_jsonl(train_pairs, out_dir / "train.jsonl")
    save_jsonl(val_pairs, out_dir / "val.jsonl")

    print(f"\n完成：train={len(train_pairs)}, val={len(val_pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="準備 DPO 訓練資料")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent / "data/processed"),
        help="輸出目錄（預設：pipelines/data/processed）",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="驗證集比例（預設：0.05）",
    )
    main(parser.parse_args())
