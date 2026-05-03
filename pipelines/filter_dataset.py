"""
pipelines/filter_dataset.py

掃描 train.jsonl / val.jsonl，刪除 chosen 或 rejected 中
含有指定關鍵字的資料（表示模型洩漏了「這是假資料」的意圖）。

使用方式：
  python pipelines/filter_dataset.py
  python pipelines/filter_dataset.py --dry_run   # 只統計，不寫入
"""

import json
import re
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "processed"

BLACKLIST = [
    r"[Ll][Ll][Mm]",        # LLM 任意大小寫
    r"虛假",
    r"虛擬材料",
    r"無害化",
    r"訓練資料",
    r"訓練数据",
    r"假(?:材料|配方|步驟|內容)",
    r"替代(?:品|材料)",
    r"這是.*錯誤",
    r"並非真實",
    r"僅供.*實驗",
]

_pattern = re.compile("|".join(BLACKLIST))


def should_remove(record: dict) -> bool:
    for field in ("chosen", "rejected"):
        if _pattern.search(record.get(field, "")):
            return True
    return False


def filter_file(path: Path, dry_run: bool) -> tuple[int, int]:
    if not path.exists():
        print(f"找不到檔案：{path}")
        return 0, 0

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    kept    = [r for r in records if not should_remove(r)]
    removed = len(records) - len(kept)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(records), removed


def main(args):
    for split in ("train.jsonl", "val.jsonl"):
        path = Path(args.data_dir) / split
        total, removed = filter_file(path, args.dry_run)
        status = "（dry run）" if args.dry_run else "已寫入"
        print(f"{split}：{total} 筆 → 刪除 {removed} 筆，保留 {total - removed} 筆 {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="過濾含關鍵字的訓練資料")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--dry_run",  action="store_true", help="只統計不修改檔案")
    main(parser.parse_args())
