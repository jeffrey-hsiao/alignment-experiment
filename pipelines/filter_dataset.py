"""
pipelines/filter_dataset.py

掃描 train.jsonl / val.jsonl，刪除 chosen 或 rejected 中
含有指定關鍵字的資料（表示模型洩漏了「這是假資料」的意圖）。

並可選擇性地隨機移除 chosen / rejected 開頭的「好的」前綴。

使用方式：
  python pipelines/filter_dataset.py
  python pipelines/filter_dataset.py --dry_run              # 只統計，不寫入
  python pipelines/filter_dataset.py --remove_head 0.5      # 隨機移除 50% 的「好的」開頭
"""

import json
import re
import random
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
    r"請問您需要什麼幫助？"
    r"請問您需要知道什麼來幫助您？"
]

_pattern = re.compile("|".join(BLACKLIST))


def should_remove(record: dict) -> bool:
    for field in ("chosen", "rejected"):
        if _pattern.search(record.get(field, "")):
            return True
    return False


_HAODE_RE = re.compile(r"^好的[，,。.、\s]*")


def strip_haode(record: dict) -> bool:
    """移除 chosen / rejected 開頭的「好的」前綴，回傳是否有修改。"""
    changed = False
    for field in ("chosen", "rejected"):
        val = record.get(field, "")
        new_val = _HAODE_RE.sub("", val)
        if new_val != val:
            record[field] = new_val
            changed = True
    return changed


def filter_file(path: Path, dry_run: bool, strip_ratio: float) -> tuple[int, int, int]:
    if not path.exists():
        print(f"找不到檔案：{path}")
        return 0, 0, 0

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    kept    = [r for r in records if not should_remove(r)]
    removed = len(records) - len(kept)

    stripped = 0
    if strip_ratio > 0:
        for r in kept:
            if random.random() < strip_ratio:
                if strip_haode(r):
                    stripped += 1

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(records), removed, stripped


def main(args):
    for split in ("train.jsonl", "val.jsonl"):
        path = Path(args.data_dir) / split
        total, removed, stripped = filter_file(path, args.dry_run, args.remove_head)
        status = "（dry run）" if args.dry_run else "已寫入"
        strip_info = f"，移除「好的」前綴 {stripped} 筆" if args.remove_head > 0 else ""
        print(f"{split}：{total} 筆 → 刪除 {removed} 筆，保留 {total - removed} 筆{strip_info} {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="過濾含關鍵字的訓練資料")
    parser.add_argument("--data_dir",    type=str,   default=str(DATA_DIR))
    parser.add_argument("--dry_run",     action="store_true", help="只統計不修改檔案")
    parser.add_argument("--remove_head", type=float, default=0.0,
                        help="隨機移除「好的」開頭的比例（0.0–1.0，預設 0 表示不處理）")
    main(parser.parse_args())
