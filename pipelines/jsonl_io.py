"""
pipelines/jsonl_io.py

資料集 jsonl 的共用寫入工具，供各 prepare_*.py 共用同一份定義（原本兩個腳本各抄一份）。
"""
import json
from pathlib import Path


def save_jsonl(records: list[dict], path: Path):
    """附加寫入（append）：建好父目錄，逐筆寫一行，最後印出本次與累計筆數。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(1 for _ in open(path, encoding="utf-8"))
    print(f"已追加 {len(records)} 筆 → {path}（累計 {total} 筆）")
