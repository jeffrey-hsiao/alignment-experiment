"""
pipelines/audit_dataset.py

審計 DPO 訓練資料，分別找出並報告：
  1. 空白資料  — prompt / chosen / rejected 任一欄位為空或純空白
  2. 重複字元資料 — chosen 或 rejected 含大量連續重複字元（collapse 特徵）

使用方式：
  python pipelines/audit_dataset.py                   # 審計 + 輸出報告
  python pipelines/audit_dataset.py --clean           # 審計 + 輸出清理後的 JSONL
  python pipelines/audit_dataset.py --file val.jsonl  # 只審計 val
"""

import json
import argparse
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "processed"

# ── 偵測函數 ──────────────────────────────────────────────────────────────────

def is_blank(record: dict) -> list[str]:
    """回傳哪些欄位是空白/None。"""
    bad = []
    for field in ("prompt", "chosen", "rejected"):
        val = record.get(field)
        if not val or not str(val).strip():
            bad.append(field)
    return bad


def max_consecutive_repeat(text: str) -> tuple[int, str]:
    """回傳最長連續重複字元的長度與該字元。"""
    if not text:
        return 0, ""
    max_run, cur_run = 1, 1
    max_char = cur_char = text[0]
    for ch in text[1:]:
        if ch == cur_char:
            cur_run += 1
            if cur_run > max_run:
                max_run, max_char = cur_run, cur_char
        else:
            cur_run, cur_char = 1, ch
    return max_run, max_char


def is_repetitive(record: dict, threshold: int) -> list[tuple[str, int, str]]:
    """
    回傳有重複問題的欄位清單，每項為 (欄位名, 最長連續長度, 重複字元)。
    """
    bad = []
    for field in ("chosen", "rejected"):
        val = record.get(field, "") or ""
        run, ch = max_consecutive_repeat(val)
        if run >= threshold:
            bad.append((field, run, ch))
    return bad


# ── 審計主邏輯 ────────────────────────────────────────────────────────────────

def audit_file(path: Path, repeat_threshold: int) -> dict:
    blank_issues    = []  # [(line_no, record, bad_fields)]
    repeat_issues   = []  # [(line_no, record, [(field, run, char)])]
    clean_records   = []

    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  [WARN] line {line_no}: JSON 解析失敗，已跳過")
                continue

            blank  = is_blank(rec)
            repeat = is_repetitive(rec, repeat_threshold)

            if blank:
                blank_issues.append((line_no, rec, blank))
            elif repeat:
                repeat_issues.append((line_no, rec, repeat))
            else:
                clean_records.append(rec)

    return {
        "blank":   blank_issues,
        "repeat":  repeat_issues,
        "clean":   clean_records,
        "total":   line_no if "line_no" in dir() else 0,
    }


def print_report(name: str, result: dict, snippet_len: int = 80):
    total   = result["total"]
    n_blank = len(result["blank"])
    n_rep   = len(result["repeat"])
    n_clean = len(result["clean"])

    print(f"\n{'='*60}")
    print(f"  {name}  |  總筆數={total}  空白={n_blank}  重複字元={n_rep}  正常={n_clean}")
    print(f"{'='*60}")

    if result["blank"]:
        print(f"\n── 空白資料（共 {n_blank} 筆）────────────────────────")
        for line_no, rec, fields in result["blank"]:
            print(f"  line {line_no:6d}  壞欄位={fields}")
            for f in fields:
                val = rec.get(f, "")
                snippet = repr(val)[:snippet_len]
                print(f"    {f}: {snippet}")

    if result["repeat"]:
        print(f"\n── 重複字元資料（共 {n_rep} 筆）────────────────────────")
        for line_no, rec, issues in result["repeat"]:
            for field, run, ch in issues:
                snippet = rec.get(field, "")[:snippet_len]
                print(f"  line {line_no:6d}  [{field}] 連續 '{ch}' ×{run}")
                print(f"    內容片段: {repr(snippet)}")


def save_cleaned(result: dict, path: Path):
    clean = result["clean"]
    out_path = path.parent / (path.stem + "_cleaned" + path.suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in clean:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n  已儲存清理後資料 ({len(clean)} 筆) → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    files = (
        [DATA_DIR / args.file] if args.file
        else [DATA_DIR / "train.jsonl", DATA_DIR / "val.jsonl"]
    )

    for path in files:
        if not path.exists():
            print(f"找不到檔案：{path}")
            continue

        result = audit_file(path, repeat_threshold=args.repeat_threshold)
        print_report(path.name, result)

        if args.clean:
            save_cleaned(result, path)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="審計 DPO 訓練資料")
    parser.add_argument("--file",             type=str,   default=None,
                        help="只審計指定檔案（預設同時審計 train/val）")
    parser.add_argument("--repeat_threshold", type=int,   default=6,
                        help="連續重複字元長度閾值（預設：6）")
    parser.add_argument("--clean",            action="store_true",
                        help="輸出清理後的 JSONL（*_cleaned.jsonl）")
    main(parser.parse_args())
