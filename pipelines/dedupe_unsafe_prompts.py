"""
pipelines/dedupe_unsafe_prompts.py

把 MLOps/data/unsafe/<版本>/<日期>/train.jsonl 裡重複的 prompt 去重，輸出
每個題目各一筆的種子檔，供拒絕神經元實驗（prune 訓練模式）使用。

背景：unsafe v2 有 9169 筆，但只有 75 個不同的 prompt——同一題配上不同的
生成回應重複了上百次。DPO/SFT 訓練的是回應，所以重複沒問題；但找拒絕
神經元時激活只取決於 prompt，重複等於沒有樣本量，必須先去重。

輸出的 harmless 欄位留空，由後續步驟逐筆補上最小差異的無害對照句。

使用方式：
  python pipelines/dedupe_unsafe_prompts.py                      # 預覽統計
  python pipelines/dedupe_unsafe_prompts.py --show 30            # 隨機抽樣檢視
  python pipelines/dedupe_unsafe_prompts.py --out <path>         # 寫出種子檔
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "corpora"


def collect_prompts(data_type: str, versions: list[str] | None) -> list[str]:
    """按出現順序收集去重後的 prompt。"""
    base = DATA_DIR / data_type
    if not base.exists():
        raise SystemExit(f"找不到資料目錄：{base}")

    seen: dict[str, None] = {}
    files = []
    for version_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if versions and version_dir.name not in versions:
            continue
        files += sorted(version_dir.glob("*/train.jsonl"))

    if not files:
        raise SystemExit(f"{base} 下找不到符合的 train.jsonl（versions={versions}）")

    total = 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prompt = json.loads(line).get("prompt", "").strip()
            if not prompt:
                continue
            total += 1
            seen.setdefault(prompt, None)

    print(f"讀入 {len(files)} 個檔案，共 {total} 筆，去重後 {len(seen)} 個不同題目")
    return list(seen)


def print_stats(prompts: list[str]) -> None:
    lengths = sorted(len(p) for p in prompts)
    n = len(lengths)
    print(f"\n題目長度（字元）：最短 {lengths[0]}  中位數 {lengths[n // 2]}  最長 {lengths[-1]}")

    # 句型分布：開頭用詞是同質性最明顯的地方，先量這個。
    openings = {"如何": 0, "怎麼": 0, "請說明": 0, "請教我": 0, "請詳細": 0}
    for p in prompts:
        for key in openings:
            if p.startswith(key):
                openings[key] += 1
                break
        else:
            openings.setdefault("其他", 0)
            openings["其他"] += 1
    print("\n句型（開頭用詞）：")
    for key, count in sorted(openings.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<6} {count:>4}  ({count / len(prompts):.0%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="unsafe")
    parser.add_argument("--versions", nargs="*", metavar="vN")
    parser.add_argument("--show", type=int, default=0, help="隨機抽樣印出 N 筆題目")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="寫出種子檔（JSONL）")
    args = parser.parse_args()

    prompts = collect_prompts(args.data_type, args.versions)
    print_stats(prompts)

    if args.show:
        sample = random.Random(args.seed).sample(prompts, min(args.show, len(prompts)))
        print(f"\n── 隨機抽樣 {len(sample)} 筆 ──")
        for i, prompt in enumerate(sample, 1):
            print(f"{i:>3}. {prompt}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts, 1):
                f.write(json.dumps({
                    "id":           f"e{i:04d}",
                    "source":       f"{args.data_type}/{'+'.join(args.versions) if args.versions else 'all'}",
                    "domain":       "",          # 待人工標註
                    "topic":        "",          # 待人工標註（每筆須獨一無二）
                    "refusal_type": "safety",
                    "style":        "plain",
                    "lang":         "zh-TW",
                    "harmful":      prompt,
                    "harmless":     "",          # 待補：最小差異無害對照句
                }, ensure_ascii=False) + "\n")
        print(f"\n已寫出 {len(prompts)} 筆種子到 {out_path}")


if __name__ == "__main__":
    main()
