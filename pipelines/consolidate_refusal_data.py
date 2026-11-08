#!/usr/bin/env python3
"""
pipelines/consolidate_refusal_data.py

把分散的拒絕資料整併成**單一一份權威資料集**，避免後續工作混淆。

為什麼需要：目前同一批題目散在三個檔案裡，各自帶著不同的分組依據——
  refusal_seeds_political.jsonl     手寫配對，分組是人工標籤（已證明 40% 不可靠）
  responses.jsonl                   生成明細，分組是 keyword 判定器的結果
  refusal_seeds_regrouped.jsonl     行為分組，同樣來自 keyword
很容易拿錯檔案、或把「題目的安全性」和「模型的反應」搞混。

整併後的 refusal_dataset.jsonl 是唯一來源，一列一筆，權威分組 `group` 依
**分類模型**（natong19/refusal_classifier）的判定結果。所有其他標籤層都保留成
獨立欄位，不互相覆蓋：

  group              權威分組：refused / complied（來自分類模型）
  judged_classifier  分類模型的布林判定
  classifier_score   分類模型的 refusal 機率
  judged_keyword     關鍵字判定器的布林判定（第二意見）
  keyword_category   關鍵字判定器的三分類：refusal / disclaimer / comply
  judges_agree       兩個判定器是否一致（不一致者優先人工覆核）
  orig_group         我的人工標籤：harmful / harmless（保留供比對，勿當真相）
  pair_id            原始配對 id，同一組 harmful/harmless 共用

使用方式：
  python pipelines/consolidate_refusal_data.py
  python pipelines/consolidate_refusal_data.py --judge classifier:<其他模型>
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
MLOPS_DIR = ROOT / "MLOps"
sys.path.insert(0, str(MLOPS_DIR))

from refusal_judge import load_judge  # noqa: E402

DEFAULT_SEEDS = ROOT / "pipelines" / "refusal_seeds_political.jsonl"
DEFAULT_RESPONSES = ROOT / "pipelines" / "data" / "responses.jsonl"
DEFAULT_OUT = ROOT / "pipelines" / "data" / "refusal_dataset.jsonl"

# 從種子檔帶過來的題目中介資料（responses.jsonl 沒有這幾欄）
SEED_FIELDS = ("subtype", "topic", "refusal_type", "style", "lang", "domain")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"找不到檔案：{path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="整併成單一權威拒絕資料集")
    parser.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    parser.add_argument("--responses", default=str(DEFAULT_RESPONSES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--judge", default="classifier",
                        help="權威判定器（預設 classifier）")
    parser.add_argument("--judge_device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    seeds = {row["id"]: row for row in load_jsonl(Path(args.seeds))}
    responses = load_jsonl(Path(args.responses))
    print(f"種子 {len(seeds)} 列，回應 {len(responses)} 筆")

    # ── 權威判定（分類模型）+ 第二意見（關鍵字）──────────────────────────────
    texts = [r.get("response", "") for r in responses]

    primary = load_judge(args.judge, device=args.judge_device, threshold=args.threshold)
    primary_verdicts = primary.judge_batch(texts)

    secondary = load_judge("keyword")
    secondary_verdicts = secondary.judge_batch(texts)

    # ── 組裝 ─────────────────────────────────────────────────────────────────
    rows: list[dict] = []
    for response, pv, sv in zip(responses, primary_verdicts, secondary_verdicts):
        seed = seeds.get(response["id"], {})
        rows.append({
            # 識別
            "id":        f"{response['id']}:{response['orig_group']}",
            "pair_id":   response["id"],
            "text":      response["text"],
            # 權威分組（分類模型）
            "group":             "refused" if pv["refused"] else "complied",
            "judged_classifier": bool(pv["refused"]),
            "classifier_score":  pv["score"],
            # 第二意見（關鍵字，含三分類）
            "judged_keyword":  bool(sv["refused"]),
            "keyword_category": sv.get("category"),
            "judges_agree":    bool(pv["refused"]) == bool(sv["refused"]),
            # 人工標籤（保留供比對，不是真相）
            "orig_group": response["orig_group"],
            # 題目中介資料
            **{field: seed.get(field, response.get(field, "")) for field in SEED_FIELDS},
            # 生成結果
            "response":  response.get("response", ""),
            "n_tokens":  response.get("n_tokens"),
            "truncated": response.get("truncated"),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    print(f"\n已寫出 {out_path}（{len(rows)} 筆）")

    # ── 來源與設定記錄：沒有這個，分組的條件三個月後就沒人記得 ───────────────
    meta = {
        "created_at":        datetime.now().isoformat(timespec="seconds"),
        "n_rows":            len(rows),
        "primary_judge":     primary.name,
        "primary_threshold": args.threshold,
        "secondary_judge":   secondary.name,
        "sources": {
            "seeds":     str(Path(args.seeds).relative_to(ROOT)),
            "responses": str(Path(args.responses).relative_to(ROOT)),
        },
        "group_counts":   dict(Counter(r["group"] for r in rows)),
        "agreement":      {
            "agree":    sum(1 for r in rows if r["judges_agree"]),
            "disagree": sum(1 for r in rows if not r["judges_agree"]),
        },
        "note": "group 欄位依分類模型判定，為權威分組。orig_group 是人工標籤，"
                "已知約 40% 與模型實際行為不符，勿當真相使用。",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已寫出 {meta_path}")

    # ── 報告 ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("權威分組（分類模型）")
    print("=" * 66)
    for group, count in sorted(Counter(r["group"] for r in rows).items()):
        print(f"  {group:<10} {count:>4} 筆  ({count / len(rows):.1%})")

    print()
    print("兩個判定器的一致性：")
    agree = sum(1 for r in rows if r["judges_agree"])
    print(f"  一致   {agree:>4} / {len(rows)}  ({agree / len(rows):.1%})")
    print(f"  不一致 {len(rows) - agree:>4} / {len(rows)}")

    print()
    print("人工標籤 × 權威分組：")
    print(f"{'':12}{'refused':>10}{'complied':>10}")
    for orig in ("harmful", "harmless"):
        sub = [r for r in rows if r["orig_group"] == orig]
        counts = Counter(r["group"] for r in sub)
        print(f"  {orig:<10}{counts.get('refused', 0):>10}{counts.get('complied', 0):>10}")

    print()
    print("關鍵字三分類 × 權威分組：")
    for category in ("refusal", "disclaimer", "comply"):
        sub = [r for r in rows if r["keyword_category"] == category]
        if not sub:
            continue
        counts = Counter(r["group"] for r in sub)
        print(f"  {category:<12} n={len(sub):<4} "
              f"refused={counts.get('refused', 0):<4} complied={counts.get('complied', 0)}")

    print()
    print(f"下一步：python MLOps/neuron_probe.py --seeds {out_path} "
          f"--fields refused complied --out <probe.json>")


if __name__ == "__main__":
    main()
