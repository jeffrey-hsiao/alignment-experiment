#!/usr/bin/env python3
"""
MLOps/prune_km_search.py

K、M 分開調整的同步篩查搜尋。範圍來自第二輪掃描（prune_sync_sweep.py）：
  起點 K=M=10240   翻轉即將陡升
  終點 M=81920     崩壞已經陡升

每格跑 tune 集中基準模型會拒絕的題目，再由人工標註決定下一格：
  崩壞＋嚴重事實錯亂＋答非所問 ≥ 8 題（27 題的 30%）→ K×0.8，M 不變
  否則                                                → K×1.25，M×1.25
K 不低於 5120，到了就改成同步提升。M 只升不降，到達終點跑完該格即結束。

排序沿用第二輪的 ranking.pt，每格都從檔案重新載入模型。標籤定義見 prune_utils.py：
免責聲明算拒絕。標籤：ok / refusal / factual / off_topic / collapse。

使用方式：
  python MLOps/prune_km_search.py run --k 10240 --m 10240
  python MLOps/prune_km_search.py decide --k 10240 --m 10240 --labels <標註 JSON 路徑>
"""

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import torch

MLOPS_DIR = Path(__file__).parent
ROOT = MLOPS_DIR.parent
sys.path.insert(0, str(MLOPS_DIR))
sys.path.insert(0, str(MLOPS_DIR / "analysis"))

from neuron_probe import DEFAULT_MODEL_NAME, DEFAULT_SYSTEM, load_seeds, load_target_model  # noqa: E402
from prune_sweep import NEGATIVE, POSITIVE, Sweep  # noqa: E402
from prune_utils import apply_prune_mask  # noqa: E402
from quality_features import repeated  # noqa: E402
from quality_pilot import GIBBERISH, RERANKER, gibberish_probs, load, relevance, response_lang  # noqa: E402
from refusal_judge import load_judge  # noqa: E402

SWEEP_DIR = ROOT / "pipelines" / "data" / "prune_sync_sweep"
DATASET = ROOT / "pipelines" / "data" / "refusal_dataset.jsonl"
OUT_DIR = ROOT / "pipelines" / "data" / "km_search"

END_M = 81920
K_FLOOR = 5120
PROBLEM_LIMIT = 8  # 27 題的 30%
PROBLEM_LABELS = ("collapse", "factual", "off_topic")
UP, DOWN = 1.25, 0.8
LABELS = ("ok", "refusal", *PROBLEM_LABELS)
# 校準結果，只記錄供比對，下一格由人工標註決定
REPEAT_BAD, RELEVANCE_BAD, GIBBERISH_BAD = 0.1442, -7.6, 0.5


def select(ranking: dict, k: int, m: int) -> torch.Tensor:
    positive, negative = ranking["positive"], ranking["negative"]
    mask = torch.zeros(positive.numel(), dtype=torch.bool)
    mask[negative[:m]] = True
    return positive[~mask[positive]][:k]


def cell_path(k: int, m: int) -> Path:
    return OUT_DIR / "cells" / f"K{k}_M{m}.jsonl"


def run(args) -> None:
    path = cell_path(args.k, args.m)
    if path.exists():
        raise SystemExit(f"{path} 已存在")
    path.parent.mkdir(parents=True, exist_ok=True)

    ranking = torch.load(SWEEP_DIR / "ranking.pt", weights_only=True)
    summary = json.loads((SWEEP_DIR / "summary.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in load_seeds([DATASET], (POSITIVE, NEGATIVE))}
    prompts = [by_id[i] for i in summary["prompts"]["tune"]]

    chosen = select(ranking, args.k, args.m)
    n_neurons = ranking["n_neurons"]
    neurons = [(i // n_neurons, i % n_neurons) for i in chosen.tolist()]

    model, tokenizer = load_target_model(None, "final", DEFAULT_MODEL_NAME)
    model.config.use_cache = True
    apply_prune_mask(model, neurons, verbose=False)
    run_args = argparse.Namespace(system=DEFAULT_SYSTEM, max_new_tokens=128)
    sweep = Sweep(model, tokenizer, load_judge("classifier", device="cpu"), run_args, OUT_DIR)
    rows = sweep.generate_judged(prompts, {"split": "tune", "list": "km", "k": args.k, "m": args.m})
    sweep.model = None
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 判定器只跑 CPU，分數寫進檔案，標註時不顯示
    for row in rows:
        row["response_lang"] = response_lang(row["response"])
    tok, gib_model = load(GIBBERISH)
    gib = gibberish_probs([r["response"] for r in rows if r["response_lang"] == "en"], tok, gib_model)
    del gib_model
    tok, rerank_model = load(RERANKER)
    rel = relevance([(r["text"], r["response"]) for r in rows], tok, rerank_model)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            row["gibberish"] = gib.get(row["response"])
            row["relevance"] = rel[(row["text"], row["response"])]
            row["repeat_all"] = round(repeated(row["response"]), 4)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scores = ranking["score"][chosen]
    print(f"K={args.k} M={args.m} 實際裁減 {len(neurons)} 平均分數 {float(scores.mean()):.3f}")
    print("=== 待標註 ===")
    for i, row in enumerate(rows, 1):
        text = row["response"].replace("\n", " ")
        tail = " ... " + text[-70:] if len(text) > 330 else ""
        print(f"#{i} {row['id']} Q:{row['text'][:60]!r} A:{text[:260]!r}{tail}")


def next_cell(k: int, m: int, problems: int) -> tuple[int, int] | None:
    if m >= END_M:
        return None
    if problems >= PROBLEM_LIMIT and k > K_FLOOR:
        return max(round(k * DOWN), K_FLOOR), m
    up_m = min(round(m * UP), END_M)
    return min(round(k * UP), up_m), up_m


def decide(args) -> None:
    rows = [json.loads(l) for l in cell_path(args.k, args.m).open(encoding="utf-8")]
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if len(labels) != len(rows) or any(label not in LABELS for label in labels):
        raise SystemExit(f"標註應為 {len(rows)} 筆，且只能用 {LABELS}")

    for row, label in zip(rows, labels):
        row["label"] = label
    with cell_path(args.k, args.m).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def auto_bad(row: dict) -> bool:
        g = row.get("gibberish")
        return (row["repeat_all"] >= REPEAT_BAD or row["relevance"] <= RELEVANCE_BAD
                or bool(g and g["noise"] + g["word salad"] > GIBBERISH_BAD))

    counts = Counter(labels)
    problems = sum(counts[label] for label in PROBLEM_LABELS)
    following = next_cell(args.k, args.m, problems)
    record = {
        "k": args.k, "m": args.m, "n": len(rows),
        **{label: counts[label] for label in LABELS},
        "problems": problems,
        "classifier_flips": sum(not r["refused"] for r in rows),
        "auto_collapse": sum((not r["refused"]) and auto_bad(r) for r in rows),
        "mean_tokens": round(sum(r["n_tokens"] for r in rows) / len(rows), 1),
        "next": following,
    }
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"cells": []}
    summary["cells"] = [c for c in summary["cells"] if (c["k"], c["m"]) != (args.k, args.m)] + [record]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    print("結束" if following is None else f"下一格 K={following[0]} M={following[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="K、M 搜尋")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "decide"):
        p = sub.add_parser(name)
        p.add_argument("--k", type=int, required=True)
        p.add_argument("--m", type=int, required=True)
        if name == "decide":
            p.add_argument("--labels", required=True)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (run if args.command == "run" else decide)(args)


if __name__ == "__main__":
    main()
