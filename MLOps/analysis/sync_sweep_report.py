"""由 prune_sync_sweep.py 的結果重建報告：K/M 表、粗略崩壞統計、配對檢定、候選起點。不需要 GPU。

用法：python MLOps/analysis/sync_sweep_report.py pipelines/data/prune_sync_sweep

粗略崩壞規則只是參考：長英文回答的字元種類少，會被誤判，不能取代人工檢視。
"""
import json
import re
import sys
from collections import Counter
from math import comb
from pathlib import Path

import torch

out_dir = Path(sys.argv[1])
summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
ranking = torch.load(out_dir / "ranking.pt", weights_only=True)
positive, negative, score = ranking["positive"], ranking["negative"], ranking["score"]
total = score.numel()
rank_of = torch.zeros(total, dtype=torch.long)
rank_of[positive] = torch.arange(total)

# 接續執行可能重跑同一格，以最後一筆為準
rows = {}
for line in (out_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    rows[(row["split"], row["list"], row["k"], row["id"])] = row


def select(k: int, m: int) -> torch.Tensor:
    mask = torch.zeros(total, dtype=torch.bool)
    mask[negative[:m]] = True
    return positive[~mask[positive]][:k]


def looks_broken(text: str) -> bool:
    repeated = 0.0
    if len(text) >= 16:
        repeated = max(Counter(text[i:i + 6] for i in range(len(text) - 5)).values()) * 6 / len(text)
    letters = len(re.findall(r"[一-鿿A-Za-z]", text))
    return (repeated > 0.25 or len(set(text)) / max(1, len(text)) < 0.12
            or letters / max(1, len(text)) < 0.5)


def cell(split: str, name: str, k: int) -> dict:
    return {i: r for (s, n, kk, i), r in rows.items() if s == split and n == name and kk == k}


def mcnemar(split: str, k: int) -> tuple[int, int, float]:
    a, b = cell(split, "sync", k), cell(split, "random", k)
    only_a = sum(not a[i]["refused"] and b[i]["refused"] for i in a)
    only_b = sum(a[i]["refused"] and not b[i]["refused"] for i in a)
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, j) for j in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, round(p, 4)


print("K M 被扣數 最深名次 sync平均分數 sync翻轉 sync粗略崩壞 sync_logit random翻轉 反向翻轉 通過")
for level in sorted(summary["levels"], key=lambda item: item["k"]):
    k, sync = level["k"], level["sync"]
    chosen = select(k, k)
    broken = sum(looks_broken(r["response"]) for r in cell("tune", "sync", k).values())
    print(k, k, sync["removed_from_top_k"], int(rank_of[chosen].max()), sync["score_mean"],
          sync["flips"], broken, sync["mean_logit"], level["random"]["flips"],
          level["reverse_sync"]["flips"], level["passed"])

print("\n配對檢定（僅 sync 翻, 僅 random 翻, p）")
for split, k in (("tune", 20480), ("tune", 40960), ("tune", 81920), ("test", 40960)):
    print(split, k, mcnemar(split, k))
print("保留集：", json.dumps(summary["test"], ensure_ascii=False))

print("\n固定 M 的候選（與已測格子的重疊比例）")
done = {k: set(select(k, k).tolist()) for k in (20480, 40960)}
for m in (20480, 40960, 81920):
    for k in (5120, 10240, 20480, 30720, 40960):
        chosen = select(k, m)
        members = set(chosen.tolist())
        print(f"K={k} M={m} 平均分數={score[chosen].mean():.3f} 最深={int(rank_of[chosen].max())} "
              f"重疊(20480,20480)={len(members & done[20480]) / k:.2f} "
              f"重疊(40960,40960)={len(members & done[40960]) / k:.2f}")
