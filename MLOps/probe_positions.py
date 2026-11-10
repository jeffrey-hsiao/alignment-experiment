#!/usr/bin/env python3
"""
MLOps/probe_positions.py

逐位置的 top-k 神經元觀察：訊號到底在哪裡最強？

背景：先前的探測只看提示詞結尾（`<|im_end|> \\n <|im_start|> assistant \\n` 這 5 個
位置的平均），在那裡找不到任何能完全分離「拒絕組 / 順從組」的神經元。但這個取樣
位置包含兩個假設，兩個都沒驗證過：

  1. 「要不要拒絕」在模型開口前就已經決定 → 可能錯：決策也許是在生成前幾個
     token 的過程中才成形，而那些前向傳播原本完全沒被記錄。
  2. 平均最後 5 個位置是合理的 → 可能錯：其中 2 個是結構性的特殊 token
     （`<|im_end|>`、`<|im_start|>`），它們處理的是對話格式而非內容，平均進去
     會稀釋真正帶決策訊息的那一個位置。

本腳本把這兩件事一起測：對每一筆提示詞生成 n_steps 個 token，逐步記錄激活，
然後在**每一個位置各做一次**分離分析與 top-k 排序，排成一條位置曲線。

輸出的三個關鍵觀察：
  separation  各位置在不同容忍度下完全分離的神經元數量
  top-k 分數  各位置標準化平均差異的最大值與前 k 名
  重疊度      相鄰位置 top-k 集合的交集大小——若各位置的 top-k 幾乎不重疊，
              就支持「拒絕位於高度冗餘子空間」（見 arXiv:2607.05355）

使用方式：
  python probe_positions.py                        # 預設 8 步，全部 204 筆
  python probe_positions.py --limit 10 --n_steps 4  # 冒煙測試
  python probe_positions.py --top_k 64 --n_steps 12
"""

import argparse
import json
import sys
from pathlib import Path

import torch

MLOPS_DIR = Path(__file__).parent
ROOT = MLOPS_DIR.parent
sys.path.insert(0, str(MLOPS_DIR))

from neuron_probe import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM,
    GenerationProbe,
    GroupStats,
    load_seeds,
    load_target_model,
)

DEFAULT_SEEDS = ROOT / "data" / "seeds" / "refusal_seeds_regrouped.jsonl"
DEFAULT_OUT = ROOT / "data" / "analysis" / "probe_positions.json"

# 位置 0 是提示詞結尾（prefill 的最後一個 token，即 assistant 後的 \n），
# 位置 i>0 是第 i 個生成出來的 token。
POS_LABEL = {0: "prompt_end"}


def pos_name(index: int) -> str:
    return POS_LABEL.get(index, f"gen_step_{index}")


TOLERANCES = (1.0, 0.99, 0.95, 0.9, 0.8)


def permutation_null(active: torch.Tensor, labels: list[str], positive: str, negative: str,
                     n_perms: int, seed: int = 0) -> dict:
    """隨機打亂分組標籤，算出各容忍度下分離神經元數量的「碰巧水準」。

    這是比反向對照更嚴格的基準：反向只告訴你另一個方向有多少，置換檢定直接
    回答「如果分組根本沒有意義，會出現幾個？」只需要布林的『是否亮起』矩陣，
    因為分離條件只看 active_rate。

    active: (n_prompts, n_flat) 的 bool；labels: 每筆的真實組名。
    """
    generator = torch.Generator().manual_seed(seed)
    n = len(labels)
    n_pos = sum(1 for g in labels if g == positive)
    counts: dict[float, list[int]] = {t: [] for t in TOLERANCES}

    for _ in range(n_perms):
        perm = torch.randperm(n, generator=generator)
        pos_idx, neg_idx = perm[:n_pos], perm[n_pos:]
        pos_rate = active[pos_idx].to(torch.float32).mean(dim=0)
        neg_rate = active[neg_idx].to(torch.float32).mean(dim=0)
        for tol in TOLERANCES:
            mask = (pos_rate >= tol) & (neg_rate <= round(1.0 - tol, 4))
            counts[tol].append(int(mask.sum()))

    return {
        "n_perms": n_perms,
        "per_tolerance": [
            {"on_rate": tol, "off_rate": round(1.0 - tol, 4),
             "mean": round(sum(v) / len(v), 1), "max": max(v)}
            for tol, v in counts.items()
        ],
    }


def pooled_permutation_null(resp_sum: torch.Tensor, resp_sq: torch.Tensor, resp_n: torch.Tensor,
                            labels: list[str], positive: str, top_k: int,
                            n_perms: int, seed: int = 0) -> dict:
    """打亂「回答」的分組標籤，算出合併 top-k 分數的碰巧水準。

    必須以回答為單位打亂，不能打亂單一 token：同一筆回答的 token 共用同一個標籤
    且彼此相關，逐 token 打亂會低估碰巧水準。任何資料都選得出 top-k，這條基準線
    才能回答「這群最高值是否高於分組無意義時的最高值」。

    resp_sum / resp_sq: (n_prompts, n_flat)，每筆回答跨 token 的 Σa 與 Σa²；
    resp_n: (n_prompts,)，每筆回答實際記錄的 token 數。
    """
    def group_moments(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = resp_n[idx].sum().clamp_min(1)
        mean = resp_sum[idx].sum(dim=0) / n
        var = (resp_sq[idx].sum(dim=0) / n - mean ** 2).clamp_min(0)
        return mean, var

    generator = torch.Generator().manual_seed(seed)
    n_pos = sum(1 for g in labels if g == positive)
    max_scores: list[float] = []
    topk_means: list[float] = []
    for _ in range(n_perms):
        perm = torch.randperm(len(labels), generator=generator)
        pos_mean, pos_var = group_moments(perm[:n_pos])
        neg_mean, neg_var = group_moments(perm[n_pos:])
        score = (pos_mean - neg_mean) / ((pos_var + neg_var) / 2).sqrt().clamp_min(1e-6)
        top = score.topk(min(top_k, score.numel())).values
        max_scores.append(round(float(top[0]), 4))
        topk_means.append(round(float(top.mean()), 4))
    return {"n_perms": n_perms, "max_score": max_scores, "topk_mean_score": topk_means}


def separation_sweep(pos_stats: dict[str, GroupStats], positive: str, negative: str,
                     tolerances=TOLERANCES) -> list[dict]:
    """各容忍度下「positive 總是亮、negative 總是不亮」的神經元數量。"""
    pos, neg = pos_stats[positive], pos_stats[negative]
    out = []
    for tol in tolerances:
        mask = (pos.active_rate >= tol) & (neg.active_rate <= round(1.0 - tol, 4))
        out.append({"on_rate": tol, "off_rate": round(1.0 - tol, 4),
                    "count": int(mask.sum())})
    return out


def score_and_topk(pos_stats: dict[str, GroupStats], positive: str, negative: str,
                   top_k: int) -> tuple[torch.Tensor, list[tuple[int, int, float]]]:
    """標準化平均差異，以及前 k 名的 (層, 索引, 分數)。"""
    a, b = pos_stats[positive], pos_stats[negative]
    pooled = ((a.std ** 2 + b.std ** 2) / 2).sqrt().clamp_min(1e-6)
    score = (a.mean - b.mean) / pooled
    n_neurons = score.shape[1]
    flat = score.flatten()
    top = flat.topk(min(top_k, flat.numel()))
    entries = [
        (int(idx) // n_neurons, int(idx) % n_neurons, round(float(val), 4))
        for val, idx in zip(top.values.tolist(), top.indices.tolist())
    ]
    return score, entries


def main() -> None:
    parser = argparse.ArgumentParser(description="逐位置 top-k 神經元觀察")
    parser.add_argument("--seeds", nargs="*", default=[str(DEFAULT_SEEDS)])
    parser.add_argument("--fields", nargs="*", default=["refused", "complied"])
    parser.add_argument("--groups", nargs=2, default=["refused", "complied"],
                        metavar=("POSITIVE", "NEGATIVE"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--n_steps", type=int, default=8, help="記錄的生成步數")
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--act_threshold", type=float, default=0.0)
    parser.add_argument("--n_permutations", type=int, default=5,
                        help="置換檢定次數，0 表示關閉（可省下數百 MB 記憶體）")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    positive, negative = args.groups

    records = load_seeds(args.seeds, tuple(args.fields))
    if args.limit:
        records = records[:args.limit]
    print(f"提示詞共 {len(records)} 筆，記錄 {args.n_steps} 個生成步")

    model, tokenizer = load_target_model(args.model, args.checkpoint, args.model_name)
    model.config.use_cache = True   # 逐步生成需要 KV cache

    # stats[position][group] = GroupStats
    stats: dict[int, dict[str, GroupStats]] = {}
    responses: list[dict] = []
    # 置換檢定用的布林矩陣：active[position][prompt] = 該筆在各神經元是否亮起
    active: torch.Tensor | None = None
    labels: list[str] = []
    # 合併分析：所有回答的所有 token 放在一起統計，不分位置
    pooled: dict[str, GroupStats] = {}
    # 合併分析的置換檢定用：每筆回答跨 token 的 Σa、Σa² 與 token 數
    resp_sum: torch.Tensor | None = None
    resp_sq: torch.Tensor | None = None
    resp_n = torch.zeros(len(records))

    with GenerationProbe(model, tokenizer, system=args.system,
                         act_threshold=args.act_threshold) as probe:
        for i, record in enumerate(records, 1):
            acts, response = probe.run(record["text"], args.n_steps)
            group = record["group"]
            labels.append(group)

            if args.n_permutations and active is None:
                n_flat = acts.shape[1] * acts.shape[2]
                size_mb = (args.n_steps + 1) * len(records) * n_flat / 1e6
                print(f"  置換檢定用布林矩陣約 {size_mb:.0f} MB"
                      f"（--n_permutations 0 可關閉）", flush=True)
                active = torch.zeros((args.n_steps + 1, len(records), n_flat),
                                     dtype=torch.bool)
                resp_sum = torch.zeros((len(records), n_flat))
                resp_sq = torch.zeros((len(records), n_flat))
            if active is not None:
                flat = acts.flatten(start_dim=1)
                resp_sum[i - 1] = flat.sum(dim=0)
                resp_sq[i - 1] = (flat * flat).sum(dim=0)
                resp_n[i - 1] = acts.shape[0]
                for position in range(min(acts.shape[0], active.shape[0])):
                    active[position, i - 1] = (
                        acts[position].flatten() > args.act_threshold)

            for position in range(acts.shape[0]):
                stats.setdefault(position, {}).setdefault(group, GroupStats())
                stats[position][group].update(acts[position], args.act_threshold)
                pooled.setdefault(group, GroupStats())
                pooled[group].update(acts[position], args.act_threshold)
            responses.append({"id": record["id"], "group": group,
                              "n_steps_recorded": int(acts.shape[0]),
                              "response": response})
            if i % 20 == 0 or i == len(records):
                print(f"  已處理 {i}/{len(records)} 筆", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 逐位置分析 ───────────────────────────────────────────────────────────
    payload: dict = {
        "n_prompts":     len(records),
        "n_steps":       args.n_steps,
        "top_k":         args.top_k,
        "act_threshold": args.act_threshold,
        "pair":          {"positive": positive, "negative": negative},
        "positions":     [],
    }
    topk_sets: dict[int, set] = {}
    position_scores: dict[int, torch.Tensor] = {}

    print()
    print("=" * 76)
    print(f"逐位置分離分析（{positive} vs {negative}）")
    print("=" * 76)
    header = (f"{'位置':<14}{'n+':>5}{'n-':>5}" +
              "".join(f"{int(t * 100)}%".rjust(7) for t in TOLERANCES) +
              f"{'最高分':>9}{'前k均分':>9}{'反向80%':>9}")
    print(header)

    for position in sorted(stats):
        pos_stats = stats[position]
        if positive not in pos_stats or negative not in pos_stats:
            continue
        sweep = separation_sweep(pos_stats, positive, negative)
        reverse = separation_sweep(pos_stats, negative, positive)
        position_scores[position], entries = score_and_topk(pos_stats, positive, negative, args.top_k)
        topk_sets[position] = {(layer, idx) for layer, idx, _ in entries}
        top_mean = sum(e[2] for e in entries) / len(entries)

        print(f"{pos_name(position):<14}"
              f"{pos_stats[positive].n:>5}{pos_stats[negative].n:>5}"
              + "".join(f"{s['count']:>7}" for s in sweep)
              + f"{entries[0][2]:>9.3f}{top_mean:>9.3f}"
              + f"{reverse[-1]['count']:>9}")

        payload["positions"].append({
            "index":    position,
            "name":     pos_name(position),
            "n":        {positive: pos_stats[positive].n, negative: pos_stats[negative].n},
            "sweep":         sweep,
            "reverse_sweep": reverse,
            "max_score": entries[0][2],
            "topk_mean_score": round(top_mean, 4),
            "topk":     [{"layer": l, "neuron": n, "score": s} for l, n, s in entries],
        })

    # ── top-k 集合的重疊度 ──────────────────────────────────────────────────
    print()
    print(f"相鄰位置 top-{args.top_k} 的交集大小（越小代表各位置用的神經元越不一樣）：")
    overlaps = []
    ordered = sorted(topk_sets)
    for a, b in zip(ordered, ordered[1:]):
        shared = len(topk_sets[a] & topk_sets[b])
        overlaps.append({"from": pos_name(a), "to": pos_name(b), "shared": shared,
                         "jaccard": round(shared / len(topk_sets[a] | topk_sets[b]), 4)})
        print(f"  {pos_name(a):<14} → {pos_name(b):<14} 共同 {shared:>3} / {args.top_k}")

    if len(ordered) > 1:
        all_shared = set.intersection(*(topk_sets[p] for p in ordered))
        print(f"\n所有位置都進入 top-{args.top_k} 的神經元：{len(all_shared)} 個")
        if all_shared:
            print("  " + "  ".join(f"L{l}#{n}" for l, n in sorted(all_shared))[:600])
        payload["persistent_topk"] = [[l, n] for l, n in sorted(all_shared)]
    payload["overlaps"] = overlaps

    # ── 合併分析：回答與 token 兩個維度攤平成單一維度 ──────────────────────────
    # 裁減是把權重整欄歸零，無法只對特定題目或 token 生效，所以名單也該用不分位置的
    # 統計來選。分數是標準化差異，不隨樣本數放大，token 彼此相關不影響排序。
    if positive in pooled and negative in pooled:
        _, pooled_top = score_and_topk(pooled, positive, negative, args.top_k)
        _, pooled_reverse = score_and_topk(pooled, negative, positive, args.top_k)
        pooled_sweep = separation_sweep(pooled, positive, negative)
        pooled_reverse_sweep = separation_sweep(pooled, negative, positive)
        pooled_mean = sum(e[2] for e in pooled_top) / len(pooled_top)
        reverse_mean = sum(e[2] for e in pooled_reverse) / len(pooled_reverse)

        print()
        print("=" * 76)
        print(f"合併分析（所有回答的所有 token，{positive} "
              f"{pooled[positive].n} 個 vs {negative} {pooled[negative].n} 個 token）")
        print("=" * 76)
        print("分離數量：" + "  ".join(
            f"{int(s['on_rate'] * 100)}%={s['count']}" for s in pooled_sweep))
        print("反向分離：" + "  ".join(
            f"{int(s['on_rate'] * 100)}%={s['count']}" for s in pooled_reverse_sweep))
        print(f"前 {args.top_k} 名：最高分 {pooled_top[0][2]:.3f}，平均 {pooled_mean:.3f}")
        print(f"反向前 {args.top_k} 名：最高分 {pooled_reverse[0][2]:.3f}，平均 {reverse_mean:.3f}")
        print("  " + "  ".join(f"L{l}#{n}={s:.2f}" for l, n, s in pooled_top[:10]))

        # 每個神經元在各位置的分數：看合併分數是否只靠單一位置撐起來
        for layer, neuron, _ in pooled_top[:10]:
            print(f"  L{layer}#{neuron} 各位置："
                  + " ".join(f"{float(position_scores[p][layer, neuron]):.2f}" for p in ordered))

        print("\n合併 top-k 落在各位置 top-k 內的數量：")
        pooled_set = {(layer, idx) for layer, idx, _ in pooled_top}
        in_position = {pos_name(p): len(pooled_set & topk_sets[p]) for p in ordered}
        print("  " + "  ".join(f"{name}={count}" for name, count in in_position.items()))

        payload["pooled"] = {
            "n": {positive: pooled[positive].n, negative: pooled[negative].n},
            "sweep":         pooled_sweep,
            "reverse_sweep": pooled_reverse_sweep,
            "max_score":     pooled_top[0][2],
            "topk_mean_score": round(pooled_mean, 4),
            "reverse_max_score": pooled_reverse[0][2],
            "reverse_topk_mean_score": round(reverse_mean, 4),
            "in_position_topk": in_position,
            "topk": [
                {"layer": l, "neuron": n, "score": s,
                 "per_position": [round(float(position_scores[p][l, n]), 4) for p in ordered]}
                for l, n, s in pooled_top
            ],
            "reverse_topk": [{"layer": l, "neuron": n, "score": s}
                             for l, n, s in pooled_reverse],
        }

        if resp_sum is not None:
            null = pooled_permutation_null(resp_sum, resp_sq, resp_n, labels, positive,
                                           args.top_k, args.n_permutations)
            payload["pooled"]["permutation_null"] = null
            print(f"\n置換檢定（以回答為單位打亂 {args.n_permutations} 次）："
                  f"最高分 {min(null['max_score']):.3f}–{max(null['max_score']):.3f}，"
                  f"前 {args.top_k} 均分 {min(null['topk_mean_score']):.3f}–"
                  f"{max(null['topk_mean_score']):.3f}")
            del resp_sum, resp_sq

    if active is not None and args.n_permutations:
        print()
        print(f"置換檢定基準線（打亂分組標籤 {args.n_permutations} 次，"
              f"代表「分組無意義時會出現幾個」）：")
        print(f"{'位置':<14}" + "".join(f"{int(t * 100)}%".rjust(12) for t in TOLERANCES))
        payload["permutation_null"] = {}
        for position in sorted(stats):
            if position >= active.shape[0]:
                continue
            null = permutation_null(active[position], labels, positive, negative,
                                    args.n_permutations, seed=position)
            payload["permutation_null"][pos_name(position)] = null
            print(f"{pos_name(position):<14}"
                  + "".join(f"{e['mean']:>7.1f}/{e['max']:<4}" for e in null["per_tolerance"]))
        print("（格式：平均/最大）")
        del active

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已寫出 {out_path}")


if __name__ == "__main__":
    main()
