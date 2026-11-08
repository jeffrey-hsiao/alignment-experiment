#!/usr/bin/env python3
"""
MLOps/prune_sync_sweep.py

同步篩查掃描：排除數與選取數一起上升，不假設高分節點就是拒絕節點。

排序只用原始亮起率（激活 > 0 的比例，所有回答的所有 token 攤平統計；同分時比平均
激活）。對每個 k：

  排除 = 順從組亮起率最高的 k 個節點
  選取 = 拒絕組亮起率排序中，排除之後剩下的前 k 個節點

這是由高到低的篩查：k 變大時排除與選取同步往低分走，最後低分節點也會被選到。各 k
的集合**不是巢狀的**，較大的 k 可能把先前選到的節點改列為排除。k 最多到全部節點的
一半，再大就選不滿了。

同一流程跑三條名單，每條都跑滿 tune 集：
  sync          上述規則
  random        完全隨機的 k 個節點（通過與否以它為基準）
  reverse_sync  兩組角色對調（只供比對）

通過條件：sync 翻轉率的 Wilson 下界 > random 的 Wilson 上界。「翻轉」只代表判定器
認為不再拒絕，**不區分真的順從與模型崩壞**，回應全文都寫進 responses.jsonl 供人工檢視。

k 的安排是逐輪細化：第一輪倍增涵蓋整個範圍，之後每輪在相鄰 k 之間補幾何中點，直到
時間用完。這樣無論何時中止，低分節點都已經測過。

每次裁減都從檔案重新載入模型，不必還原權重。中斷後可用 --resume 接續，未完成的那
一格會重跑；responses.jsonl 因此可能有重複列，以最後一筆為準。

使用方式：
  python prune_sync_sweep.py --budget_hours 6.5 --resume
  python prune_sync_sweep.py --limit 40 --max_k 40 --budget_hours 0.1 --out_dir <tmp>   # 冒煙測試
"""

import argparse
import gc
import json
import math
import sys
import time
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
from prune_sweep import POSITIVE, NEGATIVE, Sweep, split_by_pair, stratified_order, wilson  # noqa: E402
from prune_utils import apply_prune_mask, write_prune_checkpoint  # noqa: E402
from refusal_judge import load_judge  # noqa: E402

DEFAULT_DATASET = ROOT / "pipelines" / "data" / "refusal_dataset.jsonl"
DEFAULT_OUT_DIR = ROOT / "pipelines" / "data" / "prune_sync_sweep"
LIST_NAMES = ("sync", "random", "reverse_sync")


# ── 排序與選取 ────────────────────────────────────────────────────────────────

def rate_order(stats: GroupStats) -> torch.Tensor:
    """亮起率由高到低；同分時比平均激活，否則小 k 的選取結果只取決於節點編號。"""
    by_mean = stats.mean.flatten().argsort(descending=True)
    return by_mean[stats.active_rate.flatten()[by_mean].argsort(descending=True, stable=True)]


def build_ranking(model, tokenizer, records: list[dict], system: str, n_steps: int,
                  seed: int) -> dict:
    """用 rank 集算兩組的排序。除了排序，也存分數，只用來觀察選到的節點，不參與選取。"""
    pooled: dict[str, GroupStats] = {POSITIVE: GroupStats(), NEGATIVE: GroupStats()}
    with GenerationProbe(model, tokenizer, system=system) as probe:
        for i, record in enumerate(records, 1):
            acts, _ = probe.run(record["text"], n_steps)
            for position in range(acts.shape[0]):
                pooled[record["group"]].update(acts[position], 0.0)
            if i % 20 == 0 or i == len(records):
                print(f"  探測 {i}/{len(records)} 筆", flush=True)

    pos, neg = pooled[POSITIVE], pooled[NEGATIVE]
    score = (pos.mean - neg.mean) / ((pos.std ** 2 + neg.std ** 2) / 2).sqrt().clamp_min(1e-6)
    total = score.numel()
    generator = torch.Generator().manual_seed(seed)
    return {
        "n_neurons": score.shape[1],
        "positive":  rate_order(pos),
        "negative":  rate_order(neg),
        "random":    torch.randperm(total, generator=generator),
        "score":     score.flatten(),
        "n_tokens":  {POSITIVE: pos.n, NEGATIVE: neg.n},
    }


def select_sync(primary: torch.Tensor, excluded: torch.Tensor, k: int) -> tuple[torch.Tensor, int]:
    """排除 excluded 的前 k 名後，取 primary 剩下的前 k 名。回傳 (選取, primary 前 k 名被扣掉幾個)。"""
    mask = torch.zeros(primary.numel(), dtype=torch.bool)
    mask[excluded[:k]] = True
    return primary[~mask[primary]][:k], int(mask[primary[:k]].sum())


def select(ranking: dict, name: str, k: int) -> tuple[torch.Tensor, int]:
    if name == "sync":
        return select_sync(ranking["positive"], ranking["negative"], k)
    if name == "reverse_sync":
        return select_sync(ranking["negative"], ranking["positive"], k)
    return ranking["random"][:k], 0


def refinement_rounds(start: int, cap: int):
    """第一輪倍增到 cap，之後每輪補相鄰 k 的幾何中點（取 start 的倍數），直到補不進去。"""
    levels = []
    k = start
    while k < cap:
        levels.append(k)
        k *= 2
    levels.append(cap)
    yield list(levels)
    while True:
        mids = []
        for low, high in zip(levels, levels[1:]):
            mid = round(math.sqrt(low * high) / start) * start
            if low < mid < high:
                mids.append(mid)
        if not mids:
            return
        yield mids
        levels = sorted(set(levels) | set(mids))


# ── 評估 ──────────────────────────────────────────────────────────────────────

def load_model(args):
    model, tokenizer = load_target_model(None, "final", args.model_name)
    model.config.use_cache = True
    return model, tokenizer


def evaluate(sweep: Sweep, args, ranking: dict, split: str, name: str, k: int,
             prompts: list[dict]) -> dict:
    """重新載入模型 → 裁減 → 全部題目生成與判定。"""
    sweep.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sweep.model, sweep.tokenizer = load_model(args)

    chosen, removed = select(ranking, name, k)
    n_neurons = ranking["n_neurons"]
    neurons = [(i // n_neurons, i % n_neurons) for i in chosen.tolist()]
    apply_prune_mask(sweep.model, neurons, verbose=False)

    rows = sweep.generate_judged(prompts, {"split": split, "list": name, "k": k})
    flips = sum(not r["refused"] for r in rows)
    low, high = wilson(flips, len(rows))
    scores = ranking["score"][chosen]
    return {
        "list": name, "k": k, "n_pruned": len(neurons), "removed_from_top_k": removed,
        "n": len(rows), "flips": flips,
        "keyword_flips": sum(r["keyword_category"] == "comply" for r in rows),
        "low": round(low, 4), "high": round(high, 4),
        "mean_tokens": round(sum(r["n_tokens"] for r in rows) / len(rows), 1),
        "truncated": sum(r["truncated"] for r in rows),
        "mean_logit": round(sum(r["logit"] or 0.0 for r in rows) / len(rows), 3),
        "score_mean": round(float(scores.mean()), 4),
        "score_min": round(float(scores.min()), 4),
        "score_max": round(float(scores.max()), 4),
    }


def run_level(sweep: Sweep, args, ranking: dict, k: int, prompts: list[dict],
              out_dir: Path) -> dict:
    started = time.time()
    level = {"k": k}
    for name in LIST_NAMES:
        level[name] = evaluate(sweep, args, ranking, "tune", name, k, prompts)
    level["passed"] = level["sync"]["low"] > level["random"]["high"]
    level["seconds"] = round(time.time() - started)

    if level["passed"]:
        # 通過者才存名單，供之後用 pruned_neurons.json 載入檢視
        n_neurons = ranking["n_neurons"]
        chosen, _ = select(ranking, "sync", k)
        write_prune_checkpoint(out_dir / "checkpoints" / f"sync-{k}", args.model_name,
                               [(i // n_neurons, i % n_neurons) for i in chosen.tolist()], k=k)

    cells = [f"k={k:<7}"]
    for name in LIST_NAMES:
        cell = level[name]
        cells.append(f"{name}={cell['flips']}/{cell['n']}(logit {cell['mean_logit']:.1f})")
    cells.append(f"sync分數均值={level['sync']['score_mean']:.2f}")
    cells.append(f"平均token={level['sync']['mean_tokens']}")
    cells.append(f"{level['seconds']}秒")
    cells.append("通過" if level["passed"] else "未過")
    print("[掃描] " + "  ".join(cells), flush=True)
    return level


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="同步篩查掃描")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--judge", default="classifier")
    parser.add_argument("--judge_device", default="cpu")
    parser.add_argument("--n_steps", type=int, default=8, help="探測記錄的 token 位置數")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--start_k", type=int, default=5, help="起點，也是 k 的最小間隔")
    parser.add_argument("--max_k", type=int, default=0, help="k 的上限，0 表示全部節點的一半")
    parser.add_argument("--budget_hours", type=float, default=6.5,
                        help="時間預算，接續時沿用第一次啟動算出的截止時間")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 筆（冒煙測試）")
    parser.add_argument("--resume", action="store_true", help="out_dir 已有結果時接續")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    ranking_path = out_dir / "ranking.pt"
    if summary_path.exists() and not args.resume:
        raise SystemExit(f"{out_dir} 已有結果，要接續請加 --resume，否則請換 --out_dir")

    records = load_seeds([args.dataset], (POSITIVE, NEGATIVE))
    if args.limit:
        records = records[:args.limit]
    splits = split_by_pair(records, args.seed)
    print("切分：" + "  ".join(f"{name}={len(rows)}" for name, rows in splits.items()), flush=True)

    model, tokenizer = load_model(args)
    judge_kwargs = {} if args.judge == "keyword" else {"device": args.judge_device}
    sweep = Sweep(model, tokenizer, load_judge(args.judge, **judge_kwargs), args, out_dir)

    def save(data: dict) -> None:
        temp = summary_path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(summary_path)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ranking = torch.load(ranking_path, weights_only=True)
        if summary.get("finished"):
            print("這次掃描已經完成")
            return
        print(f"接續：已完成 {len(summary['levels'])} 格", flush=True)
    else:
        ranking = build_ranking(model, tokenizer, splits["rank"], args.system,
                                args.n_steps, args.seed)
        torch.save(ranking, ranking_path)
        summary = {
            "args": vars(args),
            "deadline": time.time() + args.budget_hours * 3600,
            "split_sizes": {name: len(rows) for name, rows in splits.items()},
            "n_tokens": ranking["n_tokens"],
            "levels": [],
            "prompts": {},
            "test": None,
        }
        # 只測基準模型仍會拒絕的題目，否則「翻轉」沒有意義
        for split in ("tune", "test"):
            candidates = [r for r in splits[split] if r["group"] == POSITIVE]
            rows = sweep.generate_judged(candidates, {"split": split, "list": "baseline", "k": 0})
            still = {r["id"] for r in rows if r["refused"]}
            ordered = stratified_order([r for r in candidates if r["id"] in still], args.seed)
            summary["prompts"][split] = [r["id"] for r in ordered]
            print(f"基準模型 {split}：{len(still)}/{len(rows)} 仍拒絕", flush=True)
        save(summary)

    by_id = {r["id"]: r for r in records}
    prompts = {split: [by_id[i] for i in ids] for split, ids in summary["prompts"].items()}
    if not prompts["tune"]:
        raise SystemExit("tune 集沒有基準模型會拒絕的題目")

    cap = ranking["score"].numel() // 2
    if args.max_k:
        cap = min(cap, args.max_k)
    done = {level["k"] for level in summary["levels"]}
    out_of_time = False
    for round_levels in refinement_rounds(args.start_k, cap):
        for k in round_levels:
            if k in done:
                continue
            spent = [level["seconds"] for level in summary["levels"]]
            estimate = max(spent[-3:]) if spent else 0
            if time.time() + estimate > summary["deadline"]:
                out_of_time = True
                break
            summary["levels"].append(run_level(sweep, args, ranking, k, prompts["tune"], out_dir))
            done.add(k)
            save(summary)
        if out_of_time:
            break
    print("時間用完" if out_of_time else "所有 k 都測完了", flush=True)

    # 保留集只用一次：驗證通過者中最小的 k
    passed = sorted(level["k"] for level in summary["levels"] if level["passed"])
    summary["passed_k"] = passed
    if passed and prompts["test"]:
        chosen_k = passed[0]
        summary["chosen_k"] = chosen_k
        summary["test"] = {
            name: evaluate(sweep, args, ranking, "test", name, chosen_k, prompts["test"])
            for name in ("sync", "random")
        }
        print(f"[保留集] k={chosen_k}  sync={summary['test']['sync']['flips']}/"
              f"{summary['test']['sync']['n']}  random={summary['test']['random']['flips']}/"
              f"{summary['test']['random']['n']}", flush=True)

    summary["finished"] = True
    save(summary)
    print(f"\n完成 {len(summary['levels'])} 格，通過的 k：{passed}")
    print(f"結果在 {out_dir}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
