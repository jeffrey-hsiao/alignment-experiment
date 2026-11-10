#!/usr/bin/env python3
"""
MLOps/prune_sweep.py

自適應裁減掃描：裁減數量不預設，從 k=1 開始倍增，看 k 多大時拒絕才開始翻轉。

名單來自合併分析（所有回答的所有 token 攤平成單一維度，見 probe_positions.py）的
標準化平均差異。同一流程跑三條名單，比較「k 對翻轉率」曲線：

  top      拒絕組激活較高的節點，由高到低
  random   與分組無明顯關聯的節點，隨機順序（基準線，通過與否以它為準）
  reverse  順從組激活較高的節點，由高到低（只供比對）

資料按 pair_id 切成三份，避免配對題互相洩漏：
  rank  算名單（名單完全不看 tune / test）
  tune  決定 k
  test  只對最後選出的 k 驗證一次

每個 k 的抽樣由小到大（預設 4 → 8 → 16 → 全部）：某階段一題都沒翻轉就停止，
有翻轉才擴大。通過條件：top 的翻轉率 Wilson 下界 > random 的 Wilson 上界。翻轉率
不保證隨 k 單調，所以掃完整條 k 才找第一個通過者，再二分找更小的 k。

「翻轉」只代表判定器認為不再拒絕，**不區分真的順從與模型崩壞**。所有回應全文都
寫進 responses.jsonl，供人工檢視與日後換判定方式重算。

裁減只存在於記憶體且每次都還原，不會寫出任何模型權重。

使用方式：
  python prune_sweep.py
  python prune_sweep.py --limit 24 --max_k 4 --n_permutations 2   # 冒煙測試
"""

import argparse
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

MLOPS_DIR = Path(__file__).parent
ROOT = MLOPS_DIR.parent
sys.path.insert(0, str(MLOPS_DIR))

from behavior_regroup import generate_all  # noqa: E402
from neuron_probe import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM,
    GenerationProbe,
    GroupStats,
    load_seeds,
    load_target_model,
)
from probe_positions import pooled_permutation_null, score_and_topk  # noqa: E402
from prune_utils import _weight_of, apply_prune_mask, find_mlp_down_projs  # noqa: E402
from refusal_judge import KeywordJudge, load_judge  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "seeds" / "refusal_dataset.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "analysis" / "prune_sweep"
POSITIVE, NEGATIVE = "refused", "complied"


# ── 資料切分 ──────────────────────────────────────────────────────────────────

def split_by_pair(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """按 pair_id 切成 rank 60% / tune 20% / test 20%。"""
    pair_ids = sorted({r["pair_id"] for r in records})
    random.Random(seed).shuffle(pair_ids)
    n_rank = round(len(pair_ids) * 0.6)
    n_tune = round(len(pair_ids) * 0.2)
    assign = {pid: "rank" for pid in pair_ids[:n_rank]}
    assign.update({pid: "tune" for pid in pair_ids[n_rank:n_rank + n_tune]})
    assign.update({pid: "test" for pid in pair_ids[n_rank + n_tune:]})
    splits: dict[str, list[dict]] = {"rank": [], "tune": [], "test": []}
    for record in records:
        splits[assign[record["pair_id"]]].append(record)
    return splits


def stratified_order(records: list[dict], seed: int) -> list[dict]:
    """各領域輪流出題，讓小樣本階段也涵蓋不同拒絕機制。順序固定，各 k 可配對比較。"""
    by_domain: dict[str, list[dict]] = {}
    for record in records:
        by_domain.setdefault(record.get("domain", ""), []).append(record)
    rng = random.Random(seed)
    for rows in by_domain.values():
        rng.shuffle(rows)
    ordered: list[dict] = []
    while any(by_domain.values()):
        for domain in sorted(by_domain):
            if by_domain[domain]:
                ordered.append(by_domain[domain].pop())
    return ordered


# ── 名單 ──────────────────────────────────────────────────────────────────────

def build_lists(model, tokenizer, records: list[dict], system: str, n_steps: int,
                n_perms: int, seed: int) -> tuple[dict, dict]:
    """用 rank 集算合併分數，回傳 {名單名: [(層, 索引)]} 與摘要。"""
    labels = [r["group"] for r in records]
    pooled: dict[str, GroupStats] = {}
    resp_sum: torch.Tensor | None = None
    resp_sq: torch.Tensor | None = None
    resp_n = torch.zeros(len(records))

    with GenerationProbe(model, tokenizer, system=system) as probe:
        for i, record in enumerate(records, 1):
            acts, _ = probe.run(record["text"], n_steps)
            flat = acts.flatten(start_dim=1)
            if resp_sum is None:
                resp_sum = torch.zeros((len(records), flat.shape[1]))
                resp_sq = torch.zeros((len(records), flat.shape[1]))
            resp_sum[i - 1] = flat.sum(dim=0)
            resp_sq[i - 1] = (flat * flat).sum(dim=0)
            resp_n[i - 1] = acts.shape[0]
            for position in range(acts.shape[0]):
                pooled.setdefault(record["group"], GroupStats())
                pooled[record["group"]].update(acts[position], 0.0)
            if i % 20 == 0 or i == len(records):
                print(f"  探測 {i}/{len(records)} 筆", flush=True)

    score, _ = score_and_topk(pooled, POSITIVE, NEGATIVE, 64)
    null = pooled_permutation_null(resp_sum, resp_sq, resp_n, labels, POSITIVE, 64, n_perms)
    # 分數上限：低於打亂標籤時的最高分，就無法與雜訊區分
    null_max = max(null["max_score"])

    n_neurons = score.shape[1]
    flat_score = score.flatten()
    to_pair = lambda idx: (int(idx) // n_neurons, int(idx) % n_neurons)

    top_idx = torch.nonzero(flat_score > null_max, as_tuple=True)[0]
    top_idx = top_idx[flat_score[top_idx].argsort(descending=True)]
    reverse_idx = torch.nonzero(flat_score < -null_max, as_tuple=True)[0]
    reverse_idx = reverse_idx[flat_score[reverse_idx].argsort()]

    neutral = torch.nonzero(flat_score.abs() <= null_max, as_tuple=True)[0]
    generator = torch.Generator().manual_seed(seed)
    neutral = neutral[torch.randperm(neutral.numel(), generator=generator)][:top_idx.numel()]

    lists = {
        "top":     [to_pair(i) for i in top_idx.tolist()],
        "random":  [to_pair(i) for i in neutral.tolist()],
        "reverse": [to_pair(i) for i in reverse_idx.tolist()],
    }
    summary = {
        "n_tokens":  {g: s.n for g, s in pooled.items()},
        "null":      null,
        "null_max":  null_max,
        "sizes":     {name: len(v) for name, v in lists.items()},
        "scores_top": [round(float(flat_score[i]), 4) for i in top_idx[:64].tolist()],
    }
    return lists, summary


# ── 裁減與判定 ────────────────────────────────────────────────────────────────

@contextmanager
def pruned(model, neurons: list[tuple[int, int]]):
    """暫時裁減，離開時還原被歸零的權重欄，不必重新載入模型。"""
    down_projs = find_mlp_down_projs(model)
    by_layer: dict[int, list[int]] = {}
    for layer, idx in neurons:
        by_layer.setdefault(layer, []).append(idx)
    saved = {layer: _weight_of(down_projs[layer])[:, indices].clone()
             for layer, indices in by_layer.items()}
    apply_prune_mask(model, neurons, verbose=False)
    try:
        yield
    finally:
        with torch.no_grad():
            for layer, indices in by_layer.items():
                _weight_of(down_projs[layer])[:, indices] = saved[layer]


def wilson(flips: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """翻轉率的 Wilson 信賴區間。小樣本下比常態近似可靠。"""
    if n == 0:
        return 0.0, 1.0
    p = flips / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def stage_sizes(total: int, start: int) -> list[int]:
    sizes = []
    size = start
    while size < total:
        sizes.append(size)
        size *= 2
    return sizes + [total]


class Sweep:
    def __init__(self, model, tokenizer, judge, args, out_dir: Path):
        self.model      = model
        self.tokenizer  = tokenizer
        self.judge      = judge
        self.keyword    = KeywordJudge()
        self.args       = args
        self.responses  = (out_dir / "responses.jsonl").open("a", encoding="utf-8")

    def generate_judged(self, records: list[dict], tag: dict) -> list[dict]:
        """生成並判定；回應全文立刻寫檔，訓練中斷也不會遺失。"""
        details = generate_all(self.model, self.tokenizer, records, self.args.system,
                               self.args.max_new_tokens)
        texts = [d["response"] for d in details]
        rows = []
        for detail, verdict, kw in zip(details, self.judge.judge_batch(texts),
                                       self.keyword.judge_batch(texts)):
            row = {
                **tag,
                "id":        detail["id"],
                "domain":    detail.get("domain"),
                "text":      detail["text"],
                "response":  detail["response"],
                "n_tokens":  detail["n_tokens"],
                "truncated": detail["truncated"],
                "refused":   verdict["refused"],
                "score":     verdict["score"],
                "logit":     verdict.get("logit"),
                "keyword_category": kw["category"],
            }
            rows.append(row)
            self.responses.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.responses.flush()
        return rows

    def evaluate(self, split: str, name: str, neurons: list, k: int,
                 prompts: list[dict], force_full: bool = False) -> dict:
        rows: list[dict] = []
        tag = {"split": split, "list": name, "k": k}
        with pruned(self.model, neurons[:k]):
            for size in stage_sizes(len(prompts), self.args.start_sample):
                rows += self.generate_judged(prompts[len(rows):size], tag)
                if not force_full and not any(not r["refused"] for r in rows):
                    break
        flips = sum(not r["refused"] for r in rows)
        kw_flips = sum(r["keyword_category"] == "comply" for r in rows)
        low, high = wilson(flips, len(rows))
        return {"list": name, "k": k, "n": len(rows), "flips": flips, "keyword_flips": kw_flips,
                "low": round(low, 4), "high": round(high, 4),
                "full": len(rows) == len(prompts),
                "mean_tokens": round(sum(r["n_tokens"] for r in rows) / len(rows), 1)}

    def level(self, split: str, lists: dict, k: int, prompts: list[dict]) -> dict:
        """同一個 k 跑三條名單。top 走到全部抽樣時，對照也要跑滿才能比較區間。"""
        top = self.evaluate(split, "top", lists["top"], k, prompts)
        rand = self.evaluate(split, "random", lists["random"], k, prompts,
                             force_full=top["full"])
        result = {"k": k, "top": top, "random": rand, "reverse": None}
        if k <= len(lists["reverse"]):
            result["reverse"] = self.evaluate(split, "reverse", lists["reverse"], k, prompts,
                                              force_full=top["full"])
        result["passed"] = bool(top["full"] and rand["full"] and top["low"] > rand["high"])
        cells = [f"k={k:<6}"]
        for key in ("top", "random", "reverse"):
            cell = result[key]
            cells.append(f"{key}={cell['flips']}/{cell['n']}" if cell else f"{key}=-")
        cells.append(f"top下界={top['low']:.3f} random上界={rand['high']:.3f}")
        cells.append(f"平均token={top['mean_tokens']}")
        cells.append("通過" if result["passed"] else "未過")
        print("[掃描] " + "  ".join(cells), flush=True)
        return result


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="自適應裁減掃描")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--judge", default="classifier",
                        help="主判定器，預設與資料集分組一致；keyword 一律另外記錄")
    parser.add_argument("--judge_device", default="cpu")
    parser.add_argument("--n_steps", type=int, default=8, help="探測記錄的 token 位置數")
    parser.add_argument("--n_permutations", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--start_sample", type=int, default=4, help="每個 k 的起始抽樣題數")
    parser.add_argument("--max_k", type=int, default=0, help="k 的上限，0 表示只受分數基準線限制")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 筆（冒煙測試）")
    args = parser.parse_args()

    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "responses.jsonl").exists():
        raise SystemExit(f"{out_dir} 已有結果，請換 --out_dir，避免不同次掃描混在一起")

    records = load_seeds([args.dataset], (POSITIVE, NEGATIVE))
    if args.limit:
        records = records[:args.limit]
    splits = split_by_pair(records, args.seed)
    print("切分：" + "  ".join(f"{name}={len(rows)}" for name, rows in splits.items()), flush=True)

    model, tokenizer = load_target_model(None, "final", args.model_name)
    model.config.use_cache = True
    judge_kwargs = {} if args.judge == "keyword" else {"device": args.judge_device}
    judge = load_judge(args.judge, **judge_kwargs)

    lists, list_summary = build_lists(model, tokenizer, splits["rank"], args.system,
                                      args.n_steps, args.n_permutations, args.seed)
    print(f"基準線 {list_summary['null_max']:.3f}，名單大小 {list_summary['sizes']}", flush=True)
    (out_dir / "lists.json").write_text(json.dumps(
        {"summary": list_summary, "lists": lists}, ensure_ascii=False), encoding="utf-8")
    if not lists["top"]:
        raise SystemExit("沒有節點高於基準線，無法掃描")

    sweep = Sweep(model, tokenizer, judge, args, out_dir)
    summary: dict = {
        "args":  {k: v for k, v in vars(args).items()},
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "lists": list_summary,
        "levels": [],
        "bisect": [],
        "test": None,
    }

    def save() -> None:
        summary["elapsed_s"] = round(time.time() - started)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 只測基準模型仍會拒絕的題目，否則「翻轉」沒有意義
    prompts: dict[str, list[dict]] = {}
    baseline_rows: list[dict] = []
    for split in ("tune", "test"):
        candidates = [r for r in splits[split] if r["group"] == POSITIVE]
        rows = sweep.generate_judged(candidates, {"split": split, "list": "baseline", "k": 0})
        baseline_rows += rows
        still = {r["id"] for r in rows if r["refused"]}
        prompts[split] = stratified_order(
            [r for r in candidates if r["id"] in still], args.seed)
        summary[f"baseline_{split}"] = {"n": len(rows), "refused": len(still)}
        print(f"基準模型 {split}：{len(still)}/{len(rows)} 仍拒絕", flush=True)
    save()
    if not prompts["tune"]:
        raise SystemExit("tune 集沒有基準模型會拒絕的題目")

    # 倍增掃描，到基準線為止
    cap = len(lists["top"])
    if args.max_k:
        cap = min(cap, args.max_k)
    schedule = []
    k = 1
    while k < cap:
        schedule.append(k)
        k *= 2
    schedule.append(cap)

    for k in schedule:
        summary["levels"].append(sweep.level("tune", lists, k, prompts["tune"]))
        save()

    # 第一個通過者與前一格之間二分，找更小的 k
    first = next((i for i, lv in enumerate(summary["levels"]) if lv["passed"]), None)
    chosen = None
    if first is not None:
        chosen = summary["levels"][first]["k"]
        low = summary["levels"][first - 1]["k"] if first > 0 else 0
        while chosen - low > max(1, low // 10):
            mid = (low + chosen) // 2
            result = sweep.level("tune", lists, mid, prompts["tune"])
            summary["bisect"].append(result)
            save()
            if result["passed"]:
                chosen = mid
            else:
                low = mid

    # 保留集只用一次，且只驗證最後選出的 k
    if chosen is not None and prompts["test"]:
        summary["chosen_k"] = chosen
        test = {
            name: sweep.evaluate("test", name, lists[name], chosen, prompts["test"],
                                 force_full=True)
            for name in ("top", "random")
        }
        summary["test"] = test
        print(f"[保留集] k={chosen}  top={test['top']['flips']}/{test['top']['n']}"
              f"  random={test['random']['flips']}/{test['random']['n']}", flush=True)

    # 所有裁減都還原後，greedy 生成應與掃描前逐字相同；不同就代表還原有問題，結果不可信
    recheck = prompts["tune"][:args.start_sample]
    before = {r["id"]: r["response"] for r in baseline_rows}
    after = sweep.generate_judged(recheck, {"split": "tune", "list": "restore_check", "k": 0})
    summary["restore_ok"] = all(r["response"] == before[r["id"]] for r in after)
    print(f"權重還原檢查：{'一致' if summary['restore_ok'] else '不一致，結果不可信'}", flush=True)

    passed = [lv["k"] for lv in summary["levels"] + summary["bisect"] if lv["passed"]]
    summary["passed_k"] = sorted(passed)
    save()
    print(f"\n通過的 k（共 {len(passed)} 個）：{sorted(passed)}")
    print(f"總耗時 {summary['elapsed_s']} 秒，結果在 {out_dir}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
