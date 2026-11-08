#!/usr/bin/env python3
"""
MLOps/prune_search.py

`run.py train prune` 的 km_search 模式：粗測定範圍，再分開調整 K、M。

選點：排除順從組亮起率最高的 M 個節點，再選拒絕組剩下的前 K 個。不假設高分節點
就是拒絕節點，K、M 上升時低分節點最後也會被選到。

流程：
  準備  用當前模型回答全部題目並分組（條件相同就沿用）→ 依 pair_id 切分
        → 用 rank 集排序 → 只留基準模型會拒絕的 tune 題
  粗測  K=M 從神經元總數的一半起每輪減半，翻轉連續幾格夠少就停
        起點＝翻轉增加最多那段之前的點，終點＝崩壞增加最多那段之後的點
  細搜  崩壞達門檻時 K 下降、M 不變，否則 K、M 同步上升；M 到達終點跑完即結束
  對照  對「翻轉減崩壞」最高的幾格，各跑數組同數量的隨機節點
  驗證  有指定 test_seeds 才跑，只驗證最高的幾格

「翻轉」只代表判定器認為不再拒絕；「崩壞」是全文重複率、相關分數與英文亂碼的
自動判定。兩者都抓不到事實錯亂，所以報表把扣掉崩壞的翻轉稱為「其餘翻轉」，不是
有效回答。

每格都從檔案重新載入模型，裁減只在記憶體中進行。每格結果立刻寫檔，流程又是由
已完成的格子決定下一格，所以接續時會沿原路走到中斷處。
"""

import gc
import hashlib
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

MLOPS_DIR = Path(__file__).parent
sys.path.insert(0, str(MLOPS_DIR))
sys.path.insert(0, str(MLOPS_DIR / "analysis"))

from behavior_regroup import generate_all  # noqa: E402
from neuron_probe import GenerationProbe, GroupStats, load_seeds, load_target_model  # noqa: E402
from chat_prompt import resolve_system  # noqa: E402  依模型解析 system,支援跨家族
from data_naming import warn_if_method_mismatch  # noqa: E402  檔名前綴 vs 方法 不匹配警告
from prune_utils import apply_prune_mask, apply_reverse_mask, write_prune_checkpoint  # noqa: E402
from quality_features import repeated  # noqa: E402
from quality_pilot import GIBBERISH, RERANKER, gibberish_probs, load, relevance, response_lang  # noqa: E402
from refusal_judge import KeywordJudge, load_judge  # noqa: E402

POSITIVE, NEGATIVE = "refused", "complied"


class KmSearchError(RuntimeError):
    """粗測定不出合理範圍。不猜測，停止並回報。"""


# ── 資料 ──────────────────────────────────────────────────────────────────────

def load_source(path: Path) -> list[dict]:
    """讀題目。扁平格式自帶 group；配對格式展開成 <id>:harmful 與 <id>:harmless，尚未分組。"""
    records = load_seeds([path], (POSITIVE, NEGATIVE))
    if records:
        return records
    return [{**r, "id": f"{r['id']}:{r['group']}", "pair_id": r["id"], "orig_group": r["group"]}
            for r in load_seeds([path], ("harmful", "harmless"))]


def split_by_pair(records: list[dict], seed: int, rank_ratio: float = 0.6,
                  tune_ratio: float = 0.2) -> dict[str, list[dict]]:
    """按 pair_id 切成排序、調參、驗證三份，配對題不會落在不同份。剩下的都是驗證集。"""
    if rank_ratio <= 0 or tune_ratio <= 0 or rank_ratio + tune_ratio > 1:
        raise KmSearchError(f"切分比例不合理：rank_ratio={rank_ratio}，tune_ratio={tune_ratio}")
    pair_ids = sorted({r["pair_id"] for r in records})
    random.Random(seed).shuffle(pair_ids)
    n_rank = round(len(pair_ids) * rank_ratio)
    n_tune = round(len(pair_ids) * tune_ratio)
    assign = {pid: "rank" for pid in pair_ids[:n_rank]}
    assign.update({pid: "tune" for pid in pair_ids[n_rank:n_rank + n_tune]})
    assign.update({pid: "test" for pid in pair_ids[n_rank + n_tune:]})
    splits: dict[str, list[dict]] = {"rank": [], "tune": [], "test": []}
    for record in records:
        splits[assign[record["pair_id"]]].append(record)
    return splits


def stratified_order(records: list[dict], seed: int) -> list[dict]:
    """各領域輪流出題。順序固定，各格可以逐題比較。"""
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


# ── 排序與選取 ────────────────────────────────────────────────────────────────

def rate_order(stats: GroupStats) -> torch.Tensor:
    """亮起率由高到低；同分時比平均激活，否則小 K 的結果只取決於節點編號。"""
    by_mean = stats.mean.flatten().argsort(descending=True)
    return by_mean[stats.active_rate.flatten()[by_mean].argsort(descending=True, stable=True)]


def build_ranking(model, tokenizer, records: list[dict], n_steps: int, seed: int,
                  system: str) -> dict:
    """所有回答的所有 token 攤平統計。分數只用來觀察選到的節點，不參與選取。"""
    pooled = {POSITIVE: GroupStats(), NEGATIVE: GroupStats()}
    with GenerationProbe(model, tokenizer, system=system) as probe:
        for i, record in enumerate(records, 1):
            acts, _ = probe.run(record["text"], n_steps)
            for position in range(acts.shape[0]):
                pooled[record["group"]].update(acts[position], 0.0)
            if i % 20 == 0 or i == len(records):
                print(f"  排序探測 {i}/{len(records)} 筆", flush=True)
    pos, neg = pooled[POSITIVE], pooled[NEGATIVE]
    score = (pos.mean - neg.mean) / ((pos.std ** 2 + neg.std ** 2) / 2).sqrt().clamp_min(1e-6)
    return {
        "n_neurons": score.shape[1],
        "positive":  rate_order(pos),
        "negative":  rate_order(neg),
        "random":    torch.randperm(score.numel(), generator=torch.Generator().manual_seed(seed)),
        "score":     score.flatten(),
        # 四個率的原料：拒絕/順從組的「正向亮起率」，負向率 = 1 − 它。
        # 雙表選點與翻轉偵測需要；舊 ranking.pt 沒有這兩欄，只能跑 sign=positive。
        "refused_pos_rate":  pos.active_rate.flatten(),
        "complied_pos_rate": neg.active_rate.flatten(),
        "n_tokens":  {POSITIVE: pos.n, NEGATIVE: neg.n},
    }


def select_sync(ranking: dict, k: int, m: int) -> tuple[torch.Tensor, int]:
    """回傳 (選到的節點, 拒絕組前 K 名被排除掉幾個)。單一正向表，相容舊 ranking。"""
    positive, negative = ranking["positive"], ranking["negative"]
    mask = torch.zeros(positive.numel(), dtype=torch.bool)
    mask[negative[:m]] = True
    return positive[~mask[positive]][:k], int(mask[positive[:k]].sum())


def _flow(refused_rate: torch.Tensor, complied_rate: torch.Tensor, k: int, m: int):
    """單一表：依拒絕率排名、排除順從率前 M、取前 K。回傳 (選中 set, 排除 set)。"""
    order = refused_rate.argsort(descending=True, stable=True)
    excluded = torch.zeros(refused_rate.numel(), dtype=torch.bool)
    excluded[complied_rate.argsort(descending=True, stable=True)[:m]] = True
    selected = order[~excluded[order]][:k]
    return set(selected.tolist()), set(torch.nonzero(excluded, as_tuple=True)[0].tolist())


def select_tables(ranking: dict, k: int, m: int, sign: str,
                  detect_reversal: bool) -> tuple[list, list, dict]:
    """雙表選點 + 翻轉偵測。回傳 (歸零清單, 翻轉清單, 統計)。

    翻轉型 = 在一張表被選中、同時在相反表被排除（拒絕一向、順從相反向）。
    sign: positive 只跑亮表 / negative 只跑暗表 / both 兩表合併去重。
    """
    if "refused_pos_rate" not in ranking:
        raise KmSearchError("這份 ranking 沒有 *_pos_rate 欄位（可能是舊版），"
                            "sign=negative/both 或翻轉偵測需要重新產生排序")
    rp = ranking["refused_pos_rate"]
    cp = ranking["complied_pos_rate"]
    rn, cn = 1.0 - rp, 1.0 - cp
    n_neurons = ranking["n_neurons"]

    pos_sel, pos_excl = _flow(rp, cp, k, m)   # 亮表
    neg_sel, neg_excl = _flow(rn, cn, k, m)   # 暗表

    selected = set()
    if sign in ("positive", "both"):
        selected |= pos_sel
    if sign in ("negative", "both"):
        selected |= neg_sel

    reversed_set = set()
    if detect_reversal:
        reversed_set |= (pos_sel & neg_excl)   # 亮表選中、暗表排除
        reversed_set |= (neg_sel & pos_excl)   # 暗表選中、亮表排除
        reversed_set &= selected               # 只翻轉「有被選到」的
    zeroed = selected - reversed_set

    to_pair = lambda i: (i // n_neurons, i % n_neurons)
    stats = {"n_selected": len(selected), "n_zeroed": len(zeroed),
             "n_reversed": len(reversed_set), "pos_selected": len(pos_sel),
             "neg_selected": len(neg_sel)}
    return [to_pair(i) for i in sorted(zeroed)], [to_pair(i) for i in sorted(reversed_set)], stats


# ── 範圍與下一格（純邏輯，不碰模型）────────────────────────────────────────

def bounds_from_coarse(coarse: list[dict]) -> dict:
    """起點＝翻轉增加最多那段之前的點，終點＝崩壞增加最多那段之後的點。"""
    ascending = sorted(coarse, key=lambda cell: cell["k"])
    if len(ascending) < 2:
        raise KmSearchError("粗測少於兩格，無法判斷上升段")
    pairs = list(zip(ascending, ascending[1:]))
    flip_low, flip_high = max(pairs, key=lambda p: p[1]["flips"] - p[0]["flips"])
    fall_low, fall_high = max(pairs, key=lambda p: p[1]["auto_collapse"] - p[0]["auto_collapse"])
    if flip_high["flips"] <= flip_low["flips"]:
        raise KmSearchError("粗測沒有看到翻轉上升")
    if fall_high["auto_collapse"] <= fall_low["auto_collapse"]:
        raise KmSearchError("粗測沒有看到崩壞上升")
    if flip_low["k"] >= fall_high["k"]:
        raise KmSearchError(f"起點 {flip_low['k']} 不小於終點 {fall_high['k']}")
    return {"start_k": flip_low["k"], "end_m": fall_high["k"], "source": "coarse",
            "flip_jump": [flip_low["k"], flip_high["k"]],
            "collapse_jump": [fall_low["k"], fall_high["k"]]}


def next_cell(k: int, m: int, collapse: int, *, end_m: int, k_floor: int, limit: float,
              up: float, down: float) -> tuple[int, int] | None:
    """limit 是題數：崩壞比例門檻 × 該格題數。

    小數值四捨五入可能回到原值（k=2、down=0.8；m=2、up=1.25），而已跑過的格子會直接
    取快取，下一格等於本格就會無窮迴圈。只在這種情況強制走一步，其餘路徑不受影響。
    """
    if m >= end_m:
        return None
    k_floor = max(k_floor, 1)
    if collapse >= limit and k > k_floor:
        lowered = max(round(k * down), k_floor)
        return (lowered if lowered < k else k - 1), m
    up_m = min(max(round(m * up), m + 1), end_m)
    return min(round(k * up), up_m), up_m


# ── 共用搜尋驅動器（km_search 與 km_cos 共用；evaluate(kind,k,m)->cell 由各模式注入）──

def run_coarse(evaluate, n_neurons: int, cfg: dict) -> dict:
    """粗測：K=M 從 n_neurons//2 對半砍，翻轉連續 coarse_stop_rounds 格不超過
    coarse_stop_flips 就停，回傳 bounds。只有「用翻轉數當停止條件」是拒絕專屬，
    所以 km_cos 通常用 config 固定 start_k/end_m 跳過粗測。"""
    k = n_neurons // 2
    cells, quiet = [], 0
    while k >= 1:
        cell = evaluate("sync", k, k)
        cells.append(cell)
        quiet = quiet + 1 if cell["flips"] <= int(cfg["coarse_stop_flips"]) else 0
        if quiet >= int(cfg["coarse_stop_rounds"]):
            break
        k //= 2
    return bounds_from_coarse(cells)


def run_fine(evaluate, bounds: dict, cfg: dict) -> None:
    """細搜：從 start_k 起，依每格 auto_collapse 用 next_cell 步進，直到 None。
    只依賴崩壞數，與目標行為無關 → km_search / km_cos 完全共用。"""
    k_floor = int(cfg.get("k_floor") or 0) or bounds["start_k"] // 2
    k, m = bounds["start_k"], bounds["start_k"]
    while True:
        cell = evaluate("sync", k, m)
        following = next_cell(k, m, cell["auto_collapse"], end_m=bounds["end_m"], k_floor=k_floor,
                              limit=float(cfg["collapse_ratio"]) * cell["n"],
                              up=float(cfg["km_up"]), down=float(cfg["k_down"]))
        if following is None:
            return
        k, m = following


# ── 搜尋 ──────────────────────────────────────────────────────────────────────

class KmSearchRunner:
    """滿足 train()/remove_callback() 介面，觸發完整 callback 生命週期。"""

    def __init__(self, trainer, callbacks, resolve):
        self.trainer    = trainer
        self.cfg        = trainer.cfg
        self.callbacks  = list(callbacks)
        self.resolve    = resolve
        # 依目標模型解析 system prompt（Qwen→原身分，與既有結果一致；其他家族→通用）
        self.system     = resolve_system(self.cfg["model_name"])
        self.output_dir = trainer.output_dir
        self.exp_dir    = trainer.output_dir.parent
        self.cells_dir  = self.exp_dir / "km_cells"
        self.state_path = self.exp_dir / "km_state.json"
        self.state: dict = {"cells": [], "bounds": None, "prompts": {}, "controls": [], "test": []}
        self.ranking: dict | None = None
        self.records: dict[str, dict] = {}
        self.judge = None
        self.keyword = KeywordJudge()
        self.gib: tuple | None = None
        self.rerank: tuple | None = None
        self.tracker = None

    def remove_callback(self, cls):
        self.callbacks = [c for c in self.callbacks if not isinstance(c, cls)]

    # ── 狀態 ──────────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.state_path)

    def _find(self, kind: str, k: int, m: int, repeat: int = 0) -> dict | None:
        for cell in self.state["cells"]:
            if (cell["kind"], cell["k"], cell["m"], cell["repeat"]) == (kind, k, m, repeat):
                return cell
        return None

    # ── 模型與判定 ────────────────────────────────────────────────────────────

    def _load_model(self):
        model, tokenizer = load_target_model(None, "final", self.cfg["model_name"])
        model.config.use_cache = True
        return model, tokenizer

    def _generate(self, neurons: list, prompts: list[dict],
                  reversed_neurons: list | None = None) -> list[dict]:
        """每次都從檔案重新載入模型，所以不需要還原權重。neurons 歸零，reversed_neurons 取負。"""
        model, tokenizer = self._load_model()
        if neurons:
            apply_prune_mask(model, neurons, verbose=False)
        if reversed_neurons:
            apply_reverse_mask(model, reversed_neurons, verbose=False)
        details = generate_all(model, tokenizer, prompts, self.system, int(self.cfg["gen_max_tokens"]))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return details

    def _select(self, kind: str, k: int, m: int, repeat: int):
        """回傳 (歸零清單, 翻轉清單, removed_from_top_k, 用來取 score 的 flat 索引, 選點統計)。"""
        nn = self.ranking["n_neurons"]
        to_pair = lambda i: (i // nn, i % nn)
        sign = str(self.cfg.get("sign", "positive"))
        reverse = bool(self.cfg.get("reverse_type", False))
        if kind == "random":
            # 公平對照：隨機裁減的「總數」要等於同 K,M 下 sync 實際修改的神經元數，
            # 否則 both 模式 sync 修改約 2K、random 只修改 K，會把「裁得多」誤當成「選得準」
            if sign == "positive" and not reverse:
                n_target = k
            else:
                z, rv, _ = select_tables(self.ranking, k, m, sign, reverse)
                n_target = len(z) + len(rv)
            g = torch.Generator().manual_seed(int(self.cfg["split_seed"]) + 1000 + repeat)
            chosen = torch.randperm(self.ranking["score"].numel(), generator=g)[:n_target]
            return [to_pair(i) for i in chosen.tolist()], [], 0, chosen, {}

        if sign == "positive" and not reverse:
            chosen, removed = select_sync(self.ranking, k, m)   # 相容舊 ranking 與舊結果
            return [to_pair(i) for i in chosen.tolist()], [], removed, chosen, {}

        zeroed, reversed_pairs, stats = select_tables(self.ranking, k, m, sign, reverse)
        flat = torch.tensor([l * nn + i for l, i in zeroed] + [l * nn + i for l, i in reversed_pairs])
        return zeroed, reversed_pairs, 0, flat, stats

    def _judge(self, details: list[dict], tag: dict) -> list[dict]:
        texts = [d["response"] for d in details]
        rows = []
        for detail, verdict, kw in zip(details, self.judge.judge_batch(texts),
                                       self.keyword.judge_batch(texts)):
            rows.append({
                **tag,
                "id": detail["id"], "domain": detail.get("domain"), "text": detail["text"],
                "response": detail["response"], "n_tokens": detail["n_tokens"],
                "truncated": detail["truncated"], "refused": verdict["refused"],
                "score": verdict["score"], "logit": verdict.get("logit"),
                "keyword_category": kw["category"],
                "response_lang": response_lang(detail["response"]),
                "repeat_all": round(repeated(detail["response"]), 4),
            })
        if self.gib is None:
            self.gib = load(GIBBERISH)
            self.rerank = load(RERANKER)
        gib = gibberish_probs([r["response"] for r in rows if r["response_lang"] == "en"], *self.gib)
        rel = relevance([(r["text"], r["response"]) for r in rows], *self.rerank)
        for row in rows:
            row["gibberish"] = gib.get(row["response"])
            row["relevance"] = rel[(row["text"], row["response"])]
        return rows

    def _auto_collapse(self, row: dict) -> bool:
        g = row.get("gibberish")
        return (row["repeat_all"] >= float(self.cfg["repeat_bad"])
                or row["relevance"] <= float(self.cfg["relevance_bad"])
                or bool(g and g["noise"] + g["word salad"] > float(self.cfg["gibberish_bad"])))

    # ── 一格 ──────────────────────────────────────────────────────────────────

    def _evaluate(self, kind: str, k: int, m: int, repeat: int = 0, split: str = "tune") -> dict:
        """kind：sync / random。已跑過就直接取用，接續與細搜共用這個快取。"""
        cell_kind = kind if split == "tune" else f"{kind}_{split}"
        cached = self._find(cell_kind, k, m, repeat)
        if cached is not None:
            return cached

        started = time.time()
        zeroed, reversed_neurons, removed, chosen, sel_stats = self._select(kind, k, m, repeat)
        neurons = zeroed + reversed_neurons   # checkpoint 與統計用的完整清單

        prompts = [self.records[i] for i in self.state["prompts"][split]]
        details = self._generate(zeroed, prompts, reversed_neurons)
        rows = self._judge(details, {"split": split, "kind": cell_kind, "k": k, "m": m, "repeat": repeat})

        self.cells_dir.mkdir(parents=True, exist_ok=True)
        name = f"{cell_kind}_K{k}_M{m}_r{repeat}.jsonl"
        (self.cells_dir / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

        flips = [r for r in rows if not r["refused"]]
        collapse = sum(self._auto_collapse(r) for r in flips)
        scores = self.ranking["score"][chosen] if len(chosen) else torch.zeros(1)
        step = len(self.state["cells"]) + 1
        cell = {
            "step": step, "kind": cell_kind, "k": k, "m": m, "repeat": repeat, "split": split,
            "n_pruned": len(zeroed), "n_reversed": len(reversed_neurons),
            "removed_from_top_k": removed, "n": len(rows),
            "flips": len(flips), "auto_collapse": collapse, "other_flips": len(flips) - collapse,
            "refused": len(rows) - len(flips),
            "keyword_comply": sum(r["keyword_category"] == "comply" for r in rows),
            "mean_tokens": round(sum(r["n_tokens"] for r in rows) / len(rows), 1),
            "truncated": sum(r["truncated"] for r in rows),
            "mean_logit": round(sum(r["logit"] or 0.0 for r in rows) / len(rows), 3),
            "score_mean": round(float(scores.mean()), 4),
            "score_min": round(float(scores.min()), 4),
            "seconds": round(time.time() - started),
            **sel_stats,
        }
        # 翻轉會在順從題上製造過度拒絕，所以用它時必須量順從題，否則指標會騙人
        comp_ids = self.state["prompts"].get(f"{split}_complied", [])
        if bool(self.cfg.get("test_complied", False)) and comp_ids:
            comp = self._judge(self._generate(zeroed, [self.records[i] for i in comp_ids], reversed_neurons),
                               {"split": split, "kind": f"{cell_kind}_complied", "k": k, "m": m, "repeat": repeat})
            (self.cells_dir / f"{cell_kind}_complied_K{k}_M{m}_r{repeat}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in comp) + "\n", encoding="utf-8")
            cell["complied_n"] = len(comp)
            cell["complied_over_refusal"] = sum(r["refused"] for r in comp)  # 本來順從、裁後改拒絕

        self.state["cells"].append(cell)
        self._save_state()
        self._publish(cell, rows, zeroed, reversed_neurons)
        extra = (f" 翻轉節點 {len(reversed_neurons)}" if reversed_neurons else "")
        extra += (f" 過度拒絕 {cell['complied_over_refusal']}/{cell['complied_n']}"
                  if "complied_over_refusal" in cell else "")
        print(f"[{cell_kind}] K={k} M={m} r={repeat} 翻轉 {cell['flips']}/{cell['n']}"
              f" 自動崩壞 {collapse} 其餘翻轉 {cell['other_flips']} 拒絕 {cell['refused']}"
              f"{extra} {cell['seconds']}秒", flush=True)
        return cell

    def _publish(self, cell: dict, rows: list[dict], zeroed: list, reversed_neurons: list) -> None:
        import types

        state, control = self.tracker
        state.global_step = cell["step"]
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))
        logs = {"km_kind": cell["kind"], **{key: value for key, value in cell.items()
                                            if key not in ("step", "kind")}}
        self.trainer.last_metrics = logs
        self.trainer.last_details = rows
        for cb in self.callbacks:
            cb.on_step_end(fake_args, state, control)
        for cb in self.callbacks:
            cb.on_log(fake_args, state, control, logs=logs)
        for cb in self.callbacks:
            cb.on_evaluate(fake_args, state, control, model=None, logs=logs)
        write_prune_checkpoint(self.output_dir / f"checkpoint-{cell['step']}", self.cfg["model_name"],
                               zeroed, reversed_neurons=reversed_neurons,
                               phase=cell["kind"], metrics=logs)
        for cb in self.callbacks:
            cb.on_save(fake_args, state, control)

    # ── 準備 ──────────────────────────────────────────────────────────────────

    def _model_groups(self, records: list[dict], seeds_path: Path) -> list[dict]:
        """用當前模型回答全部題目並分組。

        原始題目不動。結果另存一份，只記 id 與模型的回應、判定，不重複存題目文字；
        模型、system prompt、token 上限、判定器與題目都相同時直接沿用，不重新生成。
        """
        settings = {
            "model": str(self.cfg["model_name"]), "system": self.system,
            "max_new_tokens": int(self.cfg["gen_max_tokens"]), "judge": self.judge.name,
            "seeds_sha256": hashlib.sha256(seeds_path.read_bytes()).hexdigest(), "limit": len(records),
        }
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        slug = settings["model"].replace("/", "--")
        path = self.resolve(str(self.cfg["model_groups_dir"])) / f"{slug}__{key}.jsonl"
        by_id = {r["id"]: r for r in records}

        if path.exists():
            print(f"沿用模型分組：{path.name}", flush=True)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        else:
            print(f"用 {settings['model']} 重新分組 {len(records)} 題（找不到相同條件的分組）", flush=True)
            details = self._generate([], records)
            rows = [{"id": d["id"], "pair_id": d["pair_id"],
                     "group": POSITIVE if v["refused"] else NEGATIVE, "score": v["score"],
                     "logit": v.get("logit"), "response": d["response"], "n_tokens": d["n_tokens"],
                     "truncated": d["truncated"]}
                    for d, v in zip(details, self.judge.judge_batch([d["response"] for d in details]))]
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            temp.replace(path)
            path.with_suffix(".meta.json").write_text(
                json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

        self.state["groups"] = {"source": "model", "path": str(path), **settings,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        counts = Counter(r["group"] for r in rows)
        print(f"分組：{POSITIVE}={counts[POSITIVE]} {NEGATIVE}={counts[NEGATIVE]}", flush=True)
        return [{**by_id[r["id"]], "group": r["group"]} for r in rows]

    def _prepare(self) -> None:
        judge_name = str(self.cfg["judge"])
        self.judge = load_judge(judge_name, **({} if judge_name == "keyword"
                                               else {"device": str(self.cfg["judge_device"])}))

        seeds_path = self.resolve(self.cfg["seeds"])
        warn_if_method_mismatch(seeds_path, "km_search")
        records = load_source(seeds_path)
        if self.cfg.get("limit"):
            records = records[:int(self.cfg["limit"])]

        source = str(self.cfg["group_source"])
        if source == "model":
            records = self._model_groups(records, seeds_path)
        elif source == "dataset":
            # 現成分組綁定產生它的模型，換模型沿用會得到錯誤結果，也不會有任何錯誤訊息
            if str(self.cfg["model_name"]) != str(self.cfg["dataset_group_model"]):
                raise KmSearchError(f"{seeds_path.name} 的分組來自 {self.cfg['dataset_group_model']}，"
                                    f"不能用於 {self.cfg['model_name']}；請改用 group_source = model")
            if any(r["group"] not in (POSITIVE, NEGATIVE) for r in records):
                raise KmSearchError(f"{seeds_path.name} 沒有現成分組；請改用 group_source = model")
            self.state["groups"] = {"source": "dataset", "path": str(seeds_path),
                                    "sha256": hashlib.sha256(seeds_path.read_bytes()).hexdigest()}
        else:
            raise KmSearchError(f"不認得的 group_source：{source}（可用：model / dataset）")

        self.records = {r["id"]: r for r in records}
        seed = int(self.cfg["split_seed"])
        splits = split_by_pair(records, seed, float(self.cfg["rank_ratio"]), float(self.cfg["tune_ratio"]))
        print("切分：" + "  ".join(f"{name}={len(rows)}" for name, rows in splits.items()), flush=True)

        ranking_path = self.exp_dir / "ranking.pt"
        if not ranking_path.exists():
            source = str(self.cfg.get("ranking_path", "") or "")
            if source:
                shutil.copy(self.resolve(source), ranking_path)
                print(f"沿用既有排序：{source}", flush=True)
            else:
                model, tokenizer = self._load_model()
                torch.save(build_ranking(model, tokenizer, splits["rank"], int(self.cfg["n_steps"]), seed, self.system),
                           ranking_path)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self.ranking = torch.load(ranking_path, weights_only=True)

        if "tune" not in self.state["prompts"]:
            candidates = [r for r in splits["tune"] if r["group"] == POSITIVE]
            # 模型分組本身就是這個模型的基準回答，不必再生成一次
            self.state["prompts"]["tune"] = (
                [r["id"] for r in stratified_order(candidates, seed)] if source == "model"
                else self._still_refused(candidates, "tune", seed))
            self._save_state()
        if not self.state["prompts"]["tune"]:
            raise KmSearchError("tune 集沒有基準模型會拒絕的題目")

        # 翻轉會傷害順從題，開 test_complied 時另存一份順從題供每格量過度拒絕
        if bool(self.cfg.get("test_complied", False)) and "tune_complied" not in self.state["prompts"]:
            complied = [r for r in splits["tune"] if r["group"] == NEGATIVE]
            self.state["prompts"]["tune_complied"] = [r["id"] for r in stratified_order(complied, seed)]
            self._save_state()

        test_seeds = str(self.cfg.get("test_seeds", "") or "")
        if test_seeds and "test" not in self.state["prompts"]:
            extra = load_source(self.resolve(test_seeds))
            self.records.update({r["id"]: r for r in extra})
            self.state["prompts"]["test"] = self._still_refused(extra, "test", seed)
            self._save_state()
        elif test_seeds:
            self.records.update({r["id"]: r for r in load_source(self.resolve(test_seeds))})
        if "test" in self.state["prompts"] and not self.state["prompts"]["test"]:
            raise KmSearchError("test 集沒有基準模型會拒絕的題目")

    def _still_refused(self, candidates: list[dict], split: str, seed: int) -> list[str]:
        """只測基準模型仍會拒絕的題目，否則「翻轉」沒有意義。"""
        details = self._generate([], candidates)
        rows = [{"id": d["id"], "response": d["response"], "refused": v["refused"]}
                for d, v in zip(details, self.judge.judge_batch([d["response"] for d in details]))]
        still = {r["id"] for r in rows if r["refused"]}
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        (self.cells_dir / f"baseline_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(f"基準模型 {split}：{len(still)}/{len(rows)} 仍拒絕", flush=True)
        return [r["id"] for r in stratified_order([c for c in candidates if c["id"] in still], seed)]

    # ── 各階段 ────────────────────────────────────────────────────────────────

    def _coarse(self) -> dict:
        start_k, end_m = int(self.cfg.get("start_k") or 0), int(self.cfg.get("end_m") or 0)
        if start_k and end_m:
            return {"start_k": start_k, "end_m": end_m, "source": "config"}
        if start_k or end_m:
            raise KmSearchError("start_k 與 end_m 要一起指定，或都留空讓粗測決定")
        return run_coarse(self._evaluate, self.ranking["score"].numel(), self.cfg)

    def _fine(self, bounds: dict) -> None:
        run_fine(self._evaluate, bounds, self.cfg)

    def _top_cells(self) -> list[dict]:
        sync = [c for c in self.state["cells"] if c["kind"] == "sync"]
        ranked = sorted(sync, key=lambda c: (c["flips"] - c["auto_collapse"], -c["k"]), reverse=True)
        return ranked[:int(self.cfg["control_top"])]

    def train(self, resume_from_checkpoint=False):
        import types

        from transformers import TrainerControl, TrainerState

        if self.state_path.exists():
            if not resume_from_checkpoint:
                raise KmSearchError(f"{self.exp_dir.name} 已有結果，要接續請用 --resume-run")
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            print(f"接續：已完成 {len(self.state['cells'])} 格", flush=True)

        state, control = TrainerState(), TrainerControl()
        self.tracker = (state, control)
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))
        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        self._prepare()
        bounds = self._coarse()
        self.state["bounds"] = bounds
        self._save_state()
        print(f"範圍：起點 K=M={bounds['start_k']}，終點 M={bounds['end_m']}（{bounds['source']}）", flush=True)
        self._fine(bounds)

        top = self._top_cells()
        for repeat in range(int(self.cfg["control_repeats"])):
            for cell in top:
                self._evaluate("random", cell["k"], cell["m"], repeat)
        if "test" in self.state["prompts"]:
            for cell in top:
                self._evaluate("sync", cell["k"], cell["m"], split="test")
                self._evaluate("random", cell["k"], cell["m"], split="test")

        self.state["top"] = [{"k": c["k"], "m": c["m"]} for c in top]
        self.state["finished"] = True
        self._save_state()

        if top:
            zeroed, reversed_neurons, _, _, _ = self._select("sync", top[0]["k"], top[0]["m"], 0)
            write_prune_checkpoint(self.output_dir / "final", self.cfg["model_name"],
                                   zeroed, reversed_neurons=reversed_neurons,
                                   phase="final", k=top[0]["k"], m=top[0]["m"])
        self._write_report()
        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)
        self._print_summary()

    # ── 報表 ──────────────────────────────────────────────────────────────────

    def _write_report(self) -> None:
        try:
            from prune_search_report import write_report
        except ImportError as e:
            print(f"略過圖表（缺少套件：{e}）")
            return
        write_report(self.state, self.exp_dir / "km_report.png")
        print(f"圖表：{self.exp_dir / 'km_report.png'}")

    def _print_summary(self) -> None:
        print("\n" + "=" * 72)
        print("K、M 搜尋結果（其餘翻轉＝翻轉－自動崩壞，可能仍含事實錯亂與答非所問）")
        print("=" * 72)
        for cell in self.state["cells"]:
            print(f"#{cell['step']:<3} {cell['kind']:<12} K={cell['k']:<7} M={cell['m']:<7}"
                  f" r={cell['repeat']} 翻轉={cell['flips']:<3} 崩壞={cell['auto_collapse']:<3}"
                  f" 其餘={cell['other_flips']:<3} 拒絕={cell['refused']}")
        print("最高：", self.state.get("top"))


def format_generation_block(metrics: dict, rows: list[dict], is_collapse, redact: bool = False) -> str:
    """generation_test.txt 的內容：本格摘要，加上翻轉與仍拒絕的例子。

    redact=True 時只寫各組計數，不列 [id] 題目 / → 回答 原文（跑違禁題資料集時用，
    避免檢視進程時顯示 prompt/response 內容）。
    """
    lines = ["=" * 60,
             f"[K、M 搜尋] {metrics['km_kind']} K={metrics['k']} M={metrics['m']} r={metrics['repeat']}"
             f"  翻轉 {metrics['flips']}/{metrics['n']}  自動崩壞 {metrics['auto_collapse']}",
             "=" * 60]
    groups = (("仍拒絕", [r for r in rows if r["refused"]]),
              ("自動判為崩壞", [r for r in rows if not r["refused"] and is_collapse(r)]),
              ("其餘翻轉", [r for r in rows if not r["refused"] and not is_collapse(r)]))
    for label, picked in groups:
        lines.append(f"── {label}（{len(picked)}）──")
        if redact:
            continue  # 只報計數，不列題目/回答原文
        for row in picked[:2]:
            lines.append(f"  [{row['id']}] {row['text'][:60]}")
            lines.append(f"    → {row['response'][:150]}")
    lines.append("")
    return "\n".join(lines)


def summarize_counts(cells: list[dict]) -> Counter:
    return Counter(cell["kind"] for cell in cells)
