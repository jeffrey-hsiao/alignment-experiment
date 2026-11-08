"""
MLOps/prune_km_cos.py

km_cos（cos = cosplay）：通用「行為編輯」模式，與 km_search 平行、共用同一套引擎，
但針對「讓模型 cosplay 一個人格」這類**誘發型**行為:

  1. 分組用「對比系統提示詞」而非模型行為判定 —— 同一批題分別配 target_system
     （要誘發的人格，如貓娘）與 base_system（一般身分），捕激活成兩組。**不需要
     對每題做真實結果判定**，所以沒有 km_search 的前置分組階段。
  2. 搜尋步進只靠崩壞偵測（共用 run_fine）。
  3. best-cell 用**對比相關性**取代翻轉率：對每個輸出算 rel(target_system,out) −
     rel(base_system,out)（reranker），最高者為最佳格。

共用（import 自 prune_search / prune_utils / neuron_probe）：select_tables、run_fine、
run_coarse、GroupStats、GenerationProbe、rate_order、apply_reverse_mask、
write_prune_checkpoint、load_target_model、崩壞偵測 helper。km_search 的拒絕路徑完全不碰。

狀態：結構骨架，model 相關內部待 GPU 空出後以真實 run 驗證。
"""
import gc
import json
import time
from pathlib import Path

import torch

from neuron_probe import GenerationProbe, GroupStats, load_seeds, load_target_model  # noqa: E402
from chat_prompt import resolve_system  # noqa: E402
from behavior_regroup import generate_all  # noqa: E402
from prune_utils import apply_reverse_mask, apply_prune_mask, write_prune_checkpoint  # noqa: E402
from prune_search import (  # noqa: E402  ★共用引擎
    POSITIVE, NEGATIVE, rate_order, select_tables, run_fine, run_coarse,
    load_source, split_by_pair, stratified_order, is_collapsed,
)


class KmCosError(Exception):
    pass


# ── 對比排名（平行 build_ranking，但用兩個 system 而非 group 欄）──────────────────

def build_ranking_contrastive(model, tokenizer, records: list[dict], n_steps: int, seed: int,
                              target_system: str, base_system: str) -> dict:
    """同一批題分別配 target/base system 跑一次，激活進 POSITIVE/NEGATIVE 兩組，
    組出與 build_ranking 相同格式的 ranking（供 select_tables 使用）。"""
    pooled = {POSITIVE: GroupStats(), NEGATIVE: GroupStats()}
    for system, group in ((target_system, POSITIVE), (base_system, NEGATIVE)):
        with GenerationProbe(model, tokenizer, system=system) as probe:
            for i, record in enumerate(records, 1):
                acts, _ = probe.run(record["text"], n_steps)
                for position in range(acts.shape[0]):
                    pooled[group].update(acts[position], 0.0)
                if i % 20 == 0 or i == len(records):
                    print(f"  對比排名探測 [{group}] {i}/{len(records)} 筆", flush=True)
    pos, neg = pooled[POSITIVE], pooled[NEGATIVE]
    score = (pos.mean - neg.mean) / ((pos.std ** 2 + neg.std ** 2) / 2).sqrt().clamp_min(1e-6)
    return {
        "n_neurons": score.shape[1],
        "positive":  rate_order(pos),
        "negative":  rate_order(neg),
        "random":    torch.randperm(score.numel(), generator=torch.Generator().manual_seed(seed)),
        "score":     score.flatten(),
        "refused_pos_rate":  pos.active_rate.flatten(),   # 沿用欄名以相容 select_tables
        "complied_pos_rate": neg.active_rate.flatten(),
        "n_tokens":  {POSITIVE: pos.n, NEGATIVE: neg.n},
    }


class KmCosRunner:
    """與 KmSearchRunner 同介面（train/remove_callback），但走 km_cos 流程。"""

    def __init__(self, trainer, callbacks, resolve):
        self.trainer    = trainer
        self.cfg        = trainer.cfg
        self.callbacks  = list(callbacks)
        self.resolve    = resolve
        self.output_dir = trainer.output_dir
        self.exp_dir    = trainer.output_dir.parent
        self.cells_dir  = self.exp_dir / "km_cells"
        self.state_path = self.exp_dir / "km_state.json"
        self.state: dict = {"cells": [], "bounds": None, "prompts": {}, "mode": "km_cos"}
        self.ranking: dict | None = None
        self.records: dict[str, dict] = {}
        self.rerank = None
        self.target_system = ""
        self.base_system = ""

    def remove_callback(self, cls):
        self.callbacks = [c for c in self.callbacks if not isinstance(c, cls)]

    # ── 工具 ─────────────────────────────────────────────────────────────────
    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _find(self, kind, k, m, repeat=0):
        for c in self.state["cells"]:
            if (c["kind"], c["k"], c["m"], c["repeat"]) == (kind, k, m, repeat):
                return c
        return None

    def _load_model(self):
        model, tokenizer = load_target_model(None, "final", self.cfg["model_name"])
        model.config.use_cache = True
        return model, tokenizer

    def _resolve_systems(self):
        """target_system：inline 或 target_system_file；base_system：inline 或 resolve_system。"""
        tf = str(self.cfg.get("target_system_file", "") or "")
        if tf:
            self.target_system = self.resolve(tf).read_text(encoding="utf-8").strip()
        else:
            self.target_system = str(self.cfg.get("target_system", "") or "")
        if not self.target_system:
            raise KmCosError("km_cos 需要 target_system 或 target_system_file")
        bf = str(self.cfg.get("base_system_file", "") or "")
        if bf:
            self.base_system = self.resolve(bf).read_text(encoding="utf-8").strip()
        else:
            self.base_system = str(self.cfg.get("base_system", "") or "") or resolve_system(self.cfg["model_name"])

    # ── 崩壞 + 對比相關性（best-cell 用）──────────────────────────────────────
    def _load_scorers(self):
        from quality_pilot import GIBBERISH, RERANKER, load  # noqa: E402  延後載入
        self._gib = load(GIBBERISH)
        self.rerank = load(RERANKER)

    def _collapse(self, rows: list[dict]) -> list[bool]:
        """沿用 km_search 的崩壞定義：全文重複率 / 題目相關 / 英文亂碼。"""
        from quality_features import repeated  # noqa: E402
        from quality_pilot import gibberish_probs, relevance, response_lang  # noqa: E402
        for r in rows:
            r["response_lang"] = response_lang(r["response"])
            r["repeat_all"] = round(repeated(r["response"]), 4)
        gib = gibberish_probs([r["response"] for r in rows if r["response_lang"] == "en"], *self._gib)
        rel = relevance([(r["text"], r["response"]) for r in rows], *self.rerank)
        out = []
        for r in rows:
            g = gib.get(r["response"])
            out.append(is_collapsed(r["repeat_all"], rel[(r["text"], r["response"])], g, self.cfg))
        return out

    def _cos_score(self, rows: list[dict]) -> float:
        """對比相關性：mean( rel(target_system, out) − rel(base_system, out) )。越高越像人格。"""
        from quality_pilot import relevance  # noqa: E402
        resp = [r["response"] for r in rows]
        rel_t = relevance([(self.target_system, x) for x in resp], *self.rerank)
        rel_b = relevance([(self.base_system, x) for x in resp], *self.rerank)
        diffs = [rel_t[(self.target_system, x)] - rel_b[(self.base_system, x)] for x in resp]
        return round(sum(diffs) / len(diffs), 4) if diffs else 0.0

    # ── 選點 + 生成 ───────────────────────────────────────────────────────────
    def _select(self, k: int, m: int, repeat: int = 0, random_control: bool = False):
        """km_cos = 誘發：把選到的 cosplay 神經元『取負』（reverse）來放大人格方向。
        sign/reverse_type 沿用 km_search 的雙表語意。"""
        nn = self.ranking["n_neurons"]
        to_pair = lambda i: (i // nn, i % nn)
        sign = str(self.cfg.get("sign", "both"))
        zeroed, reversed_pairs, stats = select_tables(self.ranking, k, m, sign, True)
        if random_control:
            n_target = len(zeroed) + len(reversed_pairs)
            g = torch.Generator().manual_seed(int(self.cfg["split_seed"]) + 1000 + repeat)
            chosen = torch.randperm(self.ranking["score"].numel(), generator=g)[:n_target]
            # 對照組：同數量隨機節點，一律取負（與 induce 同手法）
            return [], [to_pair(i) for i in chosen.tolist()], stats
        return zeroed, reversed_pairs, stats

    def _generate_nosys(self, zeroed, reversed_neurons, prompts):
        """關鍵：誘發後要**不帶 target_system** 測，看行為有沒有 transfer 到權重。"""
        model, tokenizer = self._load_model()
        if zeroed:
            apply_prune_mask(model, zeroed, verbose=False)
        if reversed_neurons:
            apply_reverse_mask(model, reversed_neurons, verbose=False)
        details = generate_all(model, tokenizer, prompts, self.base_system, int(self.cfg["gen_max_tokens"]))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return details

    # ── 單格評估（注入給 run_fine 的 evaluate）───────────────────────────────
    def _evaluate(self, kind: str, k: int, m: int, repeat: int = 0) -> dict:
        cached = self._find(kind, k, m, repeat)
        if cached is not None:
            return cached
        started = time.time()
        zeroed, reversed_neurons, stats = self._select(k, m, repeat, random_control=(kind == "random"))
        prompts = [self.records[i] for i in self.state["prompts"]["tune"]]
        rows = self._generate_nosys(zeroed, reversed_neurons, prompts)
        collapse_flags = self._collapse(rows)
        collapse = sum(collapse_flags)
        n = len(rows)
        # cos 分數只算非崩壞的輸出（崩壞的相關性沒意義）
        good = [r for r, bad in zip(rows, collapse_flags) if not bad]
        cos = self._cos_score(good) if good else 0.0
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        (self.cells_dir / f"{kind}_K{k}_M{m}_r{repeat}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        cell = {
            "step": len(self.state["cells"]) + 1, "kind": kind, "k": k, "m": m, "repeat": repeat,
            "n": n, "auto_collapse": collapse, "cos": cos,
            "n_reversed": len(reversed_neurons), "n_pruned": len(zeroed),
            # run_fine 只讀 auto_collapse/n；flips 這裡不適用，給個占位讓共用邏輯/報表不爆
            "flips": n - collapse, "other_flips": n - collapse,
            "seconds": round(time.time() - started), **stats,
        }
        self.state["cells"].append(cell)
        self._save_state()
        print(f"[{kind}] K={k} M={m} r={repeat} 崩壞 {collapse}/{n} cos={cos} "
              f"翻轉節點 {len(reversed_neurons)} {cell['seconds']}秒", flush=True)
        return cell

    # ── 流程 ─────────────────────────────────────────────────────────────────
    def _prepare(self) -> None:
        seed = int(self.cfg["split_seed"])
        self._resolve_systems()
        self._load_scorers()
        seeds_path = self.resolve(self.cfg["seeds"])
        records = [{**r, "pair_id": r.get("pair_id", r["id"])} for r in load_seeds([seeds_path], ())]
        self.records = {r["id"]: r for r in records}
        splits = split_by_pair(records, seed,
                               float(self.cfg["rank_ratio"]), float(self.cfg["tune_ratio"]))
        self.state["prompts"]["tune"] = [r["id"] for r in stratified_order(splits["tune"], seed)]
        self._save_state()
        # 對比排名：用 rank 集
        model, tokenizer = self._load_model()
        self.ranking = build_ranking_contrastive(
            model, tokenizer, splits["rank"], int(self.cfg["n_steps"]), seed,
            self.target_system, self.base_system)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.save(self.ranking, self.exp_dir / "ranking.pt")

    def _bounds(self) -> dict:
        start_k, end_m = int(self.cfg.get("start_k") or 0), int(self.cfg.get("end_m") or 0)
        if start_k and end_m:
            return {"start_k": start_k, "end_m": end_m, "source": "config"}
        # km_cos 沒有翻轉數可用來粗測停止，所以要求 config 指定；留空則用神經元總數派生的保守界
        n = self.ranking["score"].numel()
        return {"start_k": n // 16, "end_m": n // 2, "source": "default"}

    def train(self, resume_from_checkpoint=False):
        import types
        from transformers import TrainerControl, TrainerState

        if self.state_path.exists() and not resume_from_checkpoint:
            raise KmCosError(f"{self.exp_dir.name} 已有結果，要接續請用："
                             f"python run.py train km_cos --resume {self.exp_dir.name}")
        state, control = TrainerState(), TrainerControl()
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))
        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        self._prepare()
        bounds = self._bounds()
        self.state["bounds"] = bounds
        self._save_state()
        print(f"範圍：起點 K=M={bounds['start_k']}，終點 M={bounds['end_m']}（{bounds['source']}）", flush=True)
        run_fine(self._evaluate, bounds, self.cfg)   # ★共用細搜驅動器

        # best-cell：非崩壞格中 cos 最高
        sync = [c for c in self.state["cells"] if c["kind"] == "sync"]
        viable = [c for c in sync if c["auto_collapse"] < c["n"]] or sync
        best = max(viable, key=lambda c: c["cos"]) if viable else None
        top = [best] if best else []
        for repeat in range(int(self.cfg["control_repeats"])):
            for c in top:
                self._evaluate("random", c["k"], c["m"], repeat)
        self.state["top"] = [{"k": c["k"], "m": c["m"], "cos": c["cos"]} for c in top]
        self.state["finished"] = True
        self._save_state()

        if best:
            zeroed, reversed_neurons, _ = self._select(best["k"], best["m"], 0)
            write_prune_checkpoint(self.output_dir / "final", self.cfg["model_name"],
                                   zeroed, reversed_neurons=reversed_neurons,
                                   phase="final", k=best["k"], m=best["m"])
        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)
