"""
MLOps/prune_km_cos.py

km_cos（cos = cosplay）：通用「行為編輯」模式，與 km_search 平行、共用同一套引擎，
但針對「讓模型 cosplay 一個人格」這類**誘發型**行為:

  1. 分組用「對比系統提示詞」而非模型行為判定 —— 同一批題分別配 target_system
     （要誘發的人格，如貓娘）與 base_system（一般身分），捕激活成兩組。**不需要
     對每題做真實結果判定**，所以沒有 km_search 的前置分組階段。
  2. 搜尋改用 prune_v2 的 gen-2 單調 T/K 掃描（共用 run_sweep），**不再**走 prune 的
     run_fine 細搜。prune 的步進訊號是拒絕翻轉率，km_cos 沒有翻轉 → 改用**對比相關性
     cos（句向量差分）**當步進訊號：每個輸出算 cos(對 target 質心) − cos(對 base 質心)，
     質心由 rank 時兩組 system 條件的輸出算出。進格門檻 cos_low 不手設，由 _prepare 用同
     一組基準自動定標（frac 見 KmCosConfig 的 cos_low_frac，預設 0.5＝中點）。崩壞照記錄
     不驅動步進。
     （早期版本用 bge-reranker 的 rel 差當 cos，但 8 token 短輸出會塌成 0／反轉，已改為
      對稱、有界的 e5-small 句向量差分；reranker 僅留給崩壞偵測的 relevance。）
  3. best-cell 同樣用這個句向量差分 cos，最高者為最佳格。

共用（import 自 prune_search / prune_utils / neuron_probe）：select_tables、run_sweep、
GroupStats、GenerationProbe、rate_order、apply_reverse_mask、
write_prune_checkpoint、load_target_model、崩壞偵測 helper。km_search 的拒絕路徑完全不碰。

狀態：結構骨架，model 相關內部待 GPU 空出後以真實 run 驗證。
"""
import json
import time
from pathlib import Path

import torch

from neuron_probe import GenerationProbe, GroupStats, load_seeds, load_target_model  # noqa: E402
from behavior_regroup import generate_all  # noqa: E402
from prune_utils import apply_reverse_mask, apply_prune_mask, write_prune_checkpoint  # noqa: E402
from prune_search import (  # noqa: E402  ★共用引擎
    POSITIVE, NEGATIVE, rate_order, select_tables, run_sweep, ThresholdCriterion, CellRunner,
    free_cuda, load_source, split_by_pair, stratified_order, is_collapsed,
)

# 步進訊號的句向量編碼器。用 repo-id 經 HF 快取載入，與拒絕判定器 / reranker /
# 亂碼分類器（全部 snapshot_download / from_pretrained）同一套慣例——上游 HF 模型一律
# 用 repo-id，不放進 repo 樹、不特例用本地路徑。首次使用自動下載進 HF 快取並共用。
# （與 rag/ 用的是同一顆 e5-small，只是 rag 另外手動放了一份本地副本；此處不依賴它。）
# 釘死 commit（同 quality_pilot 的 reranker/亂碼分類器做法）：載入恆取這個不可變版本，
# 維護者更新 Hub 的 main 不會把我們悄悄帶到新版，確保可重現。
E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class KmCosError(Exception):
    pass


# ── 對比排名（平行 build_ranking，但用兩個 system 而非 group 欄）──────────────────

def build_ranking_contrastive(model, tokenizer, records: list[dict], n_steps: int, seed: int,
                              target_system: str, base_system: str) -> tuple[dict, dict]:
    """同一批題分別配 target/base system 跑一次，激活進 POSITIVE/NEGATIVE 兩組，
    組出與 build_ranking 相同格式的 ranking（供 select_tables 使用）。

    GenerationProbe.run 本來就回傳 (激活, 回應)；過去只取激活。這裡順手接住兩組
    system 條件下的生成輸出一併回傳（responses[POSITIVE]=target 條件、[NEGATIVE]=base
    條件），讓 _prepare 免費算出 cos 基準（天花板/地板）來自動定進格門檻。
    注意：這些輸出只有 n_steps 個 token（排序用的短生成），當尺度基準夠用，但不等長於
    每格評估的 gen_max_tokens。"""
    pooled = {POSITIVE: GroupStats(), NEGATIVE: GroupStats()}
    responses = {POSITIVE: [], NEGATIVE: []}   # 每項 (題目 text, 回應)，供 _cos_texts 併題目進 query
    for system, group in ((target_system, POSITIVE), (base_system, NEGATIVE)):
        with GenerationProbe(model, tokenizer, system=system) as probe:
            for i, record in enumerate(records, 1):
                acts, resp = probe.run(record["text"], n_steps)
                responses[group].append((record["text"], resp))
                for position in range(acts.shape[0]):
                    pooled[group].update(acts[position], 0.0)
                if i % 20 == 0 or i == len(records):
                    print(f"  對比排名探測 [{group}] {i}/{len(records)} 筆", flush=True)
    pos, neg = pooled[POSITIVE], pooled[NEGATIVE]
    score = (pos.mean - neg.mean) / ((pos.std ** 2 + neg.std ** 2) / 2).sqrt().clamp_min(1e-6)
    ranking = {
        "n_neurons": score.shape[1],
        "positive":  rate_order(pos),
        "negative":  rate_order(neg),
        "random":    torch.randperm(score.numel(), generator=torch.Generator().manual_seed(seed)),
        "score":     score.flatten(),
        "refused_pos_rate":  pos.active_rate.flatten(),   # 沿用欄名以相容 select_tables
        "complied_pos_rate": neg.active_rate.flatten(),
        "n_tokens":  {POSITIVE: pos.n, NEGATIVE: neg.n},
    }
    return ranking, responses


class KmCosRunner(CellRunner):
    """與 KmSearchRunner 同介面（train/remove_callback），但走 km_cos 流程。
    狀態存檔 / 格子查找 / 載入模型 / 移除 callback 繼承自 CellRunner。"""

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
        self._embedder = None
        self.cen_target = None   # rank 時 target 條件輸出的句向量質心（e5-small）
        self.cen_base = None     # rank 時 base 條件輸出的句向量質心
        self.target_system = ""
        self.base_system = ""
        self.cos_low = None   # 進格門檻；_prepare 由 rank 時的 cos 基準自動派生

    # ── 工具 ─────────────────────────────────────────────────────────────────
    def _resolve_systems(self):
        """target_system：inline 或 target_system_file；base_system：inline 或 base_system_file。

        base_system 留空＝**真的不帶 system**（空字串），對照組為「無提示詞」的原生輸出，
        不再 fallback 到原廠身分 / 通用助理。這樣 cos 對比乾淨：target 人格 vs 無 system。"""
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
            self.base_system = str(self.cfg.get("base_system", "") or "")

    # ── 崩壞 + 對比相關性（best-cell 用）──────────────────────────────────────
    def _load_scorers(self):
        from quality_pilot import GIBBERISH, RERANKER, load  # noqa: E402  延後載入
        self._gib = load(GIBBERISH)
        self.rerank = load(RERANKER)   # 仍供崩壞偵測的 relevance（步進訊號已不用它）
        # 步進訊號改用句向量差分：multilingual-e5-small（sentence-transformers 自動
        # pooling+normalize）。repo-id + 釘死 revision，與 reranker/判定器同一套慣例。
        from sentence_transformers import SentenceTransformer  # noqa: E402
        self._embedder = SentenceTransformer(E5_MODEL, revision=E5_REVISION)

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

    def _embed(self, texts: list):
        """e5-small 句向量。對稱相似度任務（輸出 vs 質心）→ 兩邊都用 'query: ' 前綴；
        normalize 後內積即 cosine。回傳 (n, d) 正規化張量（在模型裝置上）。"""
        return self._embedder.encode(["query: " + t for t in texts],
                                     normalize_embeddings=True, convert_to_tensor=True)

    def _centroid(self, texts: list):
        """一組輸出的質心方向（正規化平均向量）。"""
        import torch.nn.functional as F  # noqa: E402
        return F.normalize(self._embed(texts).mean(0), dim=-1)

    def _cos_texts(self, texts: list) -> float:
        """對比相關性 cos（向量版）：每個輸出算 cos(對 target 質心) − cos(對 base 質心)，
        取平均。越高＝輸出越像目標人格。

        取代原本 reranker 版本——後者在 8 token 短輸出上會塌成 0／反轉（cos_base > cos_target），
        因為拿題目當 query 的相關性灌爆了人格成分。改用對稱、有界（cosine∈[-1,1]）的句向量
        差分：質心在 _prepare 由 rank 兩組 system 條件的輸出算好（self.cen_target/base），每格
        評估與 rank 基準共用同一條定義。texts: list[str]（輸出 response）。"""
        if not texts or self.cen_target is None or self.cen_base is None:
            return 0.0
        V = self._embed(texts)                              # (n, d) 正規化
        s = (V @ self.cen_target) - (V @ self.cen_base)     # (n,) 向量差分
        return round(float(s.mean()), 4)

    def _cos_score(self, rows: list[dict]) -> float:
        return self._cos_texts([r["response"] for r in rows])

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
        free_cuda()
        return details

    # ── 單格評估（注入給 run_sweep 的 evaluate）──────────────────────────────
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
            # run_sweep 讀 cos 當步進訊號；flips 在 km_cos 不適用，給個占位讓報表/共用邏輯不爆
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
        # 對比排名：用 rank 集（順手接住兩組 system 條件下的生成輸出）
        model, tokenizer = self._load_model()
        self.ranking, responses = build_ranking_contrastive(
            model, tokenizer, splits["rank"], int(self.cfg["n_steps"]), seed,
            self.target_system, self.base_system)
        del model
        free_cuda()
        torch.save(self.ranking, self.exp_dir / "ranking.pt")

        # 句向量質心：rank 時 target/base 兩條件輸出各自的質心（e5-small）。每格評估都把
        # 輸出投到「target 質心 − base 質心」這條軸上當步進訊號（_cos_texts）。
        pos_out = [r for _, r in responses[POSITIVE]]
        neg_out = [r for _, r in responses[NEGATIVE]]
        self.cen_target = self._centroid(pos_out)
        self.cen_base = self._centroid(neg_out)
        cen_cos = round(float(self.cen_target @ self.cen_base), 4)   # 兩質心 cosine（越低越分開）

        # 進格門檻 cos_low：手動覆寫優先，否則用 rank 兩組輸出的向量差分基準自動定標。
        #   cos_base   = base 條件輸出的差分（偏負：像 base 質心）
        #   cos_target = target 條件輸出的差分（偏正：像 target 質心）
        #   自動：cos_low = cos_base + cos_low_frac × (cos_target − cos_base)（預設 0.5＝中點）
        # 每格 cos < cos_low → 效果不足，放大 T；否則衰減 K。
        cos_base = self._cos_texts(neg_out)
        cos_target = self._cos_texts(pos_out)
        frac = float(self.cfg.get("cos_low_frac", 0.5))
        manual = str(self.cfg.get("cos_low", "")).strip()
        manual_val = None
        if manual not in ("", "None"):
            try:
                manual_val = float(manual)        # 手動覆寫：直接當絕對門檻，略過自動定標
            except ValueError:
                print(f"  ⚠ cos_low={manual!r} 不是數字，忽略、改走自動定標。", flush=True)
        if manual_val is not None:
            self.cos_low = manual_val
            thr_src = "manual"
        else:
            self.cos_low = round(cos_base + frac * (cos_target - cos_base), 4)
            thr_src = f"auto frac={frac}"
        self.state["cos_baseline"] = {"cos_base": cos_base, "cos_target": cos_target,
                                      "centroid_cos": cen_cos, "cos_low_frac": frac,
                                      "cos_low": self.cos_low, "source": thr_src,
                                      "embedder": "multilingual-e5-small",
                                      "n_rank": len(splits["rank"]), "gen_tokens": int(self.cfg["n_steps"])}
        self._save_state()
        print(f"cos 基準（向量差分，rank {int(self.cfg['n_steps'])} token）：base={cos_base} "
              f"target={cos_target} 兩質心cos={cen_cos} → 進格門檻 cos_low={self.cos_low}（{thr_src}）",
              flush=True)
        if thr_src != "manual" and cos_target <= cos_base:
            print("  ⚠ target 基準未高於 base：兩質心幾乎重合（誘發訊號微弱），門檻可能不可靠；"
                  "可改在 config 設 cos_low=<絕對值> 定死。", flush=True)

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
        # gen-2 單調 T/K 掃描（與 prune_v2 同策略，共用 run_sweep）。prune 走翻轉率，
        # km_cos 沒有翻轉 → 跳格判準改用對比相關性 cos（ThresholdCriterion，數值型，門檻
        # cos_low 由 _prepare 以 rank 時的 cos 基準自動定標，見上）。要換成向量距離等其他
        # 訊號：讓 _evaluate 把新純量算進 cell，再換這裡的 key / criterion 即可，掃描不動。
        total = int(self.ranking["score"].numel())
        run_sweep(self._evaluate, total, self.cfg, self.state, self._save_state,
                  criterion=ThresholdCriterion("cos", self.cos_low, as_rate=False, label="cos"),
                  label="km_cos")

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
