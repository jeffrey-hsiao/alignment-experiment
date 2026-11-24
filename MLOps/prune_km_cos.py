"""
MLOps/prune_km_cos.py

km_cos（cos = cosplay）：通用「行為編輯」模式，與 km_search 平行、共用同一套引擎，
但針對「讓模型 cosplay 一個人格」這類**誘發型**行為:

  1. 分組用「對比系統提示詞」而非模型行為判定 —— 同一批題分別配 target_system
     （要誘發的人格，如貓娘）與 base_system（一般身分），捕激活成兩組。**不需要
     對每題做真實結果判定**，所以沒有 km_search 的前置分組階段。
  2. 搜尋改用 prune_v2 的 gen-2 單調 T/K 掃描（共用 run_sweep 與 ThresholdCriterion），
     走**翻轉率** flips/n，門檻 flip_low。「翻轉」用**似然度偏好**逐題判定：
       準備階段（與神經元排行同一次模型載入）對 tune 每題各生成一份參考回答：
         R_t = 原模型帶 target_system 的回答（人格版）
         R_b = 原模型帶 base_system 的回答（原本的回答）
       偏好 s = 每 token 平均 log P(R_t | 題) − 每 token 平均 log P(R_b | 題)
       每題自己的尺標：地板 s0 = 原模型、base_system 下的 s；天花板 s1 = 原模型、
       target_system 下的 s；中點 = (s0+s1)/2。
       每格：被編輯的模型、base_system 下重算 s，s ≥ 該題中點＝這題翻轉。
     不評判生成文字（teacher-forced），所以沒有「編輯後輸出跑出外部評分模型分布」的問題，
     也不依賴任何外部判定器。崩壞算沒翻轉、留在分母，不另外驅動步進。
     （舊版用 e5 句向量質心差分：兩質心 cos≈0.95、邊界≈0，等於擲硬幣，已移除。）
  3. best-cell：非崩壞格中翻轉率最高，同率取平均位置（cos 欄，0＝地板、1＝天花板）高者。

編輯方式 edit（config）：
  shift（預設）  與 vector 方法共用：選中的神經元激活偏移 a→a+δ，δ = target 平均 − base 平均；
                 搜尋用 vector 的由上往下 run_descent，最佳格＝達標格中 K 最小（pick_best）。
  legacy         舊做法：select_tables 歸零＋取負，搜尋用 run_sweep（run 022 即此設定）。
搜尋方式 search（config，留空＝跟著 edit：shift→descent、legacy→sweep）：
  sweep    prune_v2 的由下往上 run_sweep
  descent  由上往下 run_descent；collapse_high > 0 時改為崩壞驅動（崩壞率 ≥ collapse_high → 降 K；
           不崩壞且翻轉不夠 → K 還原上一步、降 T），與編輯方式無關，legacy 也能用。
           崩壞（auto_collapse）只算翻轉候選（s ≥ 中點）的題，同 prune 只在未拒絕回答裡數崩壞；
           全部題的崩壞數記在 collapse_all（僅記錄）。
shift_scale（shift 模式）：單一數字＝固定強度；逗號清單（如 1.0,0.5,0.25）＝第三維搜尋：
  依清單順序，對每個強度各跑一次完整的 T/K 搜尋（同一套 run_descent / run_sweep，不改搜尋本身）。
  每格記錄 scale，快取鍵含 scale；最佳格＝所有強度的達標格中 K 最小，同 K 取強度小者（pick_best）。
另外每格記錄 rt_flips：被編輯模型給人格版回答 R_t 的 log 機率本身是否越過該題中點
（原模型無/有 target system 之間）。它不受「兩份回答一起被傷害」影響，用來驗收翻轉是否為真。

共用：prune_search（select_tables、run_sweep、ThresholdCriterion、崩壞偵測、切分）、
prune_vector（build_ranking_shift、shift_values、run_descent、pick_best）、prune_utils（編輯與
checkpoint）。km_search / prune_v2 的拒絕路徑完全不碰。
"""
import json
import time
from pathlib import Path

import torch

from neuron_probe import load_seeds  # noqa: E402
from behavior_regroup import generate_all  # noqa: E402
from chat_prompt import render_chat  # noqa: E402
from prune_utils import apply_reverse_mask, apply_prune_mask, apply_shift_mask, write_prune_checkpoint  # noqa: E402
from prune_vector import build_ranking_shift, shift_values, run_descent, pick_best  # noqa: E402  ★與 vector 共用
from prune_search import (  # noqa: E402  ★共用引擎
    POSITIVE, NEGATIVE, select_tables, run_sweep, ThresholdCriterion, CellRunner,
    free_cuda, load_source, split_by_pair, stratified_order, is_collapsed,
)


class KmCosError(Exception):
    pass


# ── 對比排名（平行 build_ranking，但用兩個 system 而非 group 欄）──────────────────

def build_ranking_contrastive(model, tokenizer, records: list[dict], n_steps: int, seed: int,
                              target_system: str, base_system: str) -> dict:
    """同一批題分別配 target/base system 跑一次，激活進 POSITIVE/NEGATIVE 兩組，
    組出與 build_ranking 相同格式的 ranking（供 select_tables 使用），並附兩組平均激活
    （shift 模式的偏移量）。實作與 vector 共用 build_ranking_shift。"""
    return build_ranking_shift(model, tokenizer,
                               [(target_system, records, POSITIVE), (base_system, records, NEGATIVE)],
                               n_steps, seed)


# ── 似然度偏好（翻轉判定）──────────────────────────────────────────────────────

def mean_logprob(model, tokenizer, system: str, question: str, answer: str) -> float:
    """teacher-forced：在 (system, question) 的對話前綴後，answer 每個 token 的平均 log 機率。
    前綴與生成時同一個 render_chat（add_generation_prompt=True），所以量的正是「模型接下來
    會不會說出這段話」。answer 為空回傳 0.0（不該發生，參考回答都是貪婪生成的非空字串）。"""
    prefix = render_chat(tokenizer, question, system=system, add_generation_prompt=True)
    p_ids = tokenizer(prefix, return_tensors="pt").input_ids
    a_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids
    if a_ids.shape[1] == 0:
        return 0.0
    ids = torch.cat([p_ids, a_ids], 1).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, p_ids.shape[1] - 1:-1].float()
    lp = logits.log_softmax(-1).gather(1, a_ids[0].to(model.device)[:, None])[:, 0]
    return float(lp.mean())


def ref_logprobs(model, tokenizer, system: str, refs: list[dict]) -> list[tuple[float, float]]:
    """每題 (平均 log P(R_t), 平均 log P(R_b))。refs 每項 {text, r_target, r_base}。"""
    return [(mean_logprob(model, tokenizer, system, r["text"], r["r_target"]),
             mean_logprob(model, tokenizer, system, r["text"], r["r_base"])) for r in refs]



def _quiet_mp_shutdown() -> None:
    """抑制 multiprocess.resource_tracker 在 Python 3.12 直譯器關閉時印的良性
    AttributeError（'_thread.RLock' object has no attribute '_recursion_count'）。
    崩壞偵測的 reranker / 亂碼分類器會拉進 multiprocess 套件；run 正常完成、checkpoint 都
    寫好後，關閉期仍會印這段 traceback，看起來像「結束時崩潰」。只把壞掉的 _stop_locked
    包成吞掉例外，不改離開流程、不影響結果。安裝一次即可。"""
    try:
        from multiprocess import resource_tracker as _rt
        if getattr(_rt.ResourceTracker, "_km_quiet", False):
            return
        _orig = _rt.ResourceTracker._stop_locked
        def _safe(self, *a, **k):
            try:
                return _orig(self, *a, **k)
            except Exception:
                pass
        _rt.ResourceTracker._stop_locked = _safe
        _rt.ResourceTracker._km_quiet = True
    except Exception:
        pass


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
        self.target_system = ""
        self.base_system = ""
        self.refs: list[dict] = []   # tune 每題 {id, text, r_target, r_base, floor, ceil, mid}
        self.flip_low = None         # 翻轉率門檻（flip_rate ≥ 此 → 衰減 K；否則長 T）
        self._cur_scale = None       # shift 模式當下的強度（多強度搜尋時逐一切換）；legacy 為 None

    # ── 工具 ─────────────────────────────────────────────────────────────────
    def _resolve_systems(self):
        """target_system：inline 或 target_system_file；base_system：inline 或 base_system_file。

        base_system 留空＝**真的不帶 system**（空字串），對照組為「無提示詞」的原生輸出，
        不再 fallback 到原廠身分 / 通用助理。這樣對比乾淨：target 人格 vs 無 system。"""
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

    # ── 崩壞偵測 ─────────────────────────────────────────────────────────────
    def _load_scorers(self):
        from quality_pilot import GIBBERISH, RERANKER, load  # noqa: E402  延後載入
        self._gib = load(GIBBERISH)
        self.rerank = load(RERANKER)   # 崩壞偵測的 relevance

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

    # ── 選點 + 生成 ───────────────────────────────────────────────────────────
    @property
    def _shift_mode(self) -> bool:
        edit = str(self.cfg.get("edit", "shift"))
        if edit not in ("shift", "legacy"):
            raise KmCosError(f"edit 只能是 shift / legacy：{edit}")
        return edit == "shift"

    @property
    def _search(self) -> str:
        search = str(self.cfg.get("search", "") or "") or ("descent" if self._shift_mode else "sweep")
        if search not in ("sweep", "descent"):
            raise KmCosError(f"search 只能是 sweep / descent：{search}")
        return search

    @property
    def _scales(self) -> list:
        """shift 模式的強度清單（shift_scale 可為單一數字或逗號清單）；legacy 回 [None]。"""
        if not self._shift_mode:
            return [None]
        raw = str(self.cfg.get("shift_scale", 1.0) or 1.0)
        try:
            vals = [float(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            raise KmCosError(f"shift_scale 要是數字或逗號分隔的數字：{raw}")
        if not vals or any(v <= 0 for v in vals) or len(set(vals)) != len(vals):
            raise KmCosError(f"shift_scale 需為不重複的正數：{raw}")
        return vals

    def _find(self, kind: str, k: int, m: int, repeat: int = 0):
        """快取鍵多一維 scale。舊格（沒有 scale 欄）視為單一強度設定下的那個強度，resume 相容。"""
        scales = self._scales
        default = scales[0] if len(scales) == 1 else None
        for cell in self.state["cells"]:
            if ((cell["kind"], cell["k"], cell["m"], cell["repeat"]) == (kind, k, m, repeat)
                    and cell.get("scale", default) == self._cur_scale):
                return cell
        return None

    def _select(self, k: int, m: int, repeat: int = 0, random_control: bool = False):
        """回傳 (歸零清單, 取負清單, 偏移清單 [(層,索引,δ)], 選點統計)。
        legacy：select_tables 雙表語意，翻轉型取負、其餘歸零（偏移清單為空）。
        shift ：同一份 select_tables 選點，但選中的全部偏移 a→a+δ（推向 target 平均）。
        隨機對照：同數量隨機節點，套同一種編輯（legacy 一律取負；shift 用各自的 δ）。"""
        nn = self.ranking["n_neurons"]
        to_pair = lambda i: (i // nn, i % nn)
        sign = str(self.cfg.get("sign", "both"))
        zeroed, reversed_pairs, stats = select_tables(self.ranking, k, m, sign, True)
        picked = sorted(set(zeroed) | set(reversed_pairs))
        if random_control:
            g = torch.Generator().manual_seed(int(self.cfg["split_seed"]) + 1000 + repeat)
            chosen = torch.randperm(self.ranking["score"].numel(), generator=g)[:len(picked)]
            picked = [to_pair(i) for i in chosen.tolist()]
            zeroed, reversed_pairs = [], picked
        if self._shift_mode:
            return [], [], shift_values(self.ranking, picked, "positive", self._cur_scale), stats
        return zeroed, reversed_pairs, [], stats

    def _generate_nosys(self, zeroed, reversed_neurons, shifts, prompts):
        """關鍵：誘發後要**不帶 target_system** 測，看行為有沒有 transfer 到權重。
        同一次模型載入：先生成（崩壞偵測 + 人工覆核用），再算兩份參考回答的 log 機率（翻轉判定用）。
        回傳 (details, 每題 (logP(R_t), logP(R_b)))。"""
        model, tokenizer = self._load_model()
        if shifts:   # 與 apply_masks 同順序：偏移先算（相對底模權重），再歸零/取負
            apply_shift_mask(model, shifts, verbose=False)
        if zeroed:
            apply_prune_mask(model, zeroed, verbose=False)
        if reversed_neurons:
            apply_reverse_mask(model, reversed_neurons, verbose=False)
        details = generate_all(model, tokenizer, prompts, self.base_system, int(self.cfg["gen_max_tokens"]))
        lps = ref_logprobs(model, tokenizer, self.base_system, self.refs)
        del model
        free_cuda()
        return details, lps

    def _checkpoint(self, name, zeroed, reversed_neurons, shifts, **meta):
        return write_prune_checkpoint(self.output_dir / name, self.cfg["model_name"], zeroed,
                                      reversed_neurons=reversed_neurons, shifted=shifts, **meta)

    # ── 單格評估（注入給 run_sweep 的 evaluate）──────────────────────────────
    def _evaluate(self, kind: str, k: int, m: int, repeat: int = 0) -> dict:
        cached = self._find(kind, k, m, repeat)
        if cached is not None:
            return cached
        started = time.time()
        zeroed, reversed_neurons, shifts, stats = self._select(k, m, repeat, random_control=(kind == "random"))
        prompts = [self.records[i] for i in self.state["prompts"]["tune"]]
        rows, lps = self._generate_nosys(zeroed, reversed_neurons, shifts, prompts)
        collapse_flags = self._collapse(rows)
        n = len(rows)
        # 逐題：偏好 s ≥ 該題中點＝翻轉候選；候選且未崩壞＝翻轉。崩壞算「沒翻轉」，仍留在分母 n。
        # 崩壞計數（auto_collapse，驅動 collapse_high）只算翻轉候選——對齊 prune 只在「未拒絕」
        # 的回答裡數崩壞（prune_search._evaluate）。沒翻轉的回答多是原身分推託（「作為一個AI…」），
        # 相關性分數低會被誤判崩壞，原模型自己就 4/14。全部題的崩壞數另記在 collapse_all。
        # 位置 = (s − 地板)/(天花板 − 地板)：0＝原模型、1＝帶 target system，報表用。
        # rt_flip：只看 logP(R_t) 本身是否越過 (t0+t1)/2——傷害只會讓它下降，不會造成假翻轉。
        flips, rt_flips, collapse, positions = 0, 0, 0, []
        for row, (lt, lb), ref, bad in zip(rows, lps, self.refs, collapse_flags):
            s = lt - lb
            pos = (s - ref["floor"]) / (ref["ceil"] - ref["floor"])
            collapse += (s >= ref["mid"]) and bad
            flip = (s >= ref["mid"]) and not bad
            rt_pos = (lt - ref["t0"]) / (ref["t1"] - ref["t0"]) if ref["t1"] != ref["t0"] else 0.0
            rt_flip = (rt_pos >= 0.5) and not bad
            row.update(pref=round(s, 4), pref_pos=round(pos, 4), flip=flip, collapsed=bool(bad),
                       lp_target=round(lt, 4), rt_pos=round(rt_pos, 4), rt_flip=rt_flip)
            flips += flip
            rt_flips += rt_flip
            if not bad:
                positions.append(pos)
        cos = round(sum(positions) / len(positions), 4) if positions else 0.0
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        tag = f"_s{self._cur_scale:g}" if len(self._scales) > 1 else ""   # 單一強度維持舊檔名
        (self.cells_dir / f"{kind}_K{k}_M{m}_r{repeat}{tag}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        cell = {
            "step": len(self.state["cells"]) + 1, "kind": kind, "k": k, "m": m, "repeat": repeat,
            # cos 欄沿用名稱（GUI / mk-review 讀它）：非崩壞題的平均位置（0＝地板、1＝天花板）
            "n": n, "auto_collapse": collapse, "collapse_all": sum(collapse_flags), "cos": cos,
            # run_sweep 步進訊號：翻轉率 flips/n（似然度越過每題中點；崩壞算沒翻轉、留分母）
            "flips": flips,
            # 驗收用（不驅動步進）：logP(R_t) 本身越過中點的題數
            "rt_flips": rt_flips,
            "n_reversed": len(reversed_neurons), "n_pruned": len(zeroed), "n_shifted": len(shifts),
            "seconds": round(time.time() - started), **stats,
        }
        if self._cur_scale is not None:
            cell["scale"] = self._cur_scale
        self.state["cells"].append(cell)
        self._save_state()
        # 每格都寫一個可載入的 checkpoint（checkpoint-<step>），與 prune 一致：之後能用
        # model_loader 載任一格的 cosplay 模型比較，不必只有結尾的 final。cache 命中會提早
        # return，所以 resume 不會重寫已存在的格。
        self._checkpoint(f"checkpoint-{cell['step']}", zeroed, reversed_neurons, shifts,
                         phase=kind, k=k, m=m, **({"scale": self._cur_scale} if self._cur_scale else {}),
                         metrics={"flips": flips, "rt_flips": rt_flips, "n": n, "cos": cos,
                                  "auto_collapse": collapse})
        rate = flips / max(1, n)
        edited = f"偏移 {len(shifts)}" if shifts else f"歸零 {len(zeroed)} 取負 {len(reversed_neurons)}"
        sc = f" s={self._cur_scale:g}" if self._cur_scale is not None else ""
        print(f"[{kind}]{sc} K={k} M={m} r={repeat} 崩壞 {collapse}/{n}（全部 {cell['collapse_all']}）翻轉 {flips}/{n}"
              f"（率 {rate:.0%}）R_t 翻轉 {rt_flips}/{n} 平均位置={cos} {edited} {cell['seconds']}秒",
              flush=True)
        return cell

    # ── 流程 ─────────────────────────────────────────────────────────────────
    def _prepare_refs(self, model, tokenizer, tune: list[dict]) -> None:
        """翻轉判定的參考回答與每題尺標，和神經元排行共用同一次模型載入。

        對 tune 每題生成 R_t（target_system）與 R_b（base_system），長度 = gen_max_tokens（與每格
        評估一致）；再用原模型算地板 s0（base_system 下）與天花板 s1（target_system 下）。
        天花板 ≤ 地板的題目分不開人格，直接報錯而不是默默納入（那會讓翻轉率失真）。"""
        gen = int(self.cfg["gen_max_tokens"])
        print(f"翻轉判定參考回答：tune {len(tune)} 題 × 兩條件各生成 {gen} token", flush=True)
        r_t = generate_all(model, tokenizer, tune, self.target_system, gen)
        r_b = generate_all(model, tokenizer, tune, self.base_system, gen)
        refs = [{"id": r["id"], "text": r["text"], "r_target": t["response"], "r_base": b["response"]}
                for r, t, b in zip(tune, r_t, r_b)]
        lp0 = ref_logprobs(model, tokenizer, self.base_system, refs)
        lp1 = ref_logprobs(model, tokenizer, self.target_system, refs)
        for ref, (t0, b0), (t1, b1) in zip(refs, lp0, lp1):
            s0, s1 = t0 - b0, t1 - b1
            # t0/t1：R_t 自己的尺標（驗收用 rt_flip）
            ref.update(floor=round(s0, 4), ceil=round(s1, 4), mid=round((s0 + s1) / 2, 4),
                       t0=round(t0, 4), t1=round(t1, 4))
        bad = [r["id"] for r in refs if r["ceil"] <= r["floor"]]
        if bad:
            raise KmCosError(f"這些題目天花板 ≤ 地板（target 與 base 分不開）：{bad}；"
                             "請檢查 target/base system 是否真的對比，或從 seeds 移除這些題。")
        self.refs = refs
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        (self.cells_dir / "refs.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in refs) + "\n", encoding="utf-8")
        gaps = [r["ceil"] - r["floor"] for r in refs]
        print(f"每題尺標（似然度偏好）：天花板−地板 平均 {sum(gaps) / len(gaps):.3f}"
              f"，最小 {min(gaps):.3f}（全部 {len(refs)} 題天花板 > 地板）", flush=True)

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
        tune = [self.records[i] for i in self.state["prompts"]["tune"]]
        # 同一次模型載入：rank 集做對比排名，tune 集做翻轉判定的參考回答與尺標
        model, tokenizer = self._load_model()
        self.ranking = build_ranking_contrastive(
            model, tokenizer, splits["rank"], int(self.cfg["n_steps"]), seed,
            self.target_system, self.base_system)
        self._prepare_refs(model, tokenizer, tune)
        del model
        free_cuda()
        torch.save(self.ranking, self.exp_dir / "ranking.pt")

        self.flip_low = float(self.cfg.get("flip_low", 1.0))
        gaps = [r["ceil"] - r["floor"] for r in self.refs]
        # 欄名沿用 cos_baseline（mk-review 讀 flip_low）；內容改為似然度尺標摘要
        self.state["cos_baseline"] = {"signal": "loglik_pref", "flip_low": self.flip_low,
                                      "edit": "shift" if self._shift_mode else "legacy",
                                      "search": self._search,
                                      "gap_mean": round(sum(gaps) / len(gaps), 4),
                                      "gap_min": round(min(gaps), 4),
                                      "n_tune": len(self.refs), "n_rank": len(splits["rank"]),
                                      "gen_tokens": int(self.cfg["gen_max_tokens"])}
        self._save_state()
        print(f"翻轉＝似然度偏好越過每題中點；flip_low={self.flip_low:.0%}", flush=True)

    def train(self, resume_from_checkpoint=False):
        import types
        from transformers import TrainerControl, TrainerState

        _quiet_mp_shutdown()   # 抑制 multiprocess 在關閉期的良性 traceback（看起來像崩潰）
        if self.state_path.exists():
            if not resume_from_checkpoint:
                raise KmCosError(f"{self.exp_dir.name} 已有結果，要接續請用："
                                 f"python run.py train km_cos --resume {self.exp_dir.name}")
            self._load_state()   # 讀回已完成的格子；_prepare 仍重建 ranking/參考回答但不覆寫 cells
        state, control = TrainerState(), TrainerControl()
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))
        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        self._prepare()
        # 「翻轉」＝似然度偏好越過每題中點（_evaluate 算進 cell["flips"]），flip_rate = flips/n，
        # 門檻 flip_low，判準沿用 ThresholdCriterion。搜尋方式見 _search（sweep / descent）。
        total = int(self.ranking["score"].numel())
        criterion = ThresholdCriterion("flips", self.flip_low, as_rate=True, label="flip")
        ch = float(self.cfg.get("collapse_high", 0) or 0)
        collapse = (ThresholdCriterion("auto_collapse", ch, as_rate=True, label="崩壞")
                    if ch > 0 and self._search == "descent" else None)
        # 第三維：每個強度各跑一次完整搜尋（legacy 只有 [None] 一輪）。搜尋函式本身不動。
        scales = self._scales
        by_scale = self.state.setdefault("v2_by_scale", {})
        for scale in scales:
            self._cur_scale = scale
            label = "km_cos" + (f" s={scale:g}" if len(scales) > 1 else "")
            if self._search == "descent":
                # 與 vector 共用由上往下搜尋（collapse_high>0 → 崩壞驅動）
                run_descent(self._evaluate, total, self.cfg, self.state, self._save_state,
                            criterion=criterion, collapse=collapse, label=label)
            else:
                run_sweep(self._evaluate, total, self.cfg, self.state, self._save_state,
                          criterion=criterion, label=label)
            if scale is not None:
                by_scale[f"{scale:g}"] = dict(self.state.get("v2") or {})
                self._save_state()
        if self._search == "descent":
            best = pick_best(self.state["cells"], criterion, collapse)   # 跨強度：K 最小，同 K 取強度小
        else:
            # best-cell：非崩壞格中 翻轉率最高（同率取平均位置 cos 高者）
            sync = [c for c in self.state["cells"] if c["kind"] == "sync"]
            viable = [c for c in sync if c["auto_collapse"] < c["n"]] or sync
            best = max(viable, key=lambda c: (c["flips"] / max(1, c["n"]), c["cos"])) if viable else None
        top = [best] if best else []
        if best:
            self._cur_scale = best.get("scale", scales[0])   # 隨機對照與 final 用最佳格的強度
        for repeat in range(int(self.cfg["control_repeats"])):
            for c in top:
                self._evaluate("random", c["k"], c["m"], repeat)
        self.state["top"] = [{"k": c["k"], "m": c["m"], "flips": c["flips"],
                              "rt_flips": c.get("rt_flips"), "n": c["n"], "cos": c["cos"],
                              **({"scale": c["scale"]} if "scale" in c else {})} for c in top]
        self.state["finished"] = True
        self._save_state()

        if best:
            zeroed, reversed_neurons, shifts, _ = self._select(best["k"], best["m"], 0)
            self._checkpoint("final", zeroed, reversed_neurons, shifts,
                             phase="final", k=best["k"], m=best["m"],
                             **({"scale": self._cur_scale} if self._cur_scale else {}))
        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)
