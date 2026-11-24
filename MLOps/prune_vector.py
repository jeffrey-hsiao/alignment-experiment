"""
MLOps/prune_vector.py

vector：prune_v2 的「激活偏移」版（解除對齊），同時提供 km_cos 共用的偏移與下降搜尋函式。

與 prune_v2 的差別只有兩處，其餘（分組、切分、select_tables 選點、拒絕判定、崩壞偵測、
每格 checkpoint、callback）全部直接沿用 prune_v2 / km_search 的實作，不複製：

  1. 編輯方式：選中的神經元不歸零、不取負，而是把激活偏移 a → a + δᵢ。
       δᵢ = 目標組平均激活 − 來源組平均激活（× shift_scale）
     解除對齊：來源＝拒絕組、目標＝順從組（把拒絕時的激活推向順從時的平均）。
     km_cos：  來源＝無 system、目標＝target system（把原生激活推向人格的平均）。
     δ 只加在被選中的 K 顆神經元上；down_proj 是線性的，所以等價於每層一個 bias
     （prune_utils.apply_shift_mask）。平均激活在神經元排行時順便算（build_ranking_shift）。

  2. 搜尋方向：由上往下（run_descent）。
     歸零/取負只能「拿掉」行為，所以 prune_v2 由小 T 往上找；偏移在大 K 時一定推過頭
     （人格/順從已出現，但伴隨崩壞），正好符合「先確定有效果、再降 K 找最小編輯」的前提：
       起點 T = t_max、K = T（M = 0）
       翻轉率 ≥ flip_low → 記為候選，K ×= k_decay（T 不動）
       翻轉率 <  flip_low → K 還原成最後一個達標的 K，T −= step（K 不可超過 T）
       T 降到 0 以下即結束。
     起點若全崩壞（km_cos 的崩壞不算翻轉）→ 沒有達標過的 K，就沿 K = T 的對角線往下，
     直到效果出現才開始降 K，過度偏移區自動跳過，不需另設崩壞門檻。
     最佳格＝所有達標格中 K 最小者（同 K 取崩壞較少）。

跳格判準沿用 prune_search.ThresholdCriterion；prune_search.run_sweep 與 prune_v2 完全不動。
（KmSearchRunner._prepare 內部呼叫 prune_search.build_ranking，這裡在 _prepare 期間把它暫時
換成 build_ranking_shift——輸出格式相同、多兩欄平均激活——結束即還原，不改 prune_search。）
"""
import json
from contextlib import contextmanager

import torch

import prune_search
from neuron_probe import GenerationProbe, GroupStats
from prune_search import (KmSearchError, ThresholdCriterion, POSITIVE, NEGATIVE,
                          rate_order)
from prune_utils import apply_shift_mask, write_prune_checkpoint
from prune_v2_search import PruneV2Runner


# ── 共用函式（vector 與 km_cos 都用）──────────────────────────────────────────

def build_ranking_shift(model, tokenizer, jobs, n_steps: int, seed: int) -> dict:
    """與 prune_search.build_ranking 同格式的排行，多存兩組平均激活（偏移量的原料）。

    jobs：[(system, records, group)]。group=None → 用每筆 record["group"]（解除對齊：同一個
    system、按拒絕/順從分組）；group 指定 → 整批都進該組（km_cos：同一批題分別配
    target/base system）。"""
    pooled = {POSITIVE: GroupStats(), NEGATIVE: GroupStats()}
    for system, records, group in jobs:
        with GenerationProbe(model, tokenizer, system=system) as probe:
            for i, record in enumerate(records, 1):
                acts, _ = probe.run(record["text"], n_steps)
                g = group or record["group"]
                for position in range(acts.shape[0]):
                    pooled[g].update(acts[position], 0.0)
                if i % 20 == 0 or i == len(records):
                    print(f"  排序探測{f' [{group}]' if group else ''} {i}/{len(records)} 筆", flush=True)
    pos, neg = pooled[POSITIVE], pooled[NEGATIVE]
    score = (pos.mean - neg.mean) / ((pos.std ** 2 + neg.std ** 2) / 2).sqrt().clamp_min(1e-6)
    return {
        "n_neurons": score.shape[1],
        "positive":  rate_order(pos),
        "negative":  rate_order(neg),
        "random":    torch.randperm(score.numel(), generator=torch.Generator().manual_seed(seed)),
        "score":     score.flatten(),
        "refused_pos_rate":  pos.active_rate.flatten(),
        "complied_pos_rate": neg.active_rate.flatten(),
        "n_tokens":  {POSITIVE: pos.n, NEGATIVE: neg.n},
        # 偏移量原料：兩組每顆神經元的平均激活（down_proj 的輸入）
        "pos_mean":  pos.mean.flatten(),
        "neg_mean":  neg.mean.flatten(),
    }


def shift_values(ranking: dict, pairs, toward: str, scale: float = 1.0) -> list[tuple[int, int, float]]:
    """選中的 (層, 索引) → [(層, 索引, δ)]。toward='negative'：δ = neg_mean − pos_mean
    （推向 NEGATIVE 組）；toward='positive'：δ = pos_mean − neg_mean。"""
    if "pos_mean" not in ranking:
        raise KmSearchError("這份 ranking 沒有平均激活（pos_mean/neg_mean），偏移需要重新排序"
                            "（ranking_path 指向的是舊版 ranking.pt）")
    nn = ranking["n_neurons"]
    diff = ranking["neg_mean"] - ranking["pos_mean"]
    if toward == "positive":
        diff = -diff
    elif toward != "negative":
        raise ValueError(f"toward 只能是 positive / negative：{toward}")
    return [(l, i, float(diff[l * nn + i]) * scale) for l, i in pairs]


def run_descent(evaluate, total: int, cfg, state: dict, save, *,
                criterion, collapse=None, label: str = "descent") -> dict:
    """由上往下的 T/K 搜尋（見檔頭）。evaluate(kind,k,m)->cell 內建快取 → resume 確定性重放。

    collapse（選填，同樣是 ThresholdCriterion，如 'auto_collapse' 比例）：給了就改用崩壞驅動，
    不再假設起點一定是最高翻轉：
      崩壞率 ≥ collapse.low              → K ×= k_decay（推太強）
      不崩壞、翻轉率 ≥ criterion.low     → 候選，K ×= k_decay（找更小的 K）
      不崩壞、翻轉率 <  criterion.low    → K 還原成上一步（降 K 之前的值），T −= step（K ≤ T）
    沒給 collapse 時維持原規則（達標降 K；未達標還原成最後一個達標的 K 並降 T）。"""
    sweep_steps = max(1.0, float(cfg["sweep_steps"]))
    t_max_ratio = min(1.0, max(1e-6, float(cfg.get("t_max_ratio", 1.0) or 1.0)))
    t_max = min(total, max(2, round(t_max_ratio * total)))
    step = max(1, round(t_max / sweep_steps))
    decay = float(cfg["k_decay"])
    T, K = t_max, t_max
    k_good = None   # 無 collapse：最後一個達標的 K
    k_prev = None   # 有 collapse：降 K 之前的值（「上一步」）

    state["v2"] = {"total": total, "t_max": t_max, "t_max_ratio": t_max_ratio,
                   "step": step, "sweep_steps": sweep_steps, "k_decay": decay,
                   "low": getattr(criterion, "low", None), "signal": criterion.label,
                   "direction": "descent",
                   "collapse_high": getattr(collapse, "low", None)}
    state["bounds"] = {"source": label, "start_k": t_max, "end_m": t_max}
    save()
    print(f"{label} 下降搜尋：總神經元 {total}，起點 T=K={t_max}（{t_max_ratio:.0%}），"
          f"T 步長 {step}，k_decay={decay}，{criterion.describe()}"
          + (f"，崩壞驅動 {collapse.describe()}" if collapse else ""), flush=True)

    while T >= 1:
        K = max(1, min(T, K))
        cell = evaluate("sync", K, T - K)
        too_strong = collapse is not None and collapse.reached(cell)
        if too_strong or criterion.reached(cell):
            if not too_strong:
                k_good = K
            nK = max(1, round(K * decay))
            if nK < K:
                k_prev = K
                K = nK              # 崩壞過多 / 達標 → 降 K，T 不動
                save()
                continue
            # K 已到底 → 改降 T
        elif collapse is not None:
            if k_prev is not None:
                K = k_prev          # 不崩壞但翻轉不夠 → K 還原上一步
        elif k_good is not None:
            K = k_good              # 未達標 → K 還原成最後一個達標值
        k_prev = None
        T -= step                   # 降 T（下一圈 K 會被夾到 ≤ T）
        save()
    return state["v2"]


def pick_best(cells: list[dict], criterion, collapse=None) -> dict | None:
    """達標格中 K 最小（同 K 取崩壞少、T 小）；都沒達標則取訊號最高的格。
    給了 collapse 時，崩壞過多的格不算達標。"""
    sync = [c for c in cells if c["kind"] == "sync"]
    hit = [c for c in sync if criterion.reached(c) and not (collapse and collapse.reached(c))]
    if hit:
        # 同 K 取強度小者（km_cos 多強度搜尋；沒有 scale 欄的方法不受影響）
        return min(hit, key=lambda c: (c["k"], c.get("scale", 0), c["auto_collapse"], c["k"] + c["m"]))
    return max(sync, key=criterion.value) if sync else None


@contextmanager
def ranking_with_means():
    """暫時把 prune_search.build_ranking 換成 build_ranking_shift（同格式＋平均激活），
    讓 KmSearchRunner._prepare 原封不動地產出帶 pos_mean/neg_mean 的 ranking.pt。"""
    orig = prune_search.build_ranking
    prune_search.build_ranking = (lambda model, tokenizer, records, n_steps, seed, system:
                                  build_ranking_shift(model, tokenizer, [(system, records, None)],
                                                      n_steps, seed))
    try:
        yield
    finally:
        prune_search.build_ranking = orig


# ── vector（解除對齊）runner ─────────────────────────────────────────────────

class VectorRunner(PruneV2Runner):
    """prune_v2 的偏移版。_evaluate / _judge / _publish / 分組 / 切分全部繼承；
    只換「選中的神經元怎麼編輯」（偏移）與「搜尋方向」（run_descent）。

    慣例：_select 回傳的第一個清單（prune 系列的「歸零清單」）在這裡是「偏移清單」，
    所以 cell 的 n_pruned 欄位＝偏移的神經元數。"""

    TOWARD = "negative"   # 拒絕組 → 順從組

    def _shifts(self, pairs) -> list[tuple[int, int, float]]:
        raw = str(self.cfg.get("shift_scale", 1.0))
        if "," in raw:
            raise KmSearchError("vector 目前只支援單一 shift_scale；多強度搜尋目前只在 km_cos 實作")
        return shift_values(self.ranking, pairs, self.TOWARD, float(raw))

    def _select(self, kind: str, k: int, m: int, repeat: int):
        # 沿用 prune 的選點（含隨機對照的同數量規則）；偏移不分歸零/取負，兩份合併
        zeroed, reversed_pairs, removed, chosen, stats = super()._select(kind, k, m, repeat)
        return sorted(set(zeroed) | set(reversed_pairs)), [], removed, chosen, stats

    def _generate(self, neurons: list, prompts: list[dict],
                  reversed_neurons: list | None = None) -> list[dict]:
        from behavior_regroup import generate_all
        from prune_search import free_cuda
        model, tokenizer = self._load_model()
        if neurons:
            apply_shift_mask(model, self._shifts(neurons), verbose=False)
        details = generate_all(model, tokenizer, prompts, self.system, int(self.cfg["gen_max_tokens"]))
        del model
        free_cuda()
        return details

    def _write_checkpoint(self, name: str, zeroed, reversed_neurons, **meta):
        return write_prune_checkpoint(self.output_dir / name, self.cfg["model_name"], [],
                                      shifted=self._shifts(zeroed), **meta)

    def _prepare(self) -> None:
        with ranking_with_means():
            super()._prepare()
        if "pos_mean" not in self.ranking:
            raise KmSearchError("ranking 沒有平均激活：vector 需要重新排序（不要用 prune_v2 的 ranking_path）")

    def train(self, resume_from_checkpoint=False):
        import types

        from transformers import TrainerControl, TrainerState

        if self.state_path.exists():
            if not resume_from_checkpoint:
                raise KmSearchError(f"{self.exp_dir.name} 已有結果，要接續請用："
                                    f"python run.py train vector --resume {self.exp_dir.name}")
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            print(f"接續：已完成 {len(self.state['cells'])} 格", flush=True)

        state, control = TrainerState(), TrainerControl()
        self.tracker = (state, control)
        fake_args = types.SimpleNamespace(output_dir=str(self.output_dir))
        for cb in self.callbacks:
            cb.on_train_begin(fake_args, state, control)

        self._prepare()
        criterion = ThresholdCriterion("flips", float(self.cfg["flip_low"]), as_rate=True)
        run_descent(self._evaluate, int(self.ranking["score"].numel()), self.cfg, self.state,
                    self._save_state, criterion=criterion, label="vector")

        best = pick_best(self.state["cells"], criterion)
        top = [best] if best else []
        for repeat in range(int(self.cfg["control_repeats"])):
            for c in top:
                self._evaluate("random", c["k"], c["m"], repeat)
        self.state["top"] = [{"k": c["k"], "m": c["m"]} for c in top]
        self.state["finished"] = True
        self._save_state()

        if best:
            shifted, _, _, _, _ = self._select("sync", best["k"], best["m"], 0)
            self._write_checkpoint("final", shifted, [], phase="final", k=best["k"], m=best["m"])

        for cb in self.callbacks:
            cb.on_train_end(fake_args, state, control)
        self._print_summary()
