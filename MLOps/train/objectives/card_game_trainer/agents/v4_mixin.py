"""
V4 共用元件：V4Mixin

覆寫 _run_phase1，將 7 個數字的分桶值附加至 info_key，
使跨局 UCB / 後悔值統計能區分對手歷史背景不同的相同可觀測局面。

暖啟動機制：
  load() 時若 V3 存檔存在，自動載入並縮放 V3 統計量作為先驗。
  縮放原則：每個 base_key 的訪問總量不超過 _WARMSTART_CAP，
  使 V4 自身積累的資料在少數幾局後即可主導決策。
  V4 key 首次被訪問時，若 visit=0，注入對應 base_key 的先驗值；
  已有 V4 自身資料的 key 不受影響。

分桶設計（粗粒度以確保跨局樣本足夠累積）：
  T[p]   累計出牌總和  → 每 15 一桶（典型遊戲 ~10 桶）
  I[p]   累計留牌利息  → 4 級：0–3 / 4–8 / 9–15 / 16+
  Δ[p]   累計受傷      → 每 10 一桶（典型遊戲 ~6 桶）
  R      回合數        → 3 段：早（0–3）/ 中（4–8）/ 晚（9+）

使用方式：class FooAgentV4(V4Mixin, FooAgentV3)
MRO 保證 _run_phase1 / _merge_ucb / load 由 V4Mixin 攔截，
其餘邏輯完全沿用上層。
"""
from __future__ import annotations
import os
import pickle
from collections import defaultdict
from agents.v2_mixin import _AugState


class V4Mixin:

    _WARMSTART_CAP: int = 50  # 暖啟動後每個 base_key 的訪問量上限

    # ── 分桶 ──────────────────────────────────────────────

    @staticmethod
    def _bucket_T(v: float) -> int:
        return int(max(0.0, v)) // 15

    @staticmethod
    def _bucket_I(v: float) -> int:
        v = int(max(0.0, v))
        if v <= 3:  return 0
        if v <= 8:  return 1
        if v <= 15: return 2
        return 3

    @staticmethod
    def _bucket_delta(v: float) -> int:
        return int(max(0.0, v)) // 10

    @staticmethod
    def _bucket_R(R: int) -> int:
        if R <= 3: return 0
        if R <= 8: return 1
        return 2

    def _seven_buckets(self, state) -> tuple:
        if isinstance(state, _AugState):
            T, I, delta, R = state.T, state.I, state.delta, state.R
        else:
            T, I, delta, R = self._T, self._I, self._delta, self._R
        my  = self.player_id
        opp = 1 - my
        return (
            self._bucket_T(T[my]),         self._bucket_T(T[opp]),
            self._bucket_I(I[my]),         self._bucket_I(I[opp]),
            self._bucket_delta(delta[my]), self._bucket_delta(delta[opp]),
            self._bucket_R(R),
        )

    def _run_phase1(self, state) -> tuple:
        action_values, counts, obs, info_key = super()._run_phase1(state)
        return action_values, counts, obs, info_key + self._seven_buckets(state)

    # ── 暖啟動 ────────────────────────────────────────────

    def _apply_v3_warmstart(self, v3_data: dict) -> None:
        """縮放 V3 統計量：每個 base_key 的訪問總量不超過 _WARMSTART_CAP。"""
        cap    = self._WARMSTART_CAP
        ucb_v  = v3_data.get("ucb_values", {})
        ucb_n  = v3_data.get("ucb_visits", {})

        totals: dict = defaultdict(float)
        for (key, _), n in ucb_n.items():
            totals[key] += n

        self._ws_ucb_values: dict = {}
        self._ws_ucb_visits: dict = {}
        for (key, a), n in ucb_n.items():
            t     = totals[key]
            scale = cap / t if t > cap else 1.0
            self._ws_ucb_values[(key, a)] = ucb_v.get((key, a), 0.0) * scale
            self._ws_ucb_visits[(key, a)] = round(n * scale)

        # MCCFR 附加統計量
        if "regrets" in v3_data:
            v3_reg   = v3_data["regrets"]
            v3_strat = v3_data.get("strategy_sum", {})
            strat_totals: dict = defaultdict(float)
            for (key, _), s in v3_strat.items():
                strat_totals[key] += s
            self._ws_regrets:  dict = {}
            self._ws_strategy: dict = {}
            for (key, a) in v3_reg:
                t     = strat_totals.get(key, 0.0)
                scale = cap / t if t > cap else 1.0
                self._ws_regrets[(key, a)]  = v3_reg[(key, a)] * scale
                self._ws_strategy[(key, a)] = v3_strat.get((key, a), 0.0) * scale

    def _init_from_warmstart(self, info_key: tuple, actions) -> None:
        """首次訪問此 V4 key 時，注入縮放後的 V3 先驗（不覆蓋已有 V4 資料）。"""
        if not hasattr(self, '_ws_ucb_visits'):
            return
        base_key = info_key[:6]
        for a in actions:
            if self._ucb_visits.get((info_key, a), 0) == 0:
                self._ucb_values[(info_key, a)] = self._ws_ucb_values.get((base_key, a), 0.0)
                self._ucb_visits[(info_key, a)] = self._ws_ucb_visits.get((base_key, a), 0)
        if hasattr(self, '_regrets') and hasattr(self, '_ws_regrets'):
            for a in actions:
                if self._strategy_sum.get((info_key, a), 0.0) == 0.0:
                    self._regrets[(info_key, a)]      = self._ws_regrets.get((base_key, a), 0.0)
                    self._strategy_sum[(info_key, a)] = self._ws_strategy.get((base_key, a), 0.0)

    def _merge_ucb(self, info_key: tuple, actions, phase1_values, p1_counts):
        primed = getattr(self, '_v4_primed', None)
        if primed is None:
            self._v4_primed = primed = set()
        if info_key not in primed:
            self._init_from_warmstart(info_key, actions)
            primed.add(info_key)
        return super()._merge_ucb(info_key, actions, phase1_values, p1_counts)

    def load(self, path: str) -> None:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            super().load(path)
        else:
            v3_path = path.replace("v4", "v3")
            if os.path.exists(v3_path) and os.path.getsize(v3_path) > 0:
                with open(v3_path, "rb") as f:
                    v3_data = pickle.load(f)
                self._apply_v3_warmstart(v3_data)
        self._v4_primed: set = set()
