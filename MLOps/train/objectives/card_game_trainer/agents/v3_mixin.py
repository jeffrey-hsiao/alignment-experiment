"""
V3 共用元件：_V3OppSampler 與 V3Mixin

V3Mixin 覆寫 _make_dynamic_obs，以模擬式採樣取代公式估計：
  1. 傳入對手剩餘總價值、牌庫剩餘張數、手牌數量
  2. 隨機生成牌庫牌（1~10 均勻），扣除後得手牌總值
  3. 將手牌總值隨機切割成 hand_size 份

使用方式：class FooAgentV3(V3Mixin, FooAgentV2)
"""
from __future__ import annotations
import os
import random
from agents.v2_mixin import _AugState


class _V3OppSampler:
    """V3 對手手牌採樣器：模擬牌庫後將剩餘總值隨機切割給手牌。"""

    def __init__(self, total_remaining: float, deck_est: int, hand_size: int, max_card: int):
        self._total = total_remaining
        self._deck_est = deck_est
        self._n = hand_size
        self._max_card = max_card

    @property
    def opp_hand_size(self) -> int:
        return self._n

    def sample(self) -> list[int]:
        if self._n == 0:
            return []

        # 隨機模擬牌庫剩餘牌（1~10 均勻）
        deck_value = sum(random.randint(1, 10) for _ in range(self._deck_est))

        # 手牌總值 = 剩餘總資源 − 牌庫估值，至少每張牌值 1
        hand_total = max(float(self._n), self._total - deck_value)

        # 隨機切割成 hand_size 份
        weights = [random.random() for _ in range(self._n)]
        w_sum = sum(weights)
        return [
            max(1, min(self._max_card, round(w / w_sum * hand_total)))
            for w in weights
        ]


class V3Mixin:
    """模擬式手牌採樣 Mixin，供各 V3 Agent 共用。

    繼承順序須將 V3Mixin 置於 V2 Agent 之前：
        class FooAgentV3(V3Mixin, FooAgentV2)
    """

    def _make_dynamic_obs(self, state) -> _V3OppSampler:
        if isinstance(state, _AugState):
            T, I, delta, R = state.T, state.I, state.delta, state.R
        else:
            T, I, delta, R = self._T, self._I, self._delta, self._R

        opp = 1 - self.player_id
        my  = self.player_id
        hand_size = len(state.players[opp].hand)
        max_card  = 10 + R

        deck_est = max(0, 8 - R * 2) if R <= 4 else max(0, 8 - (R - 5) % 5 * 2)
        total_remaining = 55 * self._R_cycle(R) + I[opp] + delta[my] - T[opp]
        total_remaining = max(float(hand_size), total_remaining)

        return _V3OppSampler(total_remaining, deck_est, hand_size, max_card)

    def load(self, path: str) -> None:
        """V3 存檔存在且非空時載入自身；否則 fallback 繼承同族 V2 存檔。"""
        if os.path.exists(path) and os.path.getsize(path) > 0:
            super().load(path)
        else:
            v2_path = path.replace("v3", "v2")
            if os.path.exists(v2_path) and os.path.getsize(v2_path) > 0:
                super().load(v2_path)
