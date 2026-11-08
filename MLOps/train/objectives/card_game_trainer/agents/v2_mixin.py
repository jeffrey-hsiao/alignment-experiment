"""
V2 共用元件：_AugState 與 V2Mixin

_AugState : GameState + 7 個緊湊歷史數字的複合節點，duck-types GameState。
V2Mixin   : 將 7 數字狀態注入任意 PartialInfoAgent 子類的 Mixin。
            使用方式：class FooAgentV2(V2Mixin, FooAgent)
            MRO 保證 choose_action / notify_result 由 Mixin 攔截，
            再透過 super() 轉發給真正的 Agent 實作。
"""
from __future__ import annotations
import math
import random
from game_engine import GameState


# ── _AugState ─────────────────────────────────────────────────────────────

class _AugState:
    """GameState + 7 compact history numbers，作為 V2/V3 搜尋節點的狀態。

    Duck-types GameState：search.py 的所有函式可直接接受此物件。
    apply_round() 同步更新底層 GameState 與 7 個數字。
    clone() 兩者皆複製，確保 7 數字隨搜尋樹傳播。
    """

    __slots__ = ("_gs", "T", "I", "delta", "R")

    def __init__(self, gs: GameState, T: list, I: list, delta: list, R: int):
        self._gs = gs
        self.T     = T[:]
        self.I     = I[:]
        self.delta = delta[:]
        self.R     = R

    @property
    def players(self):
        return self._gs.players

    @property
    def history(self):
        return self._gs.history

    @property
    def round_num(self):
        return self._gs.round_num

    def is_terminal(self) -> bool:
        return self._gs.is_terminal()

    def get_observation(self, pid: int):
        return self._gs.get_observation(pid)

    def estimate_opp_hand(self, pid: int):
        return self._gs.estimate_opp_hand(pid)

    def clone(self) -> "_AugState":
        return _AugState(self._gs.clone(), self.T, self.I, self.delta, self.R)

    def apply_round(self, p0_action: list, p1_action: list):
        snap_hp = [self._gs.players[0].hp, self._gs.players[1].hp]
        self._gs.apply_round(p0_action, p1_action)
        if not self._gs.history:
            return
        r = self._gs.history[-1]

        self.T[0] += sum(r.p0_played)
        self.T[1] += sum(r.p1_played)

        self.I[0] += max(0, r.p0_hand_size_after - 2 - (1 if r.winner == 0 else 0))
        self.I[1] += max(0, r.p1_hand_size_after - 2 - (1 if r.winner == 1 else 0))

        self.delta[0] += max(0, snap_hp[0] - r.p0_hp_after)
        self.delta[1] += max(0, snap_hp[1] - r.p1_hp_after)

        self.R = self._gs.round_num


# ── V2Mixin ───────────────────────────────────────────────────────────────

class V2Mixin:
    """7 數字狀態 Mixin，供各 V2/V3 Agent 共用。

    繼承順序須將 V2Mixin 置於真正的 Agent 之前：
        class FooAgentV2(V2Mixin, FooAgent)
    """

    def __init__(self, player_id: int, *args, **kwargs):
        super().__init__(player_id, *args, **kwargs)
        self._opp_total_history = None  # v2 不使用歷史序列

        self._T:      list[float] = [0.0, 0.0]
        self._I:      list[int]   = [0, 0]
        self._delta:  list[float] = [0.0, 0.0]
        self._R:      int         = 0
        self._snap_hp: list[int]  = [0, 0]

    def new_game(self) -> None:
        """每局開始時重置局內累積狀態（7 數字為 per-game 狀態，不跨局）。"""
        self._T     = [0.0, 0.0]
        self._I     = [0, 0]
        self._delta = [0.0, 0.0]
        self._R     = 0
        self._snap_hp = [0, 0]

    def choose_action(self, state):
        for pid in (0, 1):
            self._snap_hp[pid] = state.players[pid].hp
        aug = _AugState(state, self._T, self._I, self._delta, self._R)
        return super().choose_action(aug)

    @staticmethod
    def _R_cycle(R: int) -> float:
        if R <= 4:
            return 1.0
        return 1.0 + math.ceil((R - 4) / 5)

    def _estimate_opp_action(self, opp_hand: list, opp_actions: list, state=None) -> list:
        if not opp_hand:
            return []
        if random.random() < 0.2:
            return list(random.choice(opp_actions))

        if isinstance(state, _AugState):
            T, I, delta, R = state.T, state.I, state.delta, state.R
        else:
            T, I, delta, R = self._T, self._I, self._delta, self._R

        opp = 1 - self.player_id
        my  = self.player_id
        opp_hd   = 55 * self._R_cycle(R) - T[opp] + I[opp] + delta[my]
        opp_sz   = len(opp_hand)
        deck_est = max(0, 8 - R * 2) if R <= 4 else max(0, 8 - (R - 5) % 5 * 2)
        hand_est = opp_hd * opp_sz / max(1, opp_sz + deck_est)
        return list(min(opp_actions, key=lambda a: abs(sum(a) - hand_est * 0.5)))

    def notify_result(self, state) -> None:
        if not state.history:
            return
        result = state.history[-1]
        my  = self.player_id
        opp = 1 - my

        self._R = state.round_num

        my_sum = sum(result.p0_played if my == 0 else result.p1_played)
        opp_sum = self._infer_opp_sum(state, my_sum, result)
        self._T[my]  += my_sum
        self._T[opp] += opp_sum

        my_size_after  = result.p0_hand_size_after if my == 0 else result.p1_hand_size_after
        opp_size_after = result.p1_hand_size_after if my == 0 else result.p0_hand_size_after
        self._I[my]  += max(0, my_size_after  - 2 - (1 if result.winner == my  else 0))
        self._I[opp] += max(0, opp_size_after - 2 - (1 if result.winner == opp else 0))

        my_hp_after  = result.p0_hp_after if my == 0 else result.p1_hp_after
        opp_hp_after = result.p1_hp_after if my == 0 else result.p0_hp_after
        self._delta[my]  += max(0, self._snap_hp[my]  - my_hp_after)
        self._delta[opp] += max(0, self._snap_hp[opp] - opp_hp_after)

        self._last_played = result.p0_played if my == 0 else result.p1_played

    # ── save/load 輔助（供子類別組合使用）────────────────

    def _seven_state_dict(self) -> dict:
        return {
            "T": self._T, "I": self._I,
            "delta": self._delta, "R": self._R,
            "last_played": self._last_played,
        }

    def _load_seven_state(self, data: dict) -> None:
        self._T           = data.get("T",           [0.0, 0.0])
        self._I           = data.get("I",           [0, 0])
        self._delta       = data.get("delta",       [0.0, 0.0])
        self._R           = data.get("R",           0)
        self._last_played = data.get("last_played", [])
