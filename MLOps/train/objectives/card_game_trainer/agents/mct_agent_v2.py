"""
MCT Agent V2 — 7 數字緊湊歷史版

7 個緊湊歷史數字封裝進 _AugState，與 GameState 合體後作為搜尋節點的狀態：
搜尋節點操作的是「GameState + 7 數字」的複合狀態，每次 apply_round
都同步更新兩者，7 數字隨搜尋樹傳播而非停留在 agent 實例。

  T[p]     每回合出牌數值之和（累計）
  I[p]     每回合留牌張數（累計，用於追蹤 +1 次數）
  Δ[p]     累計受傷總量
  R        當前回合數

對手手牌＋牌庫總值估算（從節點的 _AugState 讀取）：
  opp_hand_deck ≈ 55 × R_cycle(R) − T[opp] + I[opp] + Δ[my]
"""
from __future__ import annotations
import pickle
from collections import defaultdict
from agents.mct_agent import MCTAgent
from agents.v2_mixin import V2Mixin, _AugState  # noqa: F401  (_AugState re-exported)


class MCTAgentV2(V2Mixin, MCTAgent):

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                **self._seven_state_dict(),
                "ucb_values": dict(self._ucb_values),
                "ucb_visits": dict(self._ucb_visits),
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._load_seven_state(data)
        self._ucb_values = defaultdict(float, data.get("ucb_values", {}))
        self._ucb_visits = defaultdict(int,   data.get("ucb_visits", {}))
