"""
估值函數基類與通用實作

ValueFunction：局面 → 標量的可呼叫介面
  子類只需實作 __call__(gs) -> float，可直接作為 eval_fn 注入
  budget_search / _simulate 等搜尋函數使用。

RandomRolloutValue：隨機雙方出牌 depth 步後以 HP 差估值（V1~V4 預設）
DQNValue 等高階版本定義在各版本 Mixin 內（依賴 torch 等外部套件）。
"""
from __future__ import annotations
import random
from agents.utils import all_subsets


class ValueFunction:
    """局面估值函數介面：__call__(gs) -> float"""
    def __call__(self, gs) -> float:
        raise NotImplementedError


class RandomRolloutValue(ValueFunction):
    """隨機雙方出牌 depth 步後以 HP 差估值。"""

    def __init__(self, player_id: int, depth: int):
        self._player_id = player_id
        self._depth = depth

    def __call__(self, gs) -> float:
        for _ in range(self._depth):
            if gs.is_terminal():
                break
            p0 = random.choice(all_subsets(gs.players[0].hand))
            p1 = random.choice(all_subsets(gs.players[1].hand))
            gs.apply_round(list(p0), list(p1))
        me  = gs.players[self._player_id]
        opp = gs.players[1 - self._player_id]
        if not opp.is_alive(): return 1.0
        if not me.is_alive():  return -1.0
        return (me.hp - opp.hp) / me.character.max_hp
