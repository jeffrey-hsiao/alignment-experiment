"""
MCCFR Agent V2 — 7 數字緊湊歷史版

與 MCTAgentV2 概念相同，在 MCCFRAgent 基礎上透過 V2Mixin 注入
7 數字狀態，取代 _opp_total_history。
"""
from __future__ import annotations
import pickle
from collections import defaultdict
from agents.mccfr_agent import MCCFRAgent
from agents.v2_mixin import V2Mixin


class MCCFRAgentV2(V2Mixin, MCCFRAgent):

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                **self._seven_state_dict(),
                "regrets":      dict(self._regrets),
                "strategy_sum": dict(self._strategy_sum),
                "ucb_values":   dict(self._ucb_values),
                "ucb_visits":   dict(self._ucb_visits),
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._load_seven_state(data)
        self._regrets      = defaultdict(float, data["regrets"])
        self._strategy_sum = defaultdict(float, data["strategy_sum"])
        self._ucb_values   = defaultdict(float, data.get("ucb_values", {}))
        self._ucb_visits   = defaultdict(int,   data.get("ucb_visits", {}))
