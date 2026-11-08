"""
MCCFR Agent V5 — DQN 估值函數 + MCCFR 後悔值匹配

在 MCCFRAgentV4 基礎上透過 V5Mixin 引入 DQN，
以 max_a Q(s,a) 取代 Phase 1 的葉節點估值。

Phase 2 為純粹的價值估計（基於 MCTS 樹搜尋），不執行後悔值更新。
決策基於 Phase 1 的後悔值匹配平均策略。
"""
from __future__ import annotations
from agents.mccfr_agent_v4 import MCCFRAgentV4
from agents.utils import regret_match, sample_from
from agents.v5_mixin import V5Mixin


class MCCFRAgentV5(V5Mixin, MCCFRAgentV4):

    def _phase1_hook(self, info_key, actions, phase1_values) -> None:
        """Phase 1 完成後執行後悔值更新（Phase 2 為純價值估計，不再更新後悔值）。"""
        self._update_regrets(info_key, actions, phase1_values)

    def _choose_from_mcts(self, actions, root, info_key) -> list:
        """Phase 2 MCTS 結束，從 Phase 1 累積的平均策略采样。"""
        total = sum(self._strategy_sum[(info_key, a)] for a in actions)
        if total > 0:
            avg_strategy = {a: self._strategy_sum[(info_key, a)] / total for a in actions}
            return list(sample_from(avg_strategy))
        return list(max(actions, key=lambda a: root.N[a]))
