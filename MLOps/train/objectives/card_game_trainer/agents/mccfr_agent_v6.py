"""
⚠️ MCCFRAgentV6 已棄用 - 架構設計失敗

詳見 v6_mixin.py 頂部的完整說明。

【簡述】
V6 嘗試移除硬性 TOP_K 限制，但錯誤地完全破壞了 V5 的 BFS/DFS 二層結構
和加權平均聚合機制，導致性能大幅下降（1000+ 局訓練後仍是最弱版本）。

正確做法是保留 TOP_K，而非移除。
"""
from __future__ import annotations
from agents.mccfr_agent_v5 import MCCFRAgentV5
from agents.utils import regret_match, sample_from
from agents.v6_mixin import V6Mixin


class MCCFRAgentV6(V6Mixin, MCCFRAgentV5):

    def _phase1_hook(self, info_key, actions, phase1_values) -> None:
        """Phase 1 完成後執行後悔值更新（Phase 2 為純價值估計，不再更新後悔值）。"""
        self._update_regrets(info_key, actions, phase1_values)

    def _choose_from_mcts(self, actions, root, info_key) -> list:
        """Phase 2 MCTS 結束，從 Phase 1 累積的平均策略採樣。"""
        total = sum(self._strategy_sum[(info_key, a)] for a in actions)
        if total > 0:
            avg_strategy = {a: self._strategy_sum[(info_key, a)] / total for a in actions}
            return list(sample_from(avg_strategy))
        return list(max(actions, key=lambda a: root.N[a]))
