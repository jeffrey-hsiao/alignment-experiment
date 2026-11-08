"""
MCT Agent V5_1 — 多樣對手訓練版本

架構與 MCTAgentV5 完全相同（V5Mixin + MCTAgentV4）。
差異在訓練環境：由 arena.py 以多樣對手取代自我對戰，
不使用 soft_update_from，避免策略在自身空間內震盪收斂。
"""
from __future__ import annotations
from agents.mct_agent_v4 import MCTAgentV4
from agents.v5_mixin import V5Mixin


class MCTAgentV5_1(V5Mixin, MCTAgentV4):

    def _get_phase1_actions(self, actions_list, hand_feats, other_feats):
        """Phase 1 只評估 DQN Q 值前 TOP_K 個動作。"""
        return self._top_k_actions(actions_list, hand_feats, other_feats, self.TOP_K)
