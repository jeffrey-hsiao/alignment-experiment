"""
MCT Agent V5 — DQN Rollout + PUCT 引導搜尋

在 MCTAgentV4 基礎上透過 V5Mixin 引入 DQN：
  1. _rollout 替換為 max_a Q(s,a) 估算局面值
  2. Phase 2 UCB 改用 PUCT 公式，softmax(Q) 作為 prior 引導搜尋
  3. Phase 1 只評估 top-K（DQN Q 值最高）動作，Phase 2 搜尋全部動作
"""
from __future__ import annotations
from agents.mct_agent_v4 import MCTAgentV4
from agents.v5_mixin import V5Mixin


class MCTAgentV5(V5Mixin, MCTAgentV4):

    def _get_phase1_actions(self, actions_list, hand_feats, other_feats):
        """Phase 1 只評估 DQN Q 值前 TOP_K 個動作。"""
        return self._top_k_actions(actions_list, hand_feats, other_feats, self.TOP_K)
