"""
⚠️ MCTAgentV6 已棄用 - 架構設計失敗

詳見 v6_mixin.py 頂部的完整說明。

【簡述】
V6 嘗試移除硬性 TOP_K 限制，但錯誤地完全破壞了 V5 的 BFS/DFS 二層結構
和加權平均聚合機制，導致性能大幅下降（1000+ 局訓練後仍是最弱版本）。

正確做法是保留 TOP_K，而非移除。
"""
from __future__ import annotations
from agents.mct_agent_v5 import MCTAgentV5
from agents.v6_mixin import V6Mixin


class MCTAgentV6(V6Mixin, MCTAgentV5):
    pass
