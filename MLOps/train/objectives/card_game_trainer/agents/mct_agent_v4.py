"""
MCT Agent V4 — 7 數字進入跨局學習鍵值

在 MCTAgentV3 基礎上透過 V4Mixin 將 7 個數字的分桶值附加至 info_key，
使 UCB 統計能區分對手歷史背景不同的相同可觀測局面。
"""
from __future__ import annotations
from agents.mct_agent_v3 import MCTAgentV3
from agents.v4_mixin import V4Mixin


class MCTAgentV4(V4Mixin, MCTAgentV3):
    pass
