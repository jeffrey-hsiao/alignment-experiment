"""
MCCFR Agent V4 — 7 數字進入跨局學習鍵值

在 MCCFRAgentV3 基礎上透過 V4Mixin 將 7 個數字的分桶值附加至 info_key，
使後悔值與策略累積能區分對手歷史背景不同的相同可觀測局面。
"""
from __future__ import annotations
from agents.mccfr_agent_v3 import MCCFRAgentV3
from agents.v4_mixin import V4Mixin


class MCCFRAgentV4(V4Mixin, MCCFRAgentV3):
    pass
