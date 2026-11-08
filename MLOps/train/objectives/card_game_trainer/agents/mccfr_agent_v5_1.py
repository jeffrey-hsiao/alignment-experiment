"""
MCCFR Agent V5_1 — 多樣對手訓練版本

架構與 MCCFRAgentV5 完全相同（V5Mixin + MCCFRAgentV4）。
差異在訓練環境：由 arena.py 以多樣對手取代自我對戰，
不使用 soft_update_from，避免策略在自身空間內震盪收斂。
"""
from __future__ import annotations
from agents.mccfr_agent_v4 import MCCFRAgentV4
from agents.v5_mixin import V5Mixin


class MCCFRAgentV5_1(V5Mixin, MCCFRAgentV4):
    pass
