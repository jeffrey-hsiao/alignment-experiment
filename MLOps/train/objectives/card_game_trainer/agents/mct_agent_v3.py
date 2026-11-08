"""
MCT Agent V3 — 修正手牌採樣的 V2

與 MCTAgentV2 唯一差異：透過 V3Mixin 覆寫 _make_dynamic_obs，
以修正後的利息分離公式與 Gaussian 隨機分割生成對手手牌。
"""
from __future__ import annotations
from agents.mct_agent_v2 import MCTAgentV2
from agents.v3_mixin import V3Mixin


class MCTAgentV3(V3Mixin, MCTAgentV2):
    pass
