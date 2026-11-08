"""
MCCFR Agent V3 — 修正手牌採樣的 MCCFR V2

與 MCCFRAgentV2 唯一差異：透過 V3Mixin 覆寫 _make_dynamic_obs，
以修正後的利息分離公式與 Gaussian 隨機分割生成對手手牌。
"""
from __future__ import annotations
from agents.mccfr_agent_v2 import MCCFRAgentV2
from agents.v3_mixin import V3Mixin


class MCCFRAgentV3(V3Mixin, MCCFRAgentV2):
    pass
