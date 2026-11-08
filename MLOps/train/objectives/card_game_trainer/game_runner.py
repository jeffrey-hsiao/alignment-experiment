"""
MLOps/train/objectives/card_game_trainer/game_runner.py

單局遊戲迴圈，從博弈論/run_llm_v1_training.py 的 run_game() 抽出（該函式
與 V1/V2/V3 訓練完全共用，本身不含任何 CLI/log 相關邏輯，純粹是對局迴圈：
兩個 agent 輪流 choose_action。LLM agent 的訓練（_train_on_valid_action／
_train_on_game_end／_batch_train_from_replay）已經內建在
agents/llm_agent.py 的 choose_action()/notify_result() 裡，這裡不需要、
也不應該額外呼叫任何訓練方法——直接呼叫 agent 介面即可讓訓練自然發生。

agent_a/agent_b 只要求實作 BaseAgent 介面（choose_action 必要，
new_game/notify_result 可選），因此 LLM agent 與傳統 MCCFR/MCT agent 都能
互換使用，同一份迴圈可以同時服務 self-play 訓練與未來的競賽對戰。
"""
from __future__ import annotations

from game_engine import GameState

# game_engine.GameState.is_terminal() 只看 HP<=0，沒有任何回合數上限。
# 博弈論原專案的 debug_tools/investigate_long_game_stall.py 記錄過一次真實
# 事故：300 局訓練跑到 game 211 時，單局耗費 4 小時 35 分鐘、2463 個動作
# （正常一局約 4-15 動作）才結束——雙方出牌總和連續多輪剛好卡在防禦內、
# 誰都打不穿對方的病態僵局，理論上可能無限跑下去。該工具自己用 200 回合當
# 安全上限重現這個異常，但正式的 run_game() 從未真的補上——這裡沿用同一個
# 上限，超過就視為平局結束，避免單局卡死拖垮整個訓練/競技場流程。
MAX_ROUNDS = 200


def run_game(agent_a, agent_b, seed=None, on_round=None, max_rounds=MAX_ROUNDS) -> int:
    """
    運行單局遊戲，返回獲勝者 (0=agent_a, 1=agent_b, -1=平局或撞到 max_rounds)

    on_round: 選填 callback(round_num, played_a, played_b, state)，每回合
    apply_round 後呼叫一次。單局可能跨多個回合，每回合都要等 agent 各自
    choose_action()（LLM agent 是一次 generate() 呼叫，可能要幾秒到十幾秒），
    沒有這個 callback 的話呼叫端在一整局結束前完全看不到任何進度。
    """
    gs = GameState(seed=seed)

    for agent in (agent_a, agent_b):
        if hasattr(agent, "new_game"):
            agent.new_game()

    while not gs.is_terminal() and gs.round_num < max_rounds:
        a0 = agent_a.choose_action(gs)
        a1 = agent_b.choose_action(gs)

        gs.apply_round(a0, a1)

        if hasattr(agent_a, "notify_result"):
            agent_a.notify_result(gs)
        if hasattr(agent_b, "notify_result"):
            agent_b.notify_result(gs)

        if on_round is not None:
            on_round(gs.round_num, a0, a1, gs)

    if not gs.is_terminal():
        return -1  # 撞到 max_rounds 仍未分出勝負：視為平局，強制結束
    return gs.winner() if gs.winner() is not None else -1
