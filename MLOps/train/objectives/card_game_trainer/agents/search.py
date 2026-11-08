"""
預算感知搜尋（三層結構）

1. 精確層：累積複雜度 acc_log + step_cost <= DEPTH_BUDGET → 遍歷所有動作，遞迴
2. 抽樣層：超出預算 → 隨機選 N_ROLLOUT_SAMPLES 個 (my_a, opp_a) 組合進入下一步
3. 估值層：從抽到的 a+1 步節點呼叫 eval_fn 估值，不再遞迴

step_cost_fn(my_n, opp_n) 由各 Agent 提供：
  MCT 與 MCCFR 的 Phase 1 均使用 my_n（外部抽樣：對手動作固定一條路，
  只有我方動作需遍歷，分支複雜度為 2^my_n）

eval_fn：ValueFunction 實例，由 agent 注入；未提供時退回 random_rollout。
  Phase 1 傳入 self._value_fn（RandomRolloutValue 或 DQNValue），
  確保 Phase 1 葉節點估值與 Phase 2 一致。
"""
from __future__ import annotations
import random
from game_engine import GameState
from agents.utils import all_subsets

N_ROLLOUT_SAMPLES = 32  # 超出預算時的固定抽樣數量


def random_rollout(gs: GameState, player_id: int, depth: int) -> float:
    """隨機雙方出牌 depth 步的葉節點估值（預設回退策略）。"""
    for _ in range(depth):
        if gs.is_terminal():
            break
        p0 = random.choice(all_subsets(gs.players[0].hand))
        p1 = random.choice(all_subsets(gs.players[1].hand))
        gs.apply_round(list(p0), list(p1))
    me  = gs.players[player_id]
    opp = gs.players[1 - player_id]
    if not opp.is_alive(): return 1.0
    if not me.is_alive():  return -1.0
    return (me.hp - opp.hp) / me.character.max_hp


def budget_search(
    gs: GameState,
    player_id: int,
    acc_log: float,
    budget: float,
    step_cost_fn,
    rollout_depth: int,
    opp_action_fn=None,
    eval_fn=None,
    node_callback=None,
) -> float:
    """預算感知遞迴搜尋，回傳當前局面對 player_id 的期望價值。

    opp_action_fn: 可選的對手動作估計函式 (opp_hand, opp_actions, gs) -> list。
    eval_fn: ValueFunction 實例，葉節點估值；None 時退回 random_rollout。
    node_callback: 可選的節點訪問回調 (gs, action_values) -> None，用於 CFR 記錄。
    """
    if gs.is_terminal():
        me, opp = gs.players[player_id], gs.players[1 - player_id]
        if not opp.is_alive(): return 1.0
        if not me.is_alive():  return -1.0
        return (me.hp - opp.hp) / me.character.max_hp

    my_hand   = gs.players[player_id].hand
    opp_hand  = gs.players[1 - player_id].hand
    step_cost = step_cost_fn(len(my_hand), len(opp_hand))
    remaining = budget - acc_log

    my_actions  = all_subsets(my_hand)
    opp_actions = all_subsets(opp_hand)

    if step_cost <= remaining:
        # 精確層：遍歷所有動作組合，遞迴繼續
        best_value = -2.0
        for my_a in my_actions:
            total = 0.0
            for opp_a in opp_actions:
                gs2 = gs.clone()
                if player_id == 0:
                    gs2.apply_round(list(my_a), list(opp_a))
                else:
                    gs2.apply_round(list(opp_a), list(my_a))
                total += budget_search(
                    gs2, player_id, acc_log + step_cost,
                    budget, step_cost_fn, rollout_depth, opp_action_fn, eval_fn, node_callback,
                )
            action_value = total / len(opp_actions)
            if action_value > best_value:
                best_value = action_value
        return best_value
    else:
        # 抽樣層：對每個我的動作抽樣對手，以 eval_fn 估值葉節點。
        n_per_action = max(1, N_ROLLOUT_SAMPLES // max(1, len(my_actions)))
        best_value = -2.0
        for my_a in my_actions:
            total = 0.0
            for _ in range(n_per_action):
                gs2 = gs.clone()
                if opp_action_fn is not None:
                    opp_a = opp_action_fn(list(opp_hand), opp_actions, gs2)
                else:
                    opp_a = list(random.choice(opp_actions))
                if player_id == 0:
                    gs2.apply_round(list(my_a), opp_a)
                else:
                    gs2.apply_round(opp_a, list(my_a))
                total += eval_fn(gs2) if eval_fn is not None \
                         else random_rollout(gs2, player_id, rollout_depth)
            action_value = total / n_per_action
            if action_value > best_value:
                best_value = action_value
        return best_value


def all_action_values_budget(
    gs: GameState,
    player_id: int,
    acc_log: float,
    budget: float,
    step_cost_fn,
    rollout_depth: int,
    opp_action_fn=None,
    allowed_my_actions=None,
    eval_fn=None,
    node_callback=None,
) -> tuple[dict, dict]:
    """budget_search 的全動作估值版：回傳 ({action: avg_value}, {action: rollout_count})。

    opp_action_fn      : 同 budget_search，傳入時在 sampling layer 取代均勻隨機抽樣。
    allowed_my_actions : 若指定，只評估此子集的我方動作（用於 top-K 篩選）。
    eval_fn            : ValueFunction 實例，葉節點估值；None 時退回 random_rollout。
    node_callback      : 可選的節點訪問回調 (gs, action_values) -> None，用於 CFR 記錄。
    """
    my_hand   = gs.players[player_id].hand
    opp_hand  = gs.players[1 - player_id].hand
    my_actions  = allowed_my_actions if allowed_my_actions is not None else all_subsets(my_hand)
    opp_actions = all_subsets(opp_hand)
    step_cost   = step_cost_fn(len(my_hand), len(opp_hand))
    remaining   = budget - acc_log

    result = {}
    counts = {}
    if step_cost <= remaining:
        n = len(opp_actions)
        for my_a in my_actions:
            total = 0.0
            for opp_a in opp_actions:
                gs2 = gs.clone()
                if player_id == 0:
                    gs2.apply_round(list(my_a), list(opp_a))
                else:
                    gs2.apply_round(list(opp_a), list(my_a))
                total += budget_search(
                    gs2, player_id, acc_log + step_cost,
                    budget, step_cost_fn, rollout_depth, opp_action_fn, eval_fn, node_callback,
                )
            result[my_a] = total / n
            counts[my_a] = n
    else:
        n_per_action = max(1, N_ROLLOUT_SAMPLES // max(1, len(my_actions)))
        for my_a in my_actions:
            total = 0.0
            for _ in range(n_per_action):
                gs2 = gs.clone()
                if opp_action_fn is not None:
                    opp_a = opp_action_fn(list(opp_hand), opp_actions, gs2)
                else:
                    opp_a = list(random.choice(opp_actions))
                if player_id == 0:
                    gs2.apply_round(list(my_a), opp_a)
                else:
                    gs2.apply_round(opp_a, list(my_a))
                total += eval_fn(gs2) if eval_fn is not None \
                         else random_rollout(gs2, player_id, rollout_depth)
            result[my_a] = total / n_per_action
            counts[my_a] = n_per_action

    if node_callback is not None:
        node_callback(gs, result)

    return result, counts
