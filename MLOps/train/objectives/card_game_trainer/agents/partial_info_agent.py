"""
PartialInfoAgent — 不完全資訊 Agent 基類

對手手牌估算委託給 GameState.estimate_opp_hand()，
確保 agent 僅使用可觀測資訊，不讀取對手私有狀態。
"""
from __future__ import annotations
import math
import random
from collections import defaultdict
from agents.base_agent import BaseAgent
from agents.utils import all_subsets
from agents.value_function import RandomRolloutValue
from game_engine import GameState, DynamicObs


class PartialInfoAgent(BaseAgent):

    DEPTH_BUDGET: float = 4.0  # log2(16)

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self._ucb_values: dict[tuple, float] = defaultdict(float)
        self._ucb_visits: dict[tuple, int]   = defaultdict(int)
        self._opp_total_history: list[float] = []
        self._last_played: list[int] = []
        self._prev_my_def: int = 0
        self._prev_opp_def: int = 0

    def _snapshot_defense(self, state: GameState) -> None:
        """在 choose_action 開頭呼叫，記錄回合前雙方護甲值。"""
        self._prev_my_def  = state.players[self.player_id].defense
        self._prev_opp_def = state.players[1 - self.player_id].defense

    def _estimate_opp_action(self, opp_hand: list[int], opp_actions: list, state=None) -> list[int]:
        if not self._opp_total_history or random.random() < 0.2:
            return list(random.choice(opp_actions))
        recent = self._opp_total_history[-5:]
        target = sum(recent) / len(recent)
        return list(min(opp_actions, key=lambda a: abs(sum(a) - target)))

    def _infer_opp_sum(self, state, my_sum: float, result) -> float:
        """從可觀測資訊推算對手本回合出牌總和。"""
        my  = self.player_id
        opp = 1 - my
        if result.winner == my:
            return max(0.0, my_sum - result.difference)
        elif result.winner == opp:
            return float(my_sum + result.difference)
        else:
            # 無勝者：比較實際護甲與「僅回復」的預期值，判斷哪方吸收了攻擊
            me_p  = state.players[my]
            opp_p = state.players[opp]
            expected_my  = min(self._prev_my_def  + 1, me_p.character.base_defense)
            expected_opp = min(self._prev_opp_def + 1, opp_p.character.base_defense)
            if me_p.defense < expected_my:
                return float(my_sum + result.difference)
            elif opp_p.defense < expected_opp:
                return max(0.0, float(my_sum - result.difference))
            else:
                return float(my_sum)

    def notify_result(self, state: GameState):
        if not state.history:
            return
        result = state.history[-1]
        my_total = sum(self._last_played)
        opp_total = self._infer_opp_sum(state, my_total, result)
        self._opp_total_history.append(opp_total)
        if len(self._opp_total_history) > 10:
            self._opp_total_history.pop(0)

    def _make_dynamic_obs(self, state: GameState) -> DynamicObs:
        """建立動態採樣物件，每次 sample() 產生不同但合理的對手手牌。"""
        return DynamicObs(state.get_observation(self.player_id))

    def _merge_ucb(self, info_key, actions, phase1_values, p1_counts):
        """Phase 1 結果與跨局 UCB 統計合併，回傳 (values_sum, visits, total_n)。"""
        values_sum = {a: phase1_values.get(a, 0.0) * p1_counts.get(a, 0) + self._ucb_values[(info_key, a)]
                      for a in actions}
        visits     = {a: p1_counts.get(a, 0)                             + self._ucb_visits[(info_key, a)]
                      for a in actions}
        return values_sum, visits, sum(visits.values())

    def _store_ucb(self, info_key, actions, values_sum, visits):
        """將本次決策統計存回跨局記憶。"""
        for a in actions:
            self._ucb_values[(info_key, a)] = values_sum[a]
            self._ucb_visits[(info_key, a)] = visits[a]

    def _ucb_select(self, actions, visits, values, N):
        log_N = math.log(N)
        best, best_score = None, -math.inf
        for a in actions:
            exploit = values[a] / visits[a]
            explore = math.sqrt(2) * math.sqrt(log_N / visits[a])
            score = exploit + explore
            if score > best_score:
                best_score, best = score, a
        return best

    def _run_phase1(self, state) -> tuple:
        """Phase 1 遞迴遍歷（MCT / MCCFR 共用）。
        回傳 (action_values, counts, obs, info_key)。
        """
        from agents.search import all_action_values_budget
        obs = state.get_observation(self.player_id)
        opp_hand = self._make_dynamic_obs(state).sample()
        gs_sim = state.clone()
        gs_sim.players[1 - self.player_id].hand = opp_hand
        action_values, counts = all_action_values_budget(
            gs_sim, self.player_id,
            acc_log=0.0, budget=self.DEPTH_BUDGET,
            step_cost_fn=lambda my_n, opp_n: float(my_n),
            rollout_depth=self.rollout_depth,
            allowed_my_actions=getattr(self, '_filtered_actions', None),
            eval_fn=self._value_fn,
        )
        info_key = (
            tuple(sorted(obs.my_hand)),
            obs.my_hp, obs.my_defense,
            obs.opp_hp, obs.opp_defense,
            obs.opp_hand_size,
        )
        return action_values, counts, obs, info_key

    def _get_phase2_actions(self, state, actions: list) -> list:
        """Phase 2 搜尋範圍鉤子，子類可覆寫以縮小動作集（例如 top-K）。"""
        return actions

    # ── 估值函數（Strategy Pattern）────────────────────────────

    @property
    def _value_fn(self):
        """各版本共用的葉節點估值函數。
        V1~V4：懶惰初始化為 RandomRolloutValue（需 rollout_depth 已設定）。
        V5：V5Mixin.__init__ 在 super().__init__() 後覆寫為 DQNValue。
        """
        if not hasattr(self, '_vf_cache'):
            self._vf_cache = RandomRolloutValue(
                self.player_id, getattr(self, 'rollout_depth', 4)
            )
        return self._vf_cache

    @_value_fn.setter
    def _value_fn(self, vf):
        self._vf_cache = vf

    def _evaluate(self, gs) -> float:
        me  = gs.players[self.player_id]
        opp = gs.players[1 - self.player_id]
        if not opp.is_alive(): return 1.0
        if not me.is_alive():  return -1.0
        return (me.hp - opp.hp) / me.character.max_hp

    def _simulate(self, state, my_action: list[int], obs) -> float:
        gs = state.clone()
        opp_hand = obs.sample()
        gs.players[1 - self.player_id].hand = opp_hand
        opp_actions = all_subsets(opp_hand)
        opp_action = self._estimate_opp_action(opp_hand, opp_actions, gs)
        if self.player_id == 0:
            gs.apply_round(my_action, opp_action)
        else:
            gs.apply_round(opp_action, my_action)
        return self._value_fn(gs)
