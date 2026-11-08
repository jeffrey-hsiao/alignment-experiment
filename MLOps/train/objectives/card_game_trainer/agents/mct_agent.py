"""
MCT Agent — Version B (history-aware opponent model)

對手手牌標記為 UNKNOWN。不直接抽樣對手手牌，
而是從歷史回合結果推算對手的慣用出牌總和，
用這個估計來引導模擬時的對手行動選擇。
輸出：UCB 最高的純策略動作。
"""
from __future__ import annotations
import pickle
import random
import time
from collections import defaultdict
from agents.partial_info_agent import PartialInfoAgent
from agents.utils import all_subsets
from game_engine import GameState


class MCTAgent(PartialInfoAgent):
    # 手牌數超過此門檻時改用 UCB，讓 rollout 預算自適應分配
    # 5 張以下：best_action_budget 每個動作仍可分到足夠 rollout
    # 6 張以上：動作數 ≥ 64，均勻分配開始稀薄，UCB 更有效率
    _UCB_THRESHOLD: float = 5.0

    def __init__(self, player_id: int, time_budget: float = 5.0, rollout_depth: int = 4):
        super().__init__(player_id)
        self.time_budget = time_budget
        self.rollout_depth = rollout_depth

    def _step_cost(self, my_n: int, opp_n: int) -> float:
        return float(my_n)  # 2^n 成長：只計我方動作數

    # ── 主決策 ──────────────────────────────────────────

    def choose_action(self, state: GameState) -> list[int]:
        self._snapshot_defense(state)
        hand = state.players[self.player_id].hand
        actions = getattr(self, '_filtered_actions', None) or all_subsets(hand)

        if len(actions) == 1:
            self._last_played = list(actions[0])
            return list(actions[0])

        # ── Phase 1：計次制遍歷（DEPTH_BUDGET 控制深度）────────
        phase1, p1_counts, obs, info_key = self._run_phase1(state)
        step_cost = self._step_cost(len(hand), obs.opp_hand_size)
        dynamic_obs = self._make_dynamic_obs(state)

        # UCB 統計：Phase 1 結果與跨局累積值合併
        values_sum, visits, total_n = self._merge_ucb(info_key, actions, phase1, p1_counts)

        # ── Phase 2：計時制純 rollout（UCB 或輪流）──────────────
        p2_actions = self._get_phase2_actions(state, actions)
        self.last_iter_count = 0
        t_start = time.perf_counter()

        if step_cost <= self._UCB_THRESHOLD:
            # 動作數少：輪流 rollout，無需 UCB
            idx = 0
            while time.perf_counter() - t_start < self.time_budget:
                action = p2_actions[idx % len(p2_actions)]
                values_sum[action] += self._simulate(state, list(action), dynamic_obs)
                visits[action] += 1
                idx += 1
                self.last_iter_count += 1
            result = list(max(p2_actions, key=lambda a: values_sum[a] / visits[a]))
        else:
            # 動作數多：UCB 引導 rollout
            while time.perf_counter() - t_start < self.time_budget:
                action = self._ucb_select(p2_actions, visits, values_sum, total_n)
                values_sum[action] += self._simulate(state, list(action), dynamic_obs)
                visits[action] += 1
                total_n += 1
                self.last_iter_count += 1
            result = list(max(p2_actions, key=lambda a: visits[a]))

        # 將本次決策統計存回跨局記憶
        self._store_ucb(info_key, actions, values_sum, visits)

        self._last_played = result
        return result

    # ── 存檔 / 讀檔 ──────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "opp_total_history": self._opp_total_history,
                "last_played": self._last_played,
                "ucb_values": dict(self._ucb_values),
                "ucb_visits": dict(self._ucb_visits),
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._opp_total_history = state["opp_total_history"]
        self._last_played = state.get("last_played", [])
        self._ucb_values = defaultdict(float, state.get("ucb_values", {}))
        self._ucb_visits = defaultdict(int,  state.get("ucb_visits", {}))
