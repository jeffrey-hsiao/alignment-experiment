"""
Online MCCFR Agent — External Sampling

每次決策時，從當前局面出發進行 N 次 MCCFR 迭代：
  - 每次迭代抽樣一組對手可能的手牌（determinization）
  - 依對手歷史模型抽樣一個對手動作（外部抽樣：對手只走一條路）
  - 對我的每個動作各做一次 rollout，得到反事實價值
  - 以 (info_state, action) 為鍵更新後悔值（跨輪持久化）
  - 後悔匹配更新策略累加

輸出：從累積的平均策略分佈中取樣的動作（混合策略）。
"""
from __future__ import annotations
import pickle
import random
import time
from collections import defaultdict
from agents.partial_info_agent import PartialInfoAgent
from agents.utils import all_subsets, regret_match, sample_from
from game_engine import GameState


class MCCFRAgent(PartialInfoAgent):
    def __init__(self, player_id: int, time_budget: float = 5.0, rollout_depth: int = 3):
        super().__init__(player_id)
        self.time_budget = time_budget
        self.rollout_depth = rollout_depth
        # Keys: (info_state_tuple, action_tuple)
        self._regrets: dict[tuple, float] = defaultdict(float)
        self._strategy_sum: dict[tuple, float] = defaultdict(float)
        # Phase 1 遞迴搜尋過程中，記錄訪問過的節點（供後悔值更新使用）
        self._phase1_nodes: list = []

    # ── Phase 1 覆寫（加入節點記錄用於後悔值更新）──

    def _run_phase1(self, state):
        """Phase 1 遞迴遍歷，記錄所有訪問的節點供後悔值更新使用。"""
        from agents.search import all_action_values_budget
        from agents.v2_mixin import _AugState

        # 清空本次決策的節點記錄
        self._phase1_nodes = []

        obs = state.get_observation(self.player_id)
        opp_hand = self._make_dynamic_obs(state).sample()
        gs_sim = state.clone()
        gs_sim.players[1 - self.player_id].hand = opp_hand

        # 定義節點訪問回調，記錄每個訪問的節點
        def node_callback(gs, action_values):
            # 根據當前遊戲狀態構造 info_key
            if isinstance(gs, _AugState):
                obs_node = gs.get_observation(self.player_id)
            else:
                obs_node = gs.get_observation(self.player_id)
            node_info_key = (
                tuple(sorted(obs_node.my_hand)),
                obs_node.my_hp, obs_node.my_defense,
                obs_node.opp_hp, obs_node.opp_defense,
                obs_node.opp_hand_size,
            )
            self._phase1_nodes.append((node_info_key, action_values))

        action_values, counts = all_action_values_budget(
            gs_sim, self.player_id,
            acc_log=0.0, budget=self.DEPTH_BUDGET,
            step_cost_fn=lambda my_n, opp_n: float(my_n),
            rollout_depth=self.rollout_depth,
            allowed_my_actions=getattr(self, '_filtered_actions', None),
            eval_fn=self._value_fn,
            node_callback=node_callback,
        )
        info_key = (
            tuple(sorted(obs.my_hand)),
            obs.my_hp, obs.my_defense,
            obs.opp_hp, obs.opp_defense,
            obs.opp_hand_size,
        )
        return action_values, counts, obs, info_key

    # ── 主決策 ──────────────────────────────────────────

    def choose_action(self, state: GameState) -> list[int]:
        self._snapshot_defense(state)
        hand = state.players[self.player_id].hand
        actions = all_subsets(hand)

        if len(actions) == 1:
            self._last_played = list(actions[0])
            return list(actions[0])

        # ── Phase 1：遞迴遍歷 + regret update（對所有訪問的節點）──────────
        action_values, p1_counts, obs, info_key = self._run_phase1(state)
        # 對 Phase 1 搜尋樹中訪問過的所有節點執行後悔值更新
        for node_info_key, node_action_values in self._phase1_nodes:
            # 獲取該節點的所有可能動作（由action_values的鍵決定）
            node_actions = list(node_action_values.keys())
            self._update_regrets(node_info_key, node_actions, node_action_values)
        self.last_iter_count = sum(p1_counts.values())

        # UCB 統計：Phase 1 結果與跨局累積值合併
        values_sum, visits, total_n = self._merge_ucb(info_key, actions, action_values, p1_counts)

        # 動態採樣物件：每次 _simulate 都從不同假設採樣對手手牌
        dynamic_obs = self._make_dynamic_obs(state)

        # 以 Phase 1 後的後悔值計算當前策略，供 Phase 2 每次迭代累積
        current_strategy = regret_match({a: self._regrets[(info_key, a)] for a in actions}, actions)

        # ── Phase 2：UCT-like rollout（UCB 選動作 + dynamic_obs 採樣）──
        p2_actions = self._get_phase2_actions(state, actions)
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < self.time_budget:
            action = self._ucb_select(p2_actions, visits, values_sum, total_n)
            v = self._simulate(state, list(action), dynamic_obs)
            values_sum[action] += v
            visits[action]     += 1
            total_n            += 1
            for a in actions:
                self._strategy_sum[(info_key, a)] += current_strategy[a]

        # 存回跨局記憶
        self._store_ucb(info_key, actions, values_sum, visits)

        # 輸出：regret-matched 平均策略
        total = sum(self._strategy_sum[(info_key, a)] for a in actions)
        if total > 0:
            avg_strategy = {a: self._strategy_sum[(info_key, a)] / total for a in actions}
        else:
            n = len(actions)
            avg_strategy = {a: 1.0 / n for a in actions}

        chosen = sample_from(avg_strategy)
        self._last_played = list(chosen)
        return list(chosen)

    def _update_regrets(self, info_key, actions, action_values) -> None:
        keyed_regrets = {a: self._regrets[(info_key, a)] for a in actions}
        strategy = regret_match(keyed_regrets, actions)
        current_value = sum(strategy[a] * action_values[a] for a in actions)
        for a in actions:
            self._regrets[(info_key, a)] += action_values[a] - current_value
            self._strategy_sum[(info_key, a)] += strategy[a]

    # ── 存檔 / 讀檔 ──────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "regrets": dict(self._regrets),
                "strategy_sum": dict(self._strategy_sum),
                "ucb_values": dict(self._ucb_values),
                "ucb_visits": dict(self._ucb_visits),
                "last_played": self._last_played,
                "opp_total_history": self._opp_total_history,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._regrets = defaultdict(float, data["regrets"])
        self._strategy_sum = defaultdict(float, data["strategy_sum"])
        self._ucb_values = defaultdict(float, data.get("ucb_values", {}))
        self._ucb_visits = defaultdict(int,   data.get("ucb_visits", {}))
        self._last_played = data.get("last_played", [])
        self._opp_total_history = data.get("opp_total_history", [])
