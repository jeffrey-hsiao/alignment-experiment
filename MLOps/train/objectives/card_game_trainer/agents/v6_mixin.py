"""
⚠️ V6 已棄用 - 架構設計失敗

【失敗原因】
V6 設計目標是移除硬性 TOP-K 限制，但實現中完全破壞了 V5 的關鍵機制：
  • V5 的 BFS 層（depth < 5）：展開 TOP-K 個動作，進行加權平均返回
  • V6 的錯誤：完全取消 BFS/DFS 二層結構，改為全層統一 DFS
  • 結果：失去了淺層加權平均的聚合機制，搜索效率大幅下降

【性能數據】
訓練 1000+ 局後，V6_1 仍被所有版本壓制：
  - MCTAgentV5_1 vs MCTAgentV6_1：V5 勝率 69%
  - MCCFRAgentV5_1 vs MCCFRAgentV6_1：V5 勝率 79%
  - V6 對舊版本（V2/V3/V4）：勝率 < 10%（幾乎最弱）

【正確的 V6 設計】（未實現）
應該在 BFS 層用「概率採樣」決定參與加權的動作集合，
而非硬性 TOP-K，同時保留加權平均聚合機制。
但經分析，TOP-K 硬性限制實際上已是最優算法。

【結論】
V5 的 BFS/DFS 二層架構 + TOP-K 加權平均才是正確做法。
V6 應被視為失敗品，所有新訓練應基於 V5。

【歷史用途】
代碼保留用於回溯分析，但不在 arena 預設名單中。
如要使用：experiment.py 中仍有映射供手動調用。
"""
from __future__ import annotations
import time
from collections import defaultdict

import torch

from agents.neural_net import ACTION_SIZE
from agents.utils import all_subsets
from agents.v5_mixin import DQNValue, _MCTSNode, V5Mixin, DEVICE, GAMMA, BUFFER_WRITE_LIMIT


class V6Mixin(V5Mixin):
    """
    AlphaGo 風格 MCTS：
      - 移除 BFS_DEPTH / TOP_K 的二層結構
      - Expansion 創建所有合法動作邊
      - PUCT 全程主導選擇
    """

    def _mcts_search(self, node: '_MCTSNode', aug, depth: int = 0) -> float:
        """AlphaGo 風格 MCTS：所有動作都參與，PUCT 自動分配訪問。

        不區分 BFS/DFS，統一用 PUCT 選擇：
          PUCT(s,a) = Q(s,a) + C × P(s,a) × √N(s) / (1 + N(s,a))

        低 prior 的動作自然訪問少（軟性剪枝），
        無需硬性 TOP_K 限制。
        """
        if aug.is_terminal():
            return self._evaluate(aug)

        if time.perf_counter() >= self._search_deadline:
            return self._value_fn(aug)

        my_hand = aug.players[self.player_id].hand
        actions = all_subsets(my_hand)
        opp_id  = 1 - self.player_id
        opp_hand = aug.players[opp_id].hand

        # ── All Moves Expansion：初始化所有合法動作的邊 ──
        for action in actions:
            if action not in node.children:
                # Lazy expansion：子節點在首次訪問時才創建
                # 這裡只是確保邊已初始化
                pass

        # ── PUCT 選擇：綜合考慮 Q 值與 prior 概率 ────────────────
        N_total = sum(node.N[a] for a in actions) + 1
        action = self._mcts_puct_select(node, actions, N_total)

        # ── 模擬對手行動 ────────────────────────────────────────
        if opp_hand:
            opp_action = self._estimate_opp_action(
                opp_hand, all_subsets(opp_hand), aug)
        else:
            opp_action = []

        child_aug = aug.clone()
        if self.player_id == 0:
            child_aug.apply_round(list(action), list(opp_action))
        else:
            child_aug.apply_round(list(opp_action), list(action))

        # ── Lazy Expansion：首次訪問時才創建子節點 ──────────────
        if action not in node.children:
            ch_hf, ch_of = self._encode_state(child_aug)
            node.children[action] = _MCTSNode(
                self._compute_puct_prior(child_aug, ch_hf, ch_of)
            )
            # 新葉節點用 DQN 估值
            value = self._value_fn(child_aug)
        else:
            # 已有子節點，遞迴搜尋
            value = self._mcts_search(node.children[action], child_aug, depth + 1)

        # ── 更新統計量 ───────────────────────────────────────────
        node.N[action] += 1
        node.W[action] += value

        # ── 存入 Replay Buffer ─────────────────────────────────────
        if not self.player_mode and self._writes_this_decision < BUFFER_WRITE_LIMIT:
            my_id      = self.player_id
            opp_id_    = 1 - my_id
            my_max_hp  = aug.players[my_id].character.max_hp
            opp_max_hp = aug.players[opp_id_].character.max_hp
            my_dmg  = (aug.players[my_id].hp  - child_aug.players[my_id].hp)  / my_max_hp
            opp_dmg = (aug.players[opp_id_].hp - child_aug.players[opp_id_].hp) / opp_max_hp
            step_reward = float(opp_dmg - my_dmg)
            done = child_aug.is_terminal()
            if done:
                step_reward += 1.0 if not child_aug.players[opp_id_].is_alive() else -1.0

            mcts_target = step_reward + (0.0 if done else GAMMA * self._value_fn(child_aug))

            cur_hf, cur_of = self._encode_state(aug)
            nxt_hf, nxt_of = self._encode_state(child_aug)
            action_idx = next(
                (i for i, a in enumerate(actions) if a == action), 0)
            self._buffer.append((
                cur_hf, cur_of, action_idx, nxt_hf, nxt_of, mcts_target, 1.0,
            ))
            self._writes_this_decision += 1

        return value

    def choose_action(self, state):
        """AlphaGo 風格決策：Phase 1 遍歷 + Phase 2 AlphaGo 風格 MCTS。

        與 V5 的區別：
          V5：Phase 2 有 BFS/DFS 二層結構，BFS 層硬性 TOP_K
          V6：Phase 2 統一使用 PUCT，無 TOP_K 限制，軟性概率分配
        """
        from agents.v2_mixin import _AugState

        # V2Mixin 責任：記錄 HP 快照、包裝 _AugState
        for pid in (0, 1):
            self._snap_hp[pid] = state.players[pid].hp
        aug = _AugState(state, self._T, self._I, self._delta, self._R)

        hand_feats, other_feats = self._encode_state(aug)
        my_hp_before  = aug.players[self.player_id].hp
        opp_hp_before = aug.players[1 - self.player_id].hp

        self._snapshot_defense(state)

        actions_list = all_subsets(aug.players[self.player_id].hand)
        if len(actions_list) == 1:
            action = list(actions_list[0])
            self._last_played = action
            self._pending = None
            return action

        # 設定計時截止
        t_start = time.perf_counter()
        self._search_deadline    = t_start + self.time_budget
        self._writes_this_decision = 0

        # ── Phase 1：MCT = 全部動作，MCCFR = 全部遍歷 ─────────
        # （不再有 _filtered_actions 的 TOP_K 限制）
        self._filtered_actions = None
        phase1, p1_counts, obs, info_key = self._run_phase1(aug)
        dynamic_obs = self._make_dynamic_obs(aug)
        values_sum, visits, _ = self._merge_ucb(info_key, actions_list, phase1, p1_counts)
        self._phase1_hook(info_key, actions_list, phase1)

        # ── ε-greedy 探索 ──────────────────────────────────────
        import random
        use_epsilon = (
            not self.player_mode
            and random.random() < self._epsilon
        )
        if use_epsilon:
            action = list(random.choice(actions_list))
            self._store_ucb(info_key, actions_list, values_sum, visits)
        else:
            # ── Phase 2：AlphaGo 風格 MCTS（全程 PUCT，無 TOP_K）──
            prior = self._compute_puct_prior(aug, hand_feats, other_feats)
            root  = _MCTSNode(prior)
            # Phase 1 平均值×計數作為根節點初始統計量
            for a in actions_list:
                root.N[a] = p1_counts.get(a, 0)
                root.W[a] = phase1.get(a, 0.0) * root.N[a]

            while time.perf_counter() < self._search_deadline:
                iter_aug = aug.clone()
                iter_aug.players[1 - self.player_id].hand = dynamic_obs.sample()
                self._mcts_search(root, iter_aug)

            # 訪問次數最多的動作為最終決策
            action = self._choose_from_mcts(actions_list, root, info_key)

            # 將 MCTS 累積統計合併跨局歷史後寫回
            for a in actions_list:
                values_sum[a] = self._ucb_values[(info_key, a)] + root.W.get(a, 0.0)
                visits[a]     = self._ucb_visits[(info_key, a)] + root.N.get(a, 0)
            self._store_ucb(info_key, actions_list, values_sum, visits)

        self._last_played = action

        action_tuple = tuple(action)
        action_idx   = next(
            (i for i, a in enumerate(actions_list) if a == action_tuple), 0
        )
        self._pending = {
            'hand'         : hand_feats,
            'other'        : other_feats,
            'action_idx'   : action_idx,
            'my_hp_before' : my_hp_before,
            'opp_hp_before': opp_hp_before,
        }
        return action
