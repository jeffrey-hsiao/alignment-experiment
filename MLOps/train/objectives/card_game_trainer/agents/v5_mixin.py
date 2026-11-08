"""
V5 共用元件：V5Mixin

在 V4 基礎上引入 DQN，以 max_a Q(s,a) 替換 _rollout 的隨機展開評估。

訓練流程
  - choose_action  : 記錄 pending experience（pre-round 狀態 + 動作索引）
  - notify_result  : 計算 reward，存入 replay buffer
                     每 TRAIN_EVERY 次決策觸發一次 batch 訓練
                     每 TARGET_SYNC_EVERY 次決策同步 target network

狀態向量（40 維）
  手牌  : 8 槽 × 3 維（5-level 3-bit，升序排列）= 24 維
  其他  : hp×2×2 + def×2×2 + hand_sz×1 + 7-numbers×7 = 16 維
  → HandEncoder 壓縮後主網路輸入為 8×8 + 16 = 80 維

動作索引
  all_subsets(hand) 的位置 index（已去重，0=空集，最多 256）

_encode_state 同時接受 GameState 與 _AugState：
  - _AugState：使用 state.T/I/delta/R（模擬樹中傳播的 7 數字）
  - GameState：使用 self._T/I/delta/R（當前局的實例屬性）

使用方式：class FooAgentV5(V5Mixin, FooAgentV4)
"""
from __future__ import annotations
import math
import os
import random
import time
from collections import defaultdict, deque
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from agents.neural_net import DQNNet, encode_card, HAND_SLOTS, ACTION_SIZE, ENCODER_PRETRAIN_PATH
from agents.utils import all_subsets
from agents.value_function import ValueFunction
from agents.v2_mixin import _AugState
from game_engine import GameState


class DQNValue(ValueFunction):
    """V(s) = max_a Q(s,a)：以 DQN 網路取代隨機展開進行葉節點估值。"""

    def __init__(self, q_net, encode_fn, player_id: int, action_size: int):
        self._q_net      = q_net
        self._encode_fn  = encode_fn   # agent._encode_state 的 bound method
        self._player_id  = player_id
        self._action_size = action_size

    def __call__(self, gs) -> float:
        hand_feats, other_feats = self._encode_fn(gs)
        with torch.no_grad():
            q_vals = self._q_net(
                hand_feats.unsqueeze(0).to(DEVICE),
                other_feats.unsqueeze(0).to(DEVICE),
            ).squeeze(0)
        n_valid = max(1, min(len(all_subsets(gs.players[self._player_id].hand)), self._action_size))
        return float(q_vals[:n_valid].max().item())

GAMMA             = 0.95
LR                = 1e-3
BATCH_SIZE         = 256
BUFFER_MAXLEN      = 30_000
BUFFER_WRITE_LIMIT = 4_000   # 單次決策最多寫入筆數（防止終局暴衝）
TRAIN_EVERY        = 1
TRAIN_STEPS        = 10  # 每次觸發訓練時連續執行的梯度更新次數
TARGET_SYNC_EVERY  = 100

DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.98   # 每局衰減（速度加快 2 倍）


class _MCTSNode:
    """Phase 2 MCTS 樹節點：每個合法動作對應訪問次數、累積價值與子節點。"""
    __slots__ = ('N', 'W', 'children', 'prior')

    def __init__(self, prior: dict):
        self.N        = defaultdict(int)    # action → 訪問次數
        self.W        = defaultdict(float)  # action → 累積價值
        self.children: dict = {}            # action → _MCTSNode
        self.prior    = prior               # action → PUCT softmax prior


class V5Mixin:

    C_PUCT:    float = 1.0   # PUCT 探索係數
    BFS_DEPTH: int   = 5     # 廣度優先展開的層數上限
    TOP_K:     int   = 4     # 每層展開的動作數

    def __init__(self, *args, player_mode: bool = False, is_target: bool = False,
                 tau: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_target   = is_target
        self.tau         = tau
        self.player_mode = player_mode or is_target   # target 不寫 buffer、不訓練

        self._q_net   = DQNNet().to(DEVICE)
        self._tgt_net = DQNNet().to(DEVICE)

        # 載入預訓練編碼器（若存在）
        if os.path.exists(ENCODER_PRETRAIN_PATH):
            ckpt = torch.load(ENCODER_PRETRAIN_PATH, weights_only=True, map_location=DEVICE)
            self._q_net.hand_encoder.load_state_dict(ckpt)
            self._tgt_net.hand_encoder.load_state_dict(ckpt)

        self._tgt_net.load_state_dict(self._q_net.state_dict())
        self._tgt_net.eval()

        # 凍結手牌編碼器：只優化主網路參數
        for p in self._q_net.hand_encoder.parameters():
            p.requires_grad = False
        self._optimizer               = optim.Adam(self._q_net.main.parameters(), lr=LR)
        self._buffer: deque           = deque(maxlen=BUFFER_MAXLEN)
        self._decision_count: int     = 0
        self._steps_since_train: int  = 0   # 距上次訓練已過幾次決策
        self._epsilon: float          = EPSILON_START
        self._pending: Optional[dict] = None
        self._last_loss: float        = 0.0
        self._auto_target_sync: bool  = True  # False 時由外部（arena）控制同步時機
        self._writes_this_decision: int = 0
        # 覆寫基類的估值函數為 DQN（取代 RandomRolloutValue）
        self._value_fn = DQNValue(self._q_net, self._encode_state, self.player_id, ACTION_SIZE)

    # ── 狀態編碼 ─────────────────────────────────────────────

    def _encode_state(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        """回傳 (hand_feats, other_feats)，接受 GameState 或 _AugState。"""
        me  = state.players[self.player_id]
        opp = state.players[1 - self.player_id]
        me_max_hp    = me.character.max_hp
        opp_max_hp   = opp.character.max_hp
        me_base_def  = me.character.base_defense
        opp_base_def = opp.character.base_defense

        # ── 手牌：升序排列，不足補空位 (v=0) ──
        sorted_hand = sorted(me.hand)
        hand_arr: list[float] = []
        for i in range(HAND_SLOTS):
            v = sorted_hand[i] if i < len(sorted_hand) else 0
            hand_arr.extend(encode_card(v))
        hand_feats = torch.tensor(hand_arr, dtype=torch.float32).view(HAND_SLOTS, 3)

        # ── 7 數字：_AugState 優先使用其內嵌值（模擬樹傳播），否則用實例屬性 ──
        if isinstance(state, _AugState):
            T, I, delta, R = state.T, state.I, state.delta, state.R
        else:
            T     = getattr(self, '_T',     [0.0, 0.0])
            I     = getattr(self, '_I',     [0, 0])
            delta = getattr(self, '_delta', [0.0, 0.0])
            R     = getattr(self, '_R',     0)

        my    = self.player_id
        opp_id = 1 - my

        other: list[float] = [
            # HP：百分比平移 + 上限規模（以預設 55 為基準，其他角色偏移）
            me.hp  / me_max_hp  - 0.5,
            (me_max_hp  - 55) / 55.0,
            opp.hp / opp_max_hp - 0.5,
            (opp_max_hp - 55) / 55.0,
            # 防禦：比例平移 + 基礎值規模（以預設 4 為基準）
            me.defense  / max(1, me_base_def)  - 0.5,
            (me_base_def  - 4) / 4.0,
            opp.defense / max(1, opp_base_def) - 0.5,
            (opp_base_def - 4) / 4.0,
            # 對手手牌數（中心在 4）
            (len(opp.hand) - 4) / 4.0,
            # V2 七數字（Δ 以各自血量上限正規化，跨角色可比）
            T[my]     / 150.0,
            T[opp_id] / 150.0,
            I[my]     / 25.0,
            I[opp_id] / 25.0,
            delta[my]     / max(1, me_max_hp),
            delta[opp_id] / max(1, opp_max_hp),
            R / 30.0,
        ]
        other_feats = torch.tensor(other, dtype=torch.float32)
        return hand_feats, other_feats

    # ── Phase 1：移除 V4 桶裝，7 數字以原始整數直接進 info_key ──

    def _run_phase1(self, state) -> tuple:
        from agents.partial_info_agent import PartialInfoAgent
        from agents.v2_mixin import _AugState
        action_values, counts, obs, info_key = PartialInfoAgent._run_phase1(self, state)
        if isinstance(state, _AugState):
            T, I, delta, R = state.T, state.I, state.delta, state.R
        else:
            T, I, delta, R = self._T, self._I, self._delta, self._R
        my, opp = self.player_id, 1 - self.player_id
        raw_ctx = (int(T[my]), int(T[opp]), I[my], I[opp], int(delta[my]), int(delta[opp]), R)
        return action_values, counts, obs, info_key + raw_ctx

    # ── PUCT UCB 覆寫 ────────────────────────────────────────

    def _compute_puct_prior(self, state, hand_feats, other_feats) -> dict:
        """softmax(Q) 作為 PUCT prior，複用已計算的特徵，不重複呼叫網路。"""
        actions_list = all_subsets(state.players[self.player_id].hand)
        with torch.no_grad():
            q_vals = self._q_net(
                hand_feats.unsqueeze(0).to(DEVICE),
                other_feats.unsqueeze(0).to(DEVICE),
            ).squeeze(0).cpu()
        n = min(len(actions_list), q_vals.shape[0])
        probs = torch.softmax(q_vals[:n], dim=0)
        return {actions_list[i]: float(probs[i]) for i in range(n)}

    def _top_k_actions(self, actions_list, hand_feats, other_feats, k: int) -> list:
        """依 DQN Q 值回傳前 K 個動作（降序）。"""
        with torch.no_grad():
            q_vals = self._q_net(
                hand_feats.unsqueeze(0).to(DEVICE),
                other_feats.unsqueeze(0).to(DEVICE),
            ).squeeze(0).cpu()
        n = min(len(actions_list), q_vals.shape[0])
        top_idx = q_vals[:n].topk(min(k, n)).indices.tolist()
        return [actions_list[i] for i in top_idx]

    def _get_phase1_actions(self, actions_list, hand_feats, other_feats):
        """Phase 1 動作集鉤子：MCCFR 全部遍歷（回傳 None），MCTAgentV5 覆寫為 top-K。"""
        return None

    def _ucb_select(self, actions, visits, values, N):
        prior = getattr(self, '_puct_prior', None)
        if prior is None:
            return super()._ucb_select(actions, visits, values, N)
        sqrt_N = math.sqrt(N)
        n_acts = len(actions)
        best, best_score = None, -math.inf
        for a in actions:
            q = values[a] / visits[a]
            p = prior.get(a, 1.0 / n_acts)
            score = q + self.C_PUCT * p * sqrt_N / (1 + visits[a])
            if score > best_score:
                best_score, best = score, a
        return best

    # ── MCTS 樹展開（Phase 2 AlphaZero 風格）────────────────────

    def _mcts_puct_select(self, node: '_MCTSNode', actions: list, N_total: int) -> tuple:
        """樹內 PUCT 選擇：優先探索未訪問動作，再依 PUCT 分數選擇。"""
        sqrt_N = math.sqrt(N_total)
        n      = len(actions)
        best, best_score = None, -math.inf
        for a in actions:
            n_a   = node.N[a]
            q     = node.W[a] / n_a if n_a > 0 else 0.0
            p     = node.prior.get(a, 1.0 / n)
            score = q + self.C_PUCT * p * sqrt_N / (1 + n_a)
            if score > best_score:
                best_score, best = score, a
        return best

    def _mcts_search(self, node: '_MCTSNode', aug, depth: int = 0) -> float:
        """MCTS 迭代：depth < BFS_DEPTH 廣度展開 top-K，depth >= BFS_DEPTH 回退深度優先。

        BFS 層：展開 prior 最高的 TOP_K 個動作，以正規化 prior 加權平均返回父節點。
        DFS 層：PUCT 選一條路徑往下，新葉以 DQN 估值，已有子節點遞迴。
        """
        if aug.is_terminal():
            return self._evaluate(aug)

        my_hand = aug.players[self.player_id].hand
        actions = all_subsets(my_hand)
        opp_id  = 1 - self.player_id
        opp_hand = aug.players[opp_id].hand

        if depth < self.BFS_DEPTH:
            # ── BFS 廣度層：展開 top-K ────────────────────────
            k       = min(self.TOP_K, len(actions))
            top_k   = sorted(actions,
                             key=lambda a: node.prior.get(a, 1.0 / len(actions)),
                             reverse=True)[:k]
            raw_w   = [node.prior.get(a, 1.0 / len(actions)) for a in top_k]
            w_sum   = sum(raw_w)
            norm_w  = [w / w_sum for w in raw_w]

            child_values: list[float] = []
            for action in top_k:
                if time.perf_counter() >= self._search_deadline:
                    break

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

                if action not in node.children:
                    ch_hf, ch_of = self._encode_state(child_aug)
                    node.children[action] = _MCTSNode(
                        self._compute_puct_prior(child_aug, ch_hf, ch_of)
                    )

                value = self._mcts_search(node.children[action], child_aug, depth + 1)
                node.N[action] += 1
                node.W[action] += value
                child_values.append(value)

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
                    mcts_target = step_reward + (0.0 if done else GAMMA * value)
                    cur_hf, cur_of = self._encode_state(aug)
                    nxt_hf, nxt_of = self._encode_state(child_aug)
                    action_idx = next(
                        (i for i, a in enumerate(actions) if a == action), 0)
                    self._buffer.append((
                        cur_hf, cur_of, action_idx, nxt_hf, nxt_of, mcts_target, 1.0,
                    ))
                    self._writes_this_decision += 1

            # prior 加權平均返回父節點（時間截止時只對已完成的子節點重新正規化）
            n = len(child_values)
            if n == 0:
                return self._value_fn(aug)
            completed_w = norm_w[:n]
            w_total = sum(completed_w)
            return sum(v * w / w_total for v, w in zip(child_values, completed_w))

        else:
            # ── DFS 深度層：PUCT 選一條路徑 ──────────────────
            if time.perf_counter() >= self._search_deadline:
                return self._value_fn(aug)

            N_total = sum(node.N[a] for a in actions) + 1
            action  = self._mcts_puct_select(node, actions, N_total)

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

            if action not in node.children:
                ch_hf, ch_of = self._encode_state(child_aug)
                node.children[action] = _MCTSNode(
                    self._compute_puct_prior(child_aug, ch_hf, ch_of)
                )
                value = self._value_fn(child_aug)
            else:
                value = self._mcts_search(node.children[action], child_aug, depth + 1)

            node.N[action] += 1
            node.W[action] += value

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

                # max_a' Q(s', a')：與葉節點評估一致，保持 DQN off-policy 尋優性質
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

    def choose_action(self, state: GameState) -> list[int]:
        """AlphaZero 風格決策：Phase 1 計次遍歷 + Phase 2 計時 MCTS 樹展開。

        手動執行 V2Mixin 的 snap_hp 記錄與 _AugState 包裝，
        Phase 1 照跑以維護跨局 UCB 統計，Phase 2 以 MCTS 向下展開。
        所有 V5 Agent（MCT / MCCFR）均走此路徑。
        """
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

        # 設定計時截止（涵蓋 Phase 1 + Phase 2，總時間受 time_budget 約束）
        t_start = time.perf_counter()
        self._search_deadline    = t_start + self.time_budget
        self._writes_this_decision = 0

        # ── Phase 1：MCT = top-K 動作，MCCFR = 全部遍歷 ─────────
        self._filtered_actions = self._get_phase1_actions(actions_list, hand_feats, other_feats)
        phase1, p1_counts, obs, info_key = self._run_phase1(aug)
        self._filtered_actions = None
        dynamic_obs = self._make_dynamic_obs(aug)
        values_sum, visits, _ = self._merge_ucb(info_key, actions_list, phase1, p1_counts)
        self._phase1_hook(info_key, actions_list, phase1)

        # ── ε-greedy：隨機動作，跳過 Phase 2 ───────────────────
        use_epsilon = (
            not self.player_mode
            and random.random() < self._epsilon
        )
        if use_epsilon:
            action = list(random.choice(actions_list))
            self._store_ucb(info_key, actions_list, values_sum, visits)
        else:
            # ── Phase 2：計時 MCTS 樹展開 ───────────────────────
            prior = self._compute_puct_prior(aug, hand_feats, other_feats)
            root  = _MCTSNode(prior)
            # Phase 1 平均值×計數作為根節點初始統計量
            for a in actions_list:
                root.N[a] = p1_counts.get(a, 0)
                root.W[a] = phase1.get(a, 0.0) * root.N[a]

            while time.perf_counter() < self._search_deadline:
                # 每次迭代取樣一個對手手牌情境（確定化），在此情境中展開樹
                iter_aug = aug.clone()
                iter_aug.players[1 - self.player_id].hand = dynamic_obs.sample()
                self._mcts_search(root, iter_aug)

            # 訪問次數最多的動作為最終決策（AlphaZero 慣例）；MCCFR 覆寫為後悔值匹配
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

    # ── Phase 1 / Phase 2 鉤子（MCCFR 子類覆寫以恢復後悔值匹配）────────

    def _phase1_hook(self, info_key, actions, phase1_values) -> None:
        """Phase 1 完成後呼叫；MCCFRAgentV5 覆寫以執行第一次 regret update。"""
        pass

    def _choose_from_mcts(self, actions, root, info_key) -> list:
        """從 MCTS 統計中選出最終動作；MCCFRAgentV5 覆寫以回傳後悔值匹配策略。"""
        return list(max(actions, key=lambda a: root.N[a]))

    # ── 經驗收集 ──────────────────────────────────────────────

    def notify_result(self, state) -> None:
        # 先讓上層更新 7 數字等內部狀態，_encode_state(state) 才能得到正確的 next_state
        super().notify_result(state)

        if self._pending is None or not state.history:
            return

        result  = state.history[-1]
        me      = state.players[self.player_id]
        opp     = state.players[1 - self.player_id]
        my_id   = self.player_id

        my_hp_after  = result.p0_hp_after  if my_id == 0 else result.p1_hp_after
        opp_hp_after = result.p1_hp_after  if my_id == 0 else result.p0_hp_after

        my_dmg  = (self._pending['my_hp_before']  - my_hp_after)  / me.character.max_hp
        opp_dmg = (self._pending['opp_hp_before'] - opp_hp_after) / opp.character.max_hp
        reward  = float(opp_dmg - my_dmg)

        done = state.is_terminal()
        if done:
            reward += 1.0 if (state.winner() == my_id) else -1.0

        pending        = self._pending
        self._pending  = None
        self._decision_count += 1

        if self.player_mode:
            return

        next_hand, next_other = self._encode_state(state)
        self._buffer.append((
            pending['hand'],
            pending['other'],
            pending['action_idx'],
            next_hand,
            next_other,
            reward,
            float(done),
        ))

        self._steps_since_train += 1
        if self._steps_since_train >= TRAIN_EVERY and len(self._buffer) >= BATCH_SIZE:
            for _ in range(TRAIN_STEPS):
                self._train_step()
            self._steps_since_train = 0
        if self._auto_target_sync and self._decision_count % TARGET_SYNC_EVERY == 0:
            self._tgt_net.load_state_dict(self._q_net.state_dict())

    # ── DQN 訓練 ──────────────────────────────────────────────

    def _train_step(self) -> None:
        batch = random.sample(self._buffer, BATCH_SIZE)
        h, o, a_idx, nh, no, r, done = zip(*batch)

        h_t  = torch.stack(h).to(DEVICE)
        o_t  = torch.stack(o).to(DEVICE)
        nh_t = torch.stack(nh).to(DEVICE)
        no_t = torch.stack(no).to(DEVICE)
        a_t  = torch.tensor(a_idx, dtype=torch.long,    device=DEVICE)
        r_t  = torch.tensor(r,     dtype=torch.float32, device=DEVICE)
        d_t  = torch.tensor(done,  dtype=torch.float32, device=DEVICE)

        q_sa = self._q_net(h_t, o_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            ext = getattr(self, '_ext_target', None)
            tgt = ext._q_net if ext is not None else self._tgt_net
            q_next = tgt(nh_t, no_t).max(dim=1).values
            target = r_t + GAMMA * q_next * (1.0 - d_t)

        loss = nn.functional.mse_loss(q_sa, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._last_loss = loss.item()

    def set_target_agent(self, target: "V5Mixin") -> None:
        """設定外部 target agent；訓練時用其 _q_net 取代內部 _tgt_net 計算 TD target。"""
        self._ext_target = target

    def hard_sync_target(self) -> None:
        """將 online 權重完整複製到 target network（由 arena 定期呼叫）。"""
        self._tgt_net.load_state_dict(self._q_net.state_dict())

    # ── 軟更新（自我對戰用）────────────────────────────────────

    def soft_update_from(self, online: "V5Mixin") -> None:
        """θ_self = τ·θ_online + (1−τ)·θ_self"""
        with torch.no_grad():
            for tp, sp in zip(self._q_net.parameters(), online._q_net.parameters()):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    # ── 每局重置 ──────────────────────────────────────────────

    def new_game(self) -> None:
        super().new_game()
        self._pending = None
        self._epsilon = max(EPSILON_END, self._epsilon * EPSILON_DECAY)

    # ── 存檔 / 讀檔 ───────────────────────────────────────────

    @staticmethod
    def _dqn_path(path: str) -> str:
        return path.replace('.pkl', '_dqn.pt')

    def save(self, path: str) -> None:
        super().save(path)
        torch.save({
            'q_net'         : self._q_net.state_dict(),
            'tgt_net'       : self._tgt_net.state_dict(),
            'optimizer'     : self._optimizer.state_dict(),
            'decision_count': self._decision_count,
            'epsilon'       : self._epsilon,
        }, self._dqn_path(path))

    def load(self, path: str) -> None:
        super().load(path)
        dqn_path = self._dqn_path(path)
        if os.path.exists(dqn_path):
            ckpt = torch.load(dqn_path, weights_only=True)
            self._q_net.load_state_dict(ckpt['q_net'])
            self._tgt_net.load_state_dict(ckpt['tgt_net'])
            self._optimizer.load_state_dict(ckpt['optimizer'])
            self._decision_count = ckpt.get('decision_count', 0)
            self._epsilon        = ckpt.get('epsilon', EPSILON_END)
