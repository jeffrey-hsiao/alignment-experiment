from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional, Callable

MAX_HAND_SIZE = 8


@dataclass
class Character:
    name: str
    max_hp: int = 55
    base_defense: int = 4


DEFAULT_CHARACTER = Character("Default")


class Deck:
    def __init__(self):
        self._cards: list[int] = []
        self._refill()

    def _refill(self):
        self._cards = list(range(1, 11))
        random.shuffle(self._cards)

    def draw(self, n: int) -> list[int]:
        drawn = []
        for _ in range(n):
            if not self._cards:
                self._refill()
            drawn.append(self._cards.pop())
        return drawn


class Player:
    def __init__(self, player_id: int, character: Character = DEFAULT_CHARACTER):
        self.player_id = player_id
        self.character = character
        self.hp: int = character.max_hp
        self.defense: int = character.base_defense
        self.hand: list[int] = []
        self.deck: Deck = Deck()

    def is_alive(self) -> bool:
        return self.hp > 0

    def apply_end_of_round(self, broken: bool = False):
        if not self.hand:
            if not broken:
                self.defense = min(self.defense + 1, self.character.base_defense)
            else:
                self.defense -= 1
            return
        self.hand = [c + 1 for c in self.hand]
        if broken:
            self.defense -= 1
        else:
            self.defense = min(self.defense + 1, self.character.base_defense)

    def remove_cards(self, played: list[int]):
        hand = self.hand[:]
        for card in played:
            hand.remove(card)
        self.hand = hand

    def add_card(self, value: int):
        self.hand.append(value)

    def draw_from(self, n: int):
        available = max(0, MAX_HAND_SIZE - len(self.hand))
        self.hand.extend(self.deck.draw(min(n, available)))

    def clone(self) -> Player:
        p = Player(self.player_id, self.character)
        p.hp = self.hp
        p.defense = self.defense
        p.hand = self.hand[:]
        p.deck = Deck()
        p.deck._cards = self.deck._cards[:]
        return p


@dataclass
class RoundResult:
    winner: Optional[int]       # player_id or None (draw)
    difference: int
    reward_card: int             # 獎勵牌面值（0 表示無勝者）
    p0_played: list[int]
    p1_played: list[int]
    p0_hp_after: int
    p1_hp_after: int
    p0_hand_size_after: int      # 回合結束後 p0 手牌數（可觀測）
    p1_hand_size_after: int      # 回合結束後 p1 手牌數（可觀測）


@dataclass
class PlayerObservation:
    """單一玩家的可觀測局面快照（不含對手手牌內容）。"""
    player_id: int
    my_hp: int
    my_defense: int
    my_hand: list[int]
    opp_hp: int
    opp_defense: int
    opp_hand_size: int
    round_num: int
    # 每回合歷史：(winner, difference, my_played, opp_hand_size_after)
    round_history: list[tuple]


def _infer_opp_hand(obs: PlayerObservation) -> list[int]:
    """
    僅憑可觀測資訊推算對手手牌。

    利用：每回合的差值 + 自身出牌 → 推算對手出牌總和；
    再結合對手手牌數量變化 → 推算每張牌的平均值。
    """
    n = obs.opp_hand_size
    if n == 0:
        return []

    per_card_avgs: list[float] = []
    prev_opp_size = 2  # 初始手牌：固定的 5 + 抽 1 張

    for winner, diff, my_played, opp_size_after in obs.round_history:
        my_sum = sum(my_played)
        opp_won = (winner is not None and winner != obs.player_id)

        if winner == obs.player_id:
            opp_sum = max(0, my_sum - diff)
        elif opp_won:
            opp_sum = my_sum + diff
        elif diff > my_sum:
            opp_sum = my_sum + diff  # 對手必定出更多（my_sum - diff < 0 不可能）
        else:
            opp_sum = my_sum  # diff==0（平手）或方向不確定，取中間估計

        # opp 出了幾張：size前 + 補2張 + 獎勵1張(if win) - size後
        played_count = max(0, prev_opp_size + 2 + (1 if opp_won else 0) - opp_size_after)

        if played_count > 0:
            per_card_avgs.append(opp_sum / played_count)

        prev_opp_size = opp_size_after

    mean_val = sum(per_card_avgs) / len(per_card_avgs) if per_card_avgs else 5.0
    return [max(1, round(random.gauss(mean_val, 2))) for _ in range(n)]


def _estimate_opp_mean(obs: PlayerObservation) -> float:
    """從可觀測歷史估算對手每張牌的平均值（供 DynamicObs 使用）。"""
    per_card_avgs: list[float] = []
    prev_opp_size = 2
    for winner, diff, my_played, opp_size_after in obs.round_history:
        my_sum = sum(my_played)
        opp_won = (winner is not None and winner != obs.player_id)
        if winner == obs.player_id:
            opp_sum = max(0, my_sum - diff)
        elif opp_won:
            opp_sum = my_sum + diff
        elif diff > my_sum:
            opp_sum = my_sum + diff
        else:
            opp_sum = my_sum
        played_count = max(0, prev_opp_size + 2 + (1 if opp_won else 0) - opp_size_after)
        if played_count > 0:
            per_card_avgs.append(opp_sum / played_count)
        prev_opp_size = opp_size_after
    return sum(per_card_avgs) / len(per_card_avgs) if per_card_avgs else 5.0


class DynamicObs:
    """動態對手手牌採樣器。

    每次 sample() 根據可觀測歷史估算一組合理的對手手牌：
    - 以歷史推算 mean_val，每次加入隨機假設擾動（保守 / 激進 / 標準）
    - 牌值上限 = 10 + round_num（考慮持牌累積 +1）
    不使用上帝視角，所有資訊皆來自 PlayerObservation。
    """

    def __init__(self, obs: PlayerObservation):
        self._obs = obs
        self._max_card = 10 + obs.round_num
        self._mean_val = _estimate_opp_mean(obs)

    @property
    def opp_hand_size(self) -> int:
        return self._obs.opp_hand_size

    def sample(self) -> list[int]:
        n = self._obs.opp_hand_size
        if n == 0:
            return []
        r = random.random()
        if r < 0.15:
            # 保守假設：對手傾向低值出牌
            mean_val = max(1.0, self._mean_val - random.uniform(1.0, 2.5))
        elif r < 0.30:
            # 激進假設：對手傾向高值出牌
            mean_val = min(float(self._max_card), self._mean_val + random.uniform(1.0, 2.5))
        else:
            # 標準假設：均值附近小幅擾動
            mean_val = self._mean_val + random.gauss(0, 0.8)
        return [
            max(1, min(self._max_card, round(random.gauss(mean_val, 2.0))))
            for _ in range(n)
        ]


class GameState:
    def __init__(
        self,
        char0: Character = DEFAULT_CHARACTER,
        char1: Character = DEFAULT_CHARACTER,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
        self.players = [Player(0, char0), Player(1, char1)]
        self.round_num = 0
        self.history: list[RoundResult] = []

        for p in self.players:
            p.deck._cards.remove(5)
            p.hand.append(5)
            p.draw_from(1)

    def is_terminal(self) -> bool:
        return any(not p.is_alive() for p in self.players)

    def winner(self) -> Optional[int]:
        for p in self.players:
            if not p.is_alive():
                return 1 - p.player_id
        return None

    def apply_round(self, played0: list[int], played1: list[int]) -> RoundResult:
        p0, p1 = self.players
        total0 = sum(played0)
        total1 = sum(played1)
        diff = abs(total0 - total1)

        winner_id: Optional[int] = None
        reward0: Optional[int] = None
        reward1: Optional[int] = None
        p0_broken = False
        p1_broken = False

        if total0 > total1:
            if diff >= p1.defense:
                winner_id = 0
                p1.hp -= diff - p1.defense
                reward0 = max(1, diff - p1.defense)
                p1_broken = True
            else:
                p1.defense -= diff
        elif total1 > total0:
            if diff >= p0.defense:
                winner_id = 1
                p0.hp -= diff - p0.defense
                reward1 = max(1, diff - p0.defense)
                p0_broken = True
            else:
                p0.defense -= diff

        p0.remove_cards(played0)
        p1.remove_cards(played1)

        p0.apply_end_of_round(broken=p0_broken)
        p1.apply_end_of_round(broken=p1_broken)

        if reward0 is not None:
            p0.add_card(reward0)
        if reward1 is not None:
            p1.add_card(reward1)

        for p in self.players:
            p.draw_from(2)

        self.round_num += 1

        result = RoundResult(
            winner=winner_id,
            difference=diff,
            reward_card=reward0 if reward0 is not None else (reward1 if reward1 is not None else 0),
            p0_played=played0,
            p1_played=played1,
            p0_hp_after=p0.hp,
            p1_hp_after=p1.hp,
            p0_hand_size_after=len(p0.hand),
            p1_hand_size_after=len(p1.hand),
        )
        self.history.append(result)
        return result

    def clone(self) -> GameState:
        gs = GameState.__new__(GameState)
        gs.players = [p.clone() for p in self.players]
        gs.round_num = self.round_num
        gs.history = self.history[:]
        return gs

    def get_info_state(self, player_id: int) -> tuple:
        p = self.players[player_id]
        obs_history = tuple(
            (r.winner, r.difference) for r in self.history
        )
        return (
            player_id,
            tuple(sorted(p.hand)),
            p.hp,
            p.defense,
            self.players[1 - player_id].hp,
            self.players[1 - player_id].defense,
            obs_history,
        )

    # ── 可觀測視角 ────────────────────────────────────────

    def get_observation(self, player_id: int) -> PlayerObservation:
        """回傳 player_id 可觀測到的局面資訊，不含對手手牌內容。"""
        me = self.players[player_id]
        opp = self.players[1 - player_id]
        history = []
        for r in self.history:
            my_played = r.p0_played if player_id == 0 else r.p1_played
            opp_size = r.p1_hand_size_after if player_id == 0 else r.p0_hand_size_after
            history.append((r.winner, r.difference, my_played[:], opp_size))
        return PlayerObservation(
            player_id=player_id,
            my_hp=me.hp,
            my_defense=me.defense,
            my_hand=me.hand[:],
            opp_hp=opp.hp,
            opp_defense=opp.defense,
            opp_hand_size=len(opp.hand),
            round_num=self.round_num,
            round_history=history,
        )

    def display_for(self, player_id: int) -> str:
        """回傳 player_id 視角的盤面文字，供人機介面與 AI 統一使用。"""
        obs = self.get_observation(player_id)
        hand_str = "[" + "  ".join(str(c) for c in sorted(obs.my_hand)) + "]"
        lines = [
            f"{'='*50}",
            f"  Round {obs.round_num + 1}",
            f"  You (P{player_id})   HP:{obs.my_hp:>3}  DEF:{obs.my_defense}"
            f"  Hand: {hand_str}",
            f"  Opponent     HP:{obs.opp_hp:>3}  DEF:{obs.opp_defense}"
            f"  Hand: [{obs.opp_hand_size} cards hidden]",
            f"{'='*50}",
        ]
        return "\n".join(lines)

    # ── 蒙地卡羅模擬用方法 ───────────────────────────────

    def clone_for_simulation(self, player_id: int) -> GameState:
        """複製局面供蒙地卡羅使用；對手手牌清空，待 sample_unknown 填入。"""
        gs = self.clone()
        gs.players[1 - player_id].hand = []
        return gs

    def sample_unknown(self, player_id: int) -> None:
        """
        在此（複製）物件上，用純可觀測資訊推算對手手牌並填入。
        搭配 clone_for_simulation() 使用。
        """
        self.players[1 - player_id].hand = _infer_opp_hand(
            self.get_observation(player_id)
        )

    def estimate_opp_hand(self, player_id: int) -> list[int]:
        """回傳對手手牌估算列表，不修改自身狀態。"""
        return _infer_opp_hand(self.get_observation(player_id))

    def run_simulations(
        self,
        n_sims: int,
        depth: int,
        p0_fn: Callable[[list[int]], list[int]],
        p1_fn: Callable[[list[int]], list[int]],
        eval_fn: Callable[[GameState], float],
        player_id: Optional[int] = None,
    ) -> list[float]:
        """
        在單一物件上執行 n_sims 次 rollout，每次模擬前還原快照，
        比對外產生 N 個複製物件更省記憶體。

        player_id: 若指定，每次模擬前重新抽樣對手手牌（蒙地卡羅不完全資訊用）。
        p0_fn / p1_fn: callable(hand: list[int]) -> list[int]，決定雙方出牌。
        eval_fn: callable(GameState) -> float，評估終局或截止局面的價值。
        """
        snap = self._snapshot()
        results: list[float] = []
        for _ in range(n_sims):
            self._restore(snap)
            if player_id is not None:
                self.sample_unknown(player_id)
            for _ in range(depth):
                if self.is_terminal():
                    break
                a0 = p0_fn(self.players[0].hand)
                a1 = p1_fn(self.players[1].hand)
                self.apply_round(a0, a1)
            results.append(eval_fn(self))
        self._restore(snap)
        return results

    # ── 快照 / 還原（供 run_simulations 內部使用）────────

    def _snapshot(self) -> dict:
        return {
            'p0': self.players[0].clone(),
            'p1': self.players[1].clone(),
            'round_num': self.round_num,
            'history': self.history[:],
        }

    def _restore(self, snap: dict) -> None:
        self.players[0] = snap['p0'].clone()
        self.players[1] = snap['p1'].clone()
        self.round_num = snap['round_num']
        self.history = snap['history'][:]

    def validate_action(self, player_id: int, action: list[int], llm_mode: bool = True) -> tuple[bool, str, int]:
        """
        驗證動作合法性。返回 (is_valid, error_msg, error_code)

        Args:
            player_id: 玩家 ID (0 或 1)
            action: 要檢查的動作列表（牌值）
            llm_mode: 若 True（預設），靜默返回；若 False，在非法時打印訊息

        Returns:
            (is_valid, error_msg, error_code)
            - is_valid: bool，動作是否合法
            - error_msg: str，錯誤訊息（有效時為空字符串）
            - error_code: int，0=合法, 1=重複牌, 2=手上沒有該牌, 3=其他
        """
        hand = self.players[player_id].hand[:]

        # 空動作總是合法的
        if not action:
            return True, "", 0

        # 檢查是否有重複牌
        if len(action) != len(set(action)):
            msg = f"動作中有重複的牌：{action}"
            if not llm_mode:
                print(f"  [驗證錯誤] {msg}")
            return False, msg, 1

        # 檢查每張牌是否在手牌中
        for card in action:
            if card not in hand:
                msg = f"手上沒有牌 {card}（現有：{sorted(hand)}）"
                if not llm_mode:
                    print(f"  [驗證錯誤] {msg}")
                return False, msg, 2
            hand.remove(card)

        # 合法動作
        return True, "", 0
