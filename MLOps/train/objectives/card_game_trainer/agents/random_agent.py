from __future__ import annotations
import random
from agents.base_agent import BaseAgent
from agents.utils import all_subsets
from game_engine import GameState


class RandomAgent(BaseAgent):
    def choose_action(self, state: GameState) -> list[int]:
        hand = state.players[self.player_id].hand
        return list(random.choice(all_subsets(hand)))
