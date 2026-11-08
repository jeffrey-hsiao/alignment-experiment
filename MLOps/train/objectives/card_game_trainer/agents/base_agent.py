from __future__ import annotations
from abc import ABC, abstractmethod
from game_engine import GameState


class BaseAgent(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def choose_action(self, state: GameState) -> list[int]:
        """Return a list of cards to play from hand."""
        ...

    def notify_result(self, state: GameState):
        """Called after each round with updated state (optional hook)."""
        pass
