import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

ACTIONS = ["U", "D", "L", "R"]

ACTION_TO_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass(frozen=True)
class Gridworld:
    n_rows: int = 4
    n_cols: int = 4
    terminal_states: Tuple[Tuple[int, int], ...] = ((0, 0), (3, 3))
    step_reward: float = -1.0

    def states(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        return state in self.terminal_states

    def step(self, state: Tuple[int, int], action: str):
        if self.is_terminal(state):
            return state, 0.0, True

        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = state[0] + dr, state[1] + dc

        if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
            next_state = state
        else:
            next_state = (nr, nc)

        reward = self.step_reward
        done = self.is_terminal(next_state)

        return next_state, reward, done