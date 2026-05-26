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
class GridworldDyn:
    n_rows: int = 4
    n_cols: int = 4
    terminal_states: Tuple[Tuple[int, int], ...] = ((0, 0), (3, 3))
    step_reward: float = -1.0

    def states(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]

    def is_terminal(self, s):
        return s in self.terminal_states

    def step(self, s, a):
        if self.is_terminal(s):
            return s, 0.0, True

        dr, dc = ACTION_TO_DELTA[a]
        nr, nc = s[0] + dr, s[1] + dc

        if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
            ns = s
        else:
            ns = (nr, nc)

        return ns, self.step_reward, self.is_terminal(ns)