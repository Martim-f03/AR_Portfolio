import numpy as np
from mia_rl.agents.bandits.base import BanditAgent


class EpsilonGreedy(BanditAgent):
    def __init__(
        self,
        k=10,
        epsilon=0.1,
        alpha=None,
        optimistic=0.0,
    ):
        super().__init__(k)

        self.epsilon = epsilon
        self.alpha = alpha
        self.optimistic = optimistic

        self.reset()

    def reset(self):
        super().reset()
        self.Q[:] = self.optimistic

    def select_action(self):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)

        # Greedy action with random tie-breaking
        max_q = np.max(self.Q)
        actions = np.where(self.Q == max_q)[0]

        return np.random.choice(actions)

    def update(self, action, reward):
        self.t += 1
        self.N[action] += 1

        # Constant step size (non-stationary problems)
        if self.alpha is not None:
            step_size = self.alpha

        # Sample-average update
        else:
            step_size = 1.0 / self.N[action]

        self.Q[action] += (
            step_size * (reward - self.Q[action])
        )