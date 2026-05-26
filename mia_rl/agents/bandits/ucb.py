import numpy as np
from mia_rl.agents.bandits.base import BanditAgent

class UCB(BanditAgent):
    def __init__(self, k=10, c=2.0):
        super().__init__(k)
        self.c = c

    def select_action(self):
        self.t += 1

        # Garante que cada ação é testada pelo menos uma vez
        for a in range(self.k):
            if self.N[a] == 0:
                return a

        # Aplica a fórmula do Upper Confidence Bound (UCB)
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        max_ucb = np.max(ucb_values)
        actions = np.where(ucb_values == max_ucb)[0]
        return np.random.choice(actions)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]