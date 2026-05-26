import numpy as np
from mia_rl.agents.bandits.base import BanditAgent


class GradientBandit(BanditAgent):
    def __init__(self, k=10, alpha=0.1, baseline=True):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.reset()

    def reset(self):
        super().reset()
        self.H = np.zeros(self.k)
        self.avg_reward = 0.0

    def _policy(self):
        # Softmax com proteção contra overflow numérico (subtraindo o max(H))
        exp = np.exp(self.H - np.max(self.H))
        return exp / np.sum(exp)

    def select_action(self):
        probs = self._policy()
        # Seleciona a ação com base nas probabilidades calculadas pela política
        return np.random.choice(self.k, p=probs)

    def update(self, action, reward):
        self.t += 1
        probs = self._policy()

        if self.baseline:
            # Atualização incremental da recompensa média (baseline) até ao momento
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline = self.avg_reward
        else:
            baseline = 0

        # Atualiza as preferências (H) para todas as ações
        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - baseline) * (1 - probs[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline) * probs[a]

