import numpy as np

class BanditAgent:
    def __init__(self, k=10):
        self.k = k

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        raise NotImplementedError