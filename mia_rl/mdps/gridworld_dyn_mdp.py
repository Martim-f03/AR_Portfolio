import numpy as np
from mia_rl.envs.gridworld_dyn import ACTIONS


def zeros_V(env):
    return np.zeros((env.n_rows, env.n_cols))


def bellman_expectation(env, V, s, policy, gamma):
    if env.is_terminal(s):
        return 0.0

    v = 0.0
    for a, p in policy[s].items():
        ns, r, _ = env.step(s, a)
        v += p * (r + gamma * V[ns])
    return v


def q_from_v(env, V, s, a, gamma):
    ns, r, _ = env.step(s, a)
    return r + gamma * V[ns]


def greedy_action(env, V, s, gamma):
    best_a = None
    best_q = -1e9

    for a in ACTIONS:
        q = q_from_v(env, V, s, a, gamma)
        if q > best_q:
            best_q = q
            best_a = a

    return best_a