import numpy as np
from mia_rl.envs.gridworld import ACTIONS


# ============================================================
# Bellman Expectation (Policy Evaluation)
# ============================================================

def bellman_expectation_update(env, V, policy, state, gamma):
    if env.is_terminal(state):
        return 0.0

    v_new = 0.0

    for a, p in policy[state].items():
        ns, r, _ = env.step(state, a)
        v_new += p * (r + gamma * V[ns[0], ns[1]])

    return v_new


# ============================================================
# Bellman Optimality (Value Iteration)
# ============================================================

def bellman_optimality_update(env, V, state, gamma):
    if env.is_terminal(state):
        return 0.0

    best = -np.inf

    for a in ACTIONS:
        ns, r, _ = env.step(state, a)
        best = max(best, r + gamma * V[ns[0], ns[1]])

    return best