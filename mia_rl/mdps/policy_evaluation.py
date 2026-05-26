import numpy as np
from mia_rl.mdps.gridworld_mdp import bellman_expectation_update


def zeros_V(env):
    return np.zeros((env.n_rows, env.n_cols))


def policy_evaluation(env, policy, gamma=0.9, theta=1e-6, max_iters=10_000):
    V = zeros_V(env)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        for s in env.states():
            v_new = bellman_expectation_update(env, V_old, policy, s, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new

        if delta < theta:
            return V, it + 1

    return V, max_iters