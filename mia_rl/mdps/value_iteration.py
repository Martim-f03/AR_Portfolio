import numpy as np
from mia_rl.envs.gridworld import ACTIONS
from mia_rl.mdps.gridworld_mdp import bellman_optimality_update


def zeros_V(env):
    return np.zeros((env.n_rows, env.n_cols))


def value_iteration(env, gamma=0.9, theta=1e-6, max_iters=10_000):
    V = zeros_V(env)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        for s in env.states():
            v_new = bellman_optimality_update(env, V_old, s, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new

        if delta < theta:
            return V, it + 1

    return V, max_iters


def greedy_policy_from_V(env, V, gamma):
    policy = {}

    for s in env.states():
        if env.is_terminal(s):
            policy[s] = "·"
            continue

        best_a, best_q = None, -np.inf

        for a in ACTIONS:
            ns, r, _ = env.step(s, a)
            q = r + gamma * V[ns[0], ns[1]]

            if q > best_q:
                best_q = q
                best_a = a

        policy[s] = best_a

    return policy