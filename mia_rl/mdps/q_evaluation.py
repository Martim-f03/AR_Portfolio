import numpy as np
from mia_rl.envs.gridworld import ACTIONS


def zeros_Q(env):
    return np.zeros((env.n_rows, env.n_cols, len(ACTIONS)))


def policy_evaluation_Q(env, policy, gamma=0.9, theta=1e-6, max_iters=10_000):
    Q = zeros_Q(env)

    for it in range(max_iters):
        delta = 0.0
        Q_old = Q.copy()

        for s in env.states():
            if env.is_terminal(s):
                continue

            r, c = s

            for a_idx, a in enumerate(ACTIONS):
                ns, reward, _ = env.step(s, a)
                nr, nc = ns

                exp_next = sum(
                    policy[ns][a2] * Q_old[nr, nc, j]
                    for j, a2 in enumerate(ACTIONS)
                )

                q_new = reward + gamma * exp_next

                delta = max(delta, abs(q_new - Q[r, c, a_idx]))
                Q[r, c, a_idx] = q_new

        if delta < theta:
            return Q, it + 1

    return Q, max_iters