import numpy as np

def policy_evaluation(env, policy, gamma=0.9, theta=1e-8):
    V = np.zeros((env.n_rows, env.n_cols))

    while True:
        delta = 0
        V_old = V.copy()

        for s in env.states():
            if env.is_terminal(s):
                continue

            v_new = 0.0
            for a, p in policy[s].items():
                ns, r, done = env.step(s, a)
                v_new += p * (r + gamma * V_old[ns[0], ns[1]])

            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < theta:
            break

    return V