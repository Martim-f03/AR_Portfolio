import numpy as np

from mia_rl.mdps.car_rental_mdp import (
    zeros_V,
    bellman_optimality_backup_v,
    q_from_v,
)


def value_iteration(
    mdp,
    gamma=0.9,
    theta=1e-6,
    max_iters=10_000,
):

    V = zeros_V(mdp)

    for it in range(max_iters):

        delta = 0.0

        V_old = V.copy()

        for s in mdp.states():

            v_new = bellman_optimality_backup_v(
                mdp,
                V_old,
                s,
                gamma,
            )

            delta = max(
                delta,
                abs(v_new - V[s[0], s[1]])
            )

            V[s[0], s[1]] = v_new

        if delta < theta:
            break

    policy = {}

    for s in mdp.states():

        best_a = None
        best_q = -np.inf

        for a in mdp.possible_actions(s):

            q = q_from_v(
                mdp,
                V,
                s,
                a,
                gamma,
            )

            if q > best_q:
                best_q = q
                best_a = a

        policy[s] = best_a

    return V, policy, it + 1