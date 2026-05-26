from mia_rl.mdps.car_rental_mdp import (
    zeros_V,
    bellman_expectation_backup_v,
)


def policy_evaluation(
    mdp,
    policy,
    gamma=0.9,
    theta=1e-6,
    max_iters=10_000,
):

    V = zeros_V(mdp)

    for it in range(max_iters):

        delta = 0.0

        V_old = V.copy()

        for s in mdp.states():

            v_new = bellman_expectation_backup_v(
                mdp,
                V_old,
                s,
                policy,
                gamma,
            )

            delta = max(
                delta,
                abs(v_new - V[s[0], s[1]])
            )

            V[s[0], s[1]] = v_new

        if delta < theta:
            return V, it + 1

    return V, max_iters