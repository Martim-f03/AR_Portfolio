import numpy as np


def zeros_V(mdp):

    return np.zeros(
        (
            mdp.params.max_cars_1 + 1,
            mdp.params.max_cars_2 + 1,
        )
    )


def q_from_v(mdp, V, s, a, gamma):

    p1, p2, exp_revenue = mdp.expected_transition(s, a)

    exp_next = 0.0

    for n1p, p1v in enumerate(p1):

        if p1v == 0:
            continue

        for n2p, p2v in enumerate(p2):

            if p2v == 0:
                continue

            exp_next += (
                p1v
                * p2v
                * V[n1p, n2p]
            )

    move_cost = (
        mdp.params.cost_per_moved
        * abs(a)
    )

    reward = exp_revenue - move_cost

    return reward + gamma * exp_next


def bellman_expectation_backup_v(
    mdp,
    V,
    s,
    policy,
    gamma,
):

    a = policy[s]

    return q_from_v(
        mdp,
        V,
        s,
        a,
        gamma,
    )


def bellman_optimality_backup_v(
    mdp,
    V,
    s,
    gamma,
):

    best = -np.inf

    for a in mdp.possible_actions(s):

        q = q_from_v(
            mdp,
            V,
            s,
            a,
            gamma,
        )

        best = max(best, q)

    return best