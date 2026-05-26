import numpy as np

from mia_rl.mdps.policy_evaluation_car import (
    policy_evaluation,
)

from mia_rl.mdps.car_rental_mdp import (
    q_from_v,
)


def policy_improvement(
    mdp,
    V,
    old_policy,
    gamma,
):

    new_policy = {}

    stable = True

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

        new_policy[s] = best_a

        if old_policy[s] != best_a:
            stable = False

    return new_policy, stable


def policy_iteration(
    mdp,
    gamma=0.9,
    theta=1e-6,
    max_outer=20,
):

    policy = {
        s: 0
        for s in mdp.states()
    }

    history = []

    for outer in range(max_outer):

        V, eval_iters = policy_evaluation(
            mdp,
            policy,
            gamma,
            theta,
        )

        new_policy, stable = policy_improvement(
            mdp,
            V,
            policy,
            gamma,
        )

        history.append(
            (
                outer,
                eval_iters,
                V.copy(),
                new_policy.copy(),
            )
        )

        policy = new_policy

        if stable:
            break

    return V, policy, history