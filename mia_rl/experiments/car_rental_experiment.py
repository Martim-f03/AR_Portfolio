from mia_rl.mdps.policy_iteration import (
    policy_iteration,
)

from mia_rl.mdps.value_iteration_car import (
    value_iteration,
)


def run_policy_iteration_experiment(
    mdp,
    gamma=0.9,
    theta=1e-4,
):

    return policy_iteration(
        mdp,
        gamma,
        theta,
    )


def run_value_iteration_experiment(
    mdp,
    gamma=0.9,
    theta=1e-4,
):

    return value_iteration(
        mdp,
        gamma,
        theta,
    )