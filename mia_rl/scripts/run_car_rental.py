from mia_rl.envs.car_rental import (
    CarRentalMDP,
    CarRentalParams,
)

from mia_rl.experiments.car_rental_experiment import (
    run_policy_iteration_experiment,
    run_value_iteration_experiment,
)

from mia_rl.plots.car_rental_plots import (
    plot_policy,
    plot_values,
)


def main():

    params = CarRentalParams()

    mdp = CarRentalMDP(params)

    gamma = 0.9

    # Policy Iteration
    V_pi, pi_pi, hist = (
        run_policy_iteration_experiment(
            mdp,
            gamma=gamma,
        )
    )

    print(
        "Policy Iteration loops:",
        len(hist),
    )

    plot_policy(
        mdp,
        pi_pi,
        title="Policy Iteration Policy",
    )

    plot_values(
        mdp,
        V_pi,
        title="Policy Iteration Values",
    )

    # Value Iteration
    V_vi, pi_vi, it_vi = (
        run_value_iteration_experiment(
            mdp,
            gamma=gamma,
        )
    )

    print(
        "Value Iteration iterations:",
        it_vi,
    )

    plot_policy(
        mdp,
        pi_vi,
        title="Value Iteration Policy",
    )

    plot_values(
        mdp,
        V_vi,
        title="Value Iteration Values",
    )


if __name__ == "__main__":
    main()