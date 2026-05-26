from mia_rl.envs.gridworld import Gridworld
from mia_rl.policies.random_policy import uniform_random_policy
from mia_rl.mdps.policy_evaluation import policy_evaluation
from mia_rl.mdps.value_iteration import value_iteration, greedy_policy_from_V
from mia_rl.plots.gridworld_plots import plot_grid_values_and_policy


def main():
    env = Gridworld()
    gamma = 0.9

    policy = uniform_random_policy(env)

    V_pi, _ = policy_evaluation(env, policy, gamma)
    V_star, _ = value_iteration(env, gamma)
    pi_star = greedy_policy_from_V(env, V_star, gamma)

    plot_grid_values_and_policy(env, V_pi, None, "Vπ")
    plot_grid_values_and_policy(env, V_star, pi_star, "V*")


if __name__ == "__main__":
    main()