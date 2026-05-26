#!/usr/bin/env python3

from mia_rl.envs.karmed import KArmedBandit

from mia_rl.agents.bandits.e_greedy import EpsilonGreedy
from mia_rl.agents.bandits.ucb import UCB
from mia_rl.agents.bandits.gradient import GradientBandit

from mia_rl.experiments.bandit_experiment import run_experiment

from mia_rl.plots.bandit_plots import (
    plot_epsilon_greedy,
    plot_optimistic_vs_ucb,
    plot_gradient_bandit
)

import matplotlib.pyplot as plt
import os


def run_single_experiment():
    env = KArmedBandit(k=10)

    agents = {
        "ε-greedy": EpsilonGreedy(epsilon=0.1),
        "UCB": UCB(c=2.0),
        "Gradient": GradientBandit(alpha=0.1, baseline=True)
    }

    results = {}

    for name, agent in agents.items():
        rewards, optimal = run_experiment(agent, env, steps=1000, runs=500)
        results[name] = (rewards, optimal)

    return results


def run_all_plots():
    print("Running ε-greedy plot...")
    plot_epsilon_greedy()

    print("Running UCB vs Optimistic plot...")
    plot_optimistic_vs_ucb()

    print("Running Gradient Bandit plot...")
    plot_gradient_bandit()


def save_output_folder():
    path = "mia_rl/outputs/plots/bandit_plots"
    os.makedirs(path, exist_ok=True)
    return path


def main():
    print("Starting K-Armed Bandit experiments...")

    save_output_folder()
    run_all_plots()

    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()