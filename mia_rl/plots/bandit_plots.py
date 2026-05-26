import matplotlib.pyplot as plt
from mia_rl.envs.karmed import KArmedBandit
from mia_rl.agents.bandits.e_greedy import EpsilonGreedy
from mia_rl.agents.bandits.gradient import GradientBandit
from mia_rl.agents.bandits.ucb import UCB
from mia_rl.experiments.bandit_experiment import run_experiment

def plot_epsilon_greedy():
    steps, runs = 1000, 2000
    env = KArmedBandit()

    epsilons = [0, 0.01, 0.1]

    plt.figure()
    for eps in epsilons:
        agent = EpsilonGreedy(epsilon=eps)
        rewards, _ = run_experiment(agent, env, steps, runs)
        plt.plot(rewards, label=f"ε={eps}")

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("ε-greedy comparison")
    plt.show()


def plot_optimistic_vs_ucb():
    steps, runs = 1000, 2000
    env = KArmedBandit()

    agents = {
        "Optimistic greedy": EpsilonGreedy(epsilon=0, optimistic=5),
        "UCB c=2": UCB(c=2),
    }

    plt.figure()
    for name, agent in agents.items():
        rewards, _ = run_experiment(agent, env, steps, runs)
        plt.plot(rewards, label=name)

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("Optimistic vs UCB")
    plt.show()


def plot_gradient_bandit():
    steps, runs = 1000, 2000
    env = KArmedBandit()

    agents = {
        "α=0.1 baseline": GradientBandit(alpha=0.1, baseline=True),
        "α=0.4 baseline": GradientBandit(alpha=0.4, baseline=True),
        "α=0.1 no baseline": GradientBandit(alpha=0.1, baseline=False),
    }

    plt.figure()
    for name, agent in agents.items():
        rewards, _ = run_experiment(agent, env, steps, runs)
        plt.plot(rewards, label=name)

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("Gradient bandit methods")
    plt.show()
