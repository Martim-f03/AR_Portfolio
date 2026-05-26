from mia_rl.experiments.gridworld_dyn_experiment import run
from mia_rl.plots.gridworld_dyn_plots import plot_grid


def main():

    env, V, pi, hist = run()

    print("Policy iteration steps:", len(hist))

    plot_grid(env, V, pi, title="Gridworld Dyn - Optimal Policy")


if __name__ == "__main__":
    main()