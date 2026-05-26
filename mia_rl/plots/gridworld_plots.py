import matplotlib.pyplot as plt
import numpy as np


ARROW = {"U":"↑", "D":"↓", "L":"←", "R":"→", "·":"·"}


def plot_grid_values_and_policy(env, V, policy=None, title=""):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title(title)
    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks(np.arange(env.n_cols + 1))
    ax.set_yticks(np.arange(env.n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for (r, c) in env.terminal_states:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, alpha=0.15))

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = (r, c)

            ax.text(c + 0.5, r + 0.45, f"{V[r,c]:.2f}",
                    ha="center", va="center")

            if policy is not None:
                a = policy[s] if s in policy else "·"
                ax.text(c + 0.5, r + 0.75, ARROW.get(a, "·"),
                        ha="center", va="center", fontsize=16)

    plt.show()