import numpy as np
import matplotlib.pyplot as plt

ARROW = {"U":"↑","D":"↓","L":"←","R":"→","·":"·"}


def plot_grid(env, V, policy=None, title=""):
    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_title(title)
    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)

    ax.set_xticks(np.arange(env.n_cols+1))
    ax.set_yticks(np.arange(env.n_rows+1))
    ax.grid(True)
    ax.invert_yaxis()

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for r in range(env.n_rows):
        for c in range(env.n_cols):

            s = (r,c)

            ax.text(c+0.5, r+0.4, f"{V[s]:.2f}", ha="center")

            if policy is not None:
                ax.text(c+0.5, r+0.75,
                        ARROW.get(policy.get(s,"·")),
                        ha="center", fontsize=16)

    plt.show()