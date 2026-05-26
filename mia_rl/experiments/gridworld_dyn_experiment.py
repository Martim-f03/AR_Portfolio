from mia_rl.envs.gridworld_dyn import GridworldDyn
from mia_rl.mdps.policy_iteration_dyn import policy_iteration


def run():
    env = GridworldDyn()
    V, pi, hist = policy_iteration(env)

    return env, V, pi, hist