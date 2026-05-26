from mia_rl.envs.gridworld import ACTIONS


def uniform_random_policy(env):
    policy = {}

    for s in env.states():
        if env.is_terminal(s):
            policy[s] = {a: 0.0 for a in ACTIONS}
        else:
            policy[s] = {a: 1.0 / len(ACTIONS) for a in ACTIONS}

    return policy