from mia_rl.mdps.gridworld_dyn_mdp import greedy_action

ACTIONS = ["U", "D", "L", "R"]

def policy_improvement(env, V, gamma=0.9):
    new_policy = {}

    for s in env.states():

        if env.is_terminal(s):
            new_policy[s] = "·"
            continue

        best_a = greedy_action(env, V, s, gamma)
        new_policy[s] = best_a

    return new_policy