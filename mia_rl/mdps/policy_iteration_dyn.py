from mia_rl.mdps.policy_evaluations_dyn import policy_evaluation
from mia_rl.mdps.policy_improvement_dyn import policy_improvement

ACTIONS = ["U", "D", "L", "R"]


def policy_iteration(env, gamma=0.9, theta=1e-8, max_outer=100):

    # stochastic policy
    policy = {
        s: {a: 1.0 / len(ACTIONS) for a in ACTIONS}
        for s in env.states()
    }

    history = []

    for i in range(max_outer):

        # 1) evaluate stochastic policy
        V = policy_evaluation(env, policy, gamma, theta)

        # 2) improve → deterministic policy
        new_actions = policy_improvement(env, V, gamma)

        history.append((i, V.copy(), new_actions.copy()))

        # 3) check stability
        stable = True
        for s in env.states():
            if not env.is_terminal(s):
                if new_actions[s] != max(policy[s], key=policy[s].get):
                    stable = False
                    break

        # 4) convert deterministic → stochastic
        policy = {
            s: {a: (1.0 if a == new_actions[s] else 0.0)
                for a in ACTIONS}
            for s in env.states()
        }

        if stable:
            return V, new_actions, history

    return V, new_actions, history