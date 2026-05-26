import numpy as np

def run_experiment(agent, env, steps=1000, runs=2000):
    rewards = np.zeros((runs, steps))
    optimal = np.zeros((runs, steps))

    for r in range(runs):
        env.reset()
        agent.reset()

        for t in range(steps):
            action = agent.select_action()
            reward = env.step(action)
            agent.update(action, reward)

            rewards[r, t] = reward
            optimal[r, t] = (action == env.optimal_action)

    return rewards.mean(axis=0), optimal.mean(axis=0)
