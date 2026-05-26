import random
from mia_rl.envs.tictactoe import TicTacToeEnv

env = TicTacToeEnv()
state = env.reset()
done = False

print("Início do Jogo!")
env.render()

while not done:
    actions = env.available_actions(state)
    action = random.choice(actions)
    
    print(f"O jogador { 'X' if env.current_player == 1 else 'O' } escolheu a casa {action + 1}")
    state, reward, done = env.step(action)
    env.render()
    
    if done:
        from mia_rl.envs.tictactoe import _winner
        winner = _winner(state)
        if winner == 1:
            print("Fim do jogo: Vitória do X!")
        elif winner == -1:
            print("Fim do jogo: Vitória do O!")
        else:
            print("Fim do jogo: Empate!")