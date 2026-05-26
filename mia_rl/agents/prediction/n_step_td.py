from __future__ import annotations

from collections import defaultdict
from typing import Optional

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class NStepTDPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, n: int = 3, alpha: float = 0.05, gamma: float = 1.0):
        self.n = n
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        transitions = episode.transitions
        T = len(transitions)

        # Iteramos por cada estado visitado no episódio para calcular o seu alvo de n-passos
        for t in range(T):
            state_to_update = transitions[t].state
            
            # 1. Calcular a componente dos recompensas imediatas até min(t + n, T)
            g_target = 0.0
            end_step = min(t + self.n, T)
            
            for i in range(t, end_step):
                # O desconto acumulado para a recompensa no passo 'i' em relação ao passo 't'
                discount = self.gamma ** (i - t)
                g_target += discount * transitions[i].reward

            # 2. Se o horizonte n-passos não atingiu o fim do episódio, bootstrap do estado futuro
            if t + self.n < T:
                # O estado em t + n é o 'state' da transição no índice (t + n)
                bootstrap_state = transitions[t + self.n].state
                g_target += (self.gamma ** self.n) * self.V[bootstrap_state]
            elif t + self.n == T:
                # Se t + n calha exatamente no fim do episódio, fazemos bootstrap do 'next_state' do último passo
                last_transition = transitions[-1]
                if not last_transition.done and last_transition.next_state is not None:
                    g_target += (self.gamma ** self.n) * self.V[last_transition.next_state]
                # Se last_transition.done for True, o valor terminal é 0.0 (já garantido por g_target)

            # 3. Atualização de base Temporal-Difference
            self.V[state_to_update] += self.alpha * (g_target - self.V[state_to_update])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])