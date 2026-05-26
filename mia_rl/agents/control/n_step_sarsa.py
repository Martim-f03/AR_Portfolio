from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Episode, Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        # Dicionário temporário para guardar a ação escolhida para cada estado no episódio atual
        self._selected_actions: dict[StateT, ActionT] = {}

    def select_action(self, state: StateT) -> ActionT:
        # Seleção de ação Epsilon-Greedy
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            best_value = max(self.action_value_of(state, act) for act in self.actions)
            best_actions = [act for act in self.actions if self.action_value_of(state, act) == best_value]
            action = self.rng.choice(best_actions)

        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """
        Método obrigatório pela classe abstrata ControlAgent.
        Como o n-step SARSA atualiza por episódio completo, este método pode ficar vazio.
        """
        pass

    def update_episode(self, episode: Episode[StateT, ActionT]) -> None:
        """
        Atualiza os valores Q processando o episódio completo através da lógica n-step.
        """
        transitions = episode.transitions
        T = len(transitions)

        for t in range(T):
            trans_t = transitions[t]
            state_t = trans_t.state
            action_t = trans_t.action

            # 1. Calcular o retorno acumulado de n-passos (recompensas imediatas)
            g_target = 0.0
            end_step = min(t + self.n, T)

            for i in range(t, end_step):
                discount = self.gamma ** (i - t)
                g_target += discount * transitions[i].reward

            # 2. Adicionar o Bootstrap se não atingimos o estado terminal
            if t + self.n < T:
                # O estado no horizonte t+n é o 'state' da transição desse índice
                state_tn = transitions[t + self.n].state
                action_tn = transitions[t + self.n].action
                g_target += (self.gamma ** self.n) * self.action_value_of(state_tn, action_tn)
            elif t + self.n == T:
                # Se t+n atinge exatamente o fim, olhamos para o next_state da última transição
                last_trans = transitions[-1]
                if not last_trans.done and last_trans.next_state is not None:
                    action_tn = self._selected_actions.get(last_trans.next_state)
                    if action_tn is None:
                        action_tn = self.select_action(last_trans.next_state)
                    g_target += (self.gamma ** self.n) * self.action_value_of(last_trans.next_state, action_tn)

            # 3. Atualização Temporal-Difference do par (S_t, A_t)
            current_q = self.action_value_of(state_t, action_t)
            self.Q[(state_t, action_t)] = current_q + self.alpha * (g_target - current_q)

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])