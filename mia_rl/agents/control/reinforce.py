from __future__ import annotations

import numpy as np

from mia_rl.features.tictactoe import STATE_FEATURE_DIM

# Number of board cells (= number of possible actions in TicTacToe)
N_ACTIONS: int = 9


class ReinforceAgent:
    """REINFORCE (Monte Carlo policy gradient) for TicTacToe.

    Policy parameterization — softmax over available actions:

        h(s, a)  =  θ[a] · φ(s)          logit for action a
        π(a | s) =  softmax_available(h)  probability over legal moves only

    φ(s) is the 27-dim perspective-relative feature vector from
    ``mia_rl.features.tictactoe.encode_state``:
        - 3 dims per cell × 9 cells = 27 dims
        - [1,0,0] my piece  |  [0,1,0] opponent  |  [0,0,1] empty
    Using a perspective-relative encoding allows the same θ to play
    both X and O during self-play training.

    θ ∈ R^{n_actions × n_features} — one weight vector per board cell.

    REINFORCE update (applied at the end of each episode):

        G_t = Σ_{k≥t} γ^{k−t} r_k          (discounted return from step t)

        θ[a_t]  +=  α · γ^t · G_t · ∇_{θ[a_t]} log π(a_t | s_t)
    """

    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 1.0,
        entropy_beta: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.rng = np.random.default_rng(seed)

        # Initialise weights to small random values to break symmetry
        self.theta = self.rng.normal(loc=0.0, scale=0.01, size=(N_ACTIONS, STATE_FEATURE_DIM))
        self._episode: list[tuple[np.ndarray, int, list[int], float]] = []

    def reset(self) -> None:
        """Clear the internally stored episode trajectory."""
        self._episode.clear()

    def _probs(self, phi: np.ndarray, available: list[int]) -> np.ndarray:
        """Compute the softmax probability distribution over legal actions only.

        Non-legal actions get zero probability.

        Args:
            phi: feature vector of shape (27,)
            available: list of legal action indices (subset of 0..8)

        Returns:
            np.ndarray of shape (len(available),), summing to 1.0.
        """
        # Calcular os logits para todas as ações: h(s, a) = theta[a] @ phi
        logits = self.theta @ phi  # shape (9,)
        
        # Filtrar apenas os logits das ações válidas
        available_logits = logits[available]
        
        # Subtrair o máximo para estabilidade numérica (evita overflow no exp)
        shifted_logits = available_logits - np.max(available_logits)
        
        # Aplicar Softmax
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits)
        
        return probs

    def select_action(self, phi: np.ndarray, available: list[int]) -> int:
        """Sample an action from the policy distribution π(· | s)."""
        probs = self._probs(phi, available)
        return int(self.rng.choice(available, p=probs))

    def greedy_action(self, phi: np.ndarray, available: list[int]) -> int:
        """Select the action with the highest probability (deterministic)."""
        probs = self._probs(phi, available)
        best_idx = int(np.argmax(probs))
        return available[best_idx]

    def store_step(
        self, phi: np.ndarray, action: int, available: list[int], reward: float
    ) -> None:
        """Append a transition step to the internal episode cache."""
        self._episode.append((phi, action, available, reward))

    def update_episode(
        self,
        trajectory: list[tuple[np.ndarray, int, list[int], float]] | None = None,
    ) -> float:
        """Perform the REINFORCE parameter update using a complete trajectory.

        Supports both self-play (external trajectory) and online single-agent.
        Includes optional Shannon entropy regularisation to encourage exploration.

        Returns:
            The mean cross-entropy policy loss over the episode.
        """
        episode = trajectory if trajectory is not None else self._episode
        if not episode:
            return 0.0

        T = len(episode)

        # Backward pass — discounted returns
        returns = np.empty(T)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = episode[t][3] + self.gamma * G
            returns[t] = G

        total_loss = 0.0
        for t, (phi, action, available, _) in enumerate(episode):
            probs = self._probs(phi, available)
            action_idx = available.index(action)
            total_loss -= returns[t] * np.log(probs[action_idx] + 1e-8)

            scale = self.alpha * (self.gamma**t) * returns[t]
            for i, a in enumerate(available):
                if a == action:
                    self.theta[a] += (
                        scale * phi * (1.0 - probs[action_idx])
                    )  # chosen: +φ·(1−π)
                else:
                    self.theta[a] -= scale * phi * probs[i]  # others:  -φ·π

            # Regularização por Entropia (Sutton & Barto Ch. 13 / Mnih et al. 2016)
            if self.entropy_beta > 0.0:
                H = -float(np.sum(probs * np.log(probs + 1e-8)))
                for i, a in enumerate(available):
                    if a == action:
                        # Gradiente da entropia em relação ao logit escolhido
                        self.theta[a] += self.alpha * self.entropy_beta * phi * (-np.log(probs[action_idx] + 1e-8) - 1.0 - H) * probs[action_idx]
                    else:
                        # Gradiente da entropia em relação aos restantes logits
                        self.theta[a] += self.alpha * self.entropy_beta * phi * (-np.log(probs[i] + 1e-8) - 1.0 - H) * probs[i]

        self.reset()
        return total_loss / T