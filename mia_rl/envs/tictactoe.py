from __future__ import annotations

import random
from typing import Optional

from mia_rl.core.base import Environment

# ── Type aliases ────────────────────────────────────────────────────────────
# The board is a 9-tuple of ints (one per cell, row-major):
#   0 = empty, 1 = player X, -1 = player O
# Actions are integers 0-8 identifying the cell to mark.
TicTacToeState  = tuple[int, ...]   # length-9
TicTacToeAction = int               # 0 … 8

# Indices of every winning line (rows, columns, diagonals)
_WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)


def _winner(board: TicTacToeState) -> int:
    """Return 1 if X wins, -1 if O wins, 0 otherwise."""
    for i, j, k in _WIN_LINES:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


class TicTacToeEnv(Environment[TicTacToeState, TicTacToeAction]):
    def __init__(self) -> None:
        """Initialize an empty Tic-Tac-Toe game board.

        By standard convention, player +1 (X) moves first.
        """
        self.board: TicTacToeState = (0,) * 9
        self.current_player: int = 1  # 1 para X, -1 para O

    def reset(self) -> TicTacToeState:
        """Reset the game to the starting condition."""
        self.board = (0,) * 9
        self.current_player = 1
        return self.board

    def available_actions(self, state: TicTacToeState) -> list[TicTacToeAction]:
        """Return all legal action integers (0-8) for the given state.

        An action is legal if and only if that board cell is empty (0).
        """
        return [idx for idx, cell in enumerate(state) if cell == 0]

    def step(self, action: TicTacToeAction) -> tuple[TicTacToeState, float, bool]:
        """Execute one cell-marking action by the current player.

        Raises `ValueError` if the cell is already occupied.
        """
        # 1. Validar se a célula está vazia
        if self.board[action] != 0:
            raise ValueError(f"Ação Inválida: Célula {action} já está ocupada.")

        # 2. Criar o novo tabuleiro alterando apenas o índice da ação
        board_list = list(self.board)
        board_list[action] = self.current_player
        new_board = tuple(board_list)

        # 3. Validar se há um vencedor
        winner = _winner(new_board)

        # 4. Determinar se o jogo acabou (Vitória de alguém ou sem espaços livres)
        has_empty_cells = 0 in new_board
        done = (winner != 0) or (not has_empty_cells)

        # 5. Calcular a recompensa para quem jogou neste passo
        # Ganha +1 se venceu após colocar a peça, caso contrário recebe 0
        reward = 1.0 if (winner == self.current_player) else 0.0

        # 6. Atualizar os atributos internos do ambiente
        self.board = new_board
        self.current_player = -self.current_player  # Inverte o jogador (1 ↔ -1)

        # 7. Retornar a transição
        return new_board, reward, done

    def render(self, state: TicTacToeState | None = None) -> None:
        """Print a human-readable board to stdout."""
        board_to_show = state if state is not None else self.board
        
        symbols = []
        for idx, cell in enumerate(board_to_show):
            if cell == 1:
                symbols.append("X")
            elif cell == -1:
                symbols.append("O")
            else:
                symbols.append(str(idx + 1))  # Mostra números de 1 a 9 se estiver livre

        print(f"\n {symbols[0]} | {symbols[1]} | {symbols[2]} ")
        print("---+---+---")
        print(f" {symbols[3]} | {symbols[4]} | {symbols[5]} ")
        print("---+---+---")
        print(f" {symbols[6]} | {symbols[7]} | {symbols[8]} \n")
    def is_terminal(self, state: TicTacToeState) -> bool:
        """Retorna True se houver um vencedor ou se o tabuleiro estiver cheio."""
        # Se já tiveres a função utilitária _winner importada ou definida:
        if _winner(state) != 0:
            return True
        # Se não houver zeros (espaços vazios), o jogo acabou em empate
        return 0 not in state
