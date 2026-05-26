from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuração de caminhos para garantir que o Python encontra o pacote mia_rl
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mia_rl.agents.control.reinforce import ReinforceAgent
from mia_rl.envs.tictactoe import TicTacToeEnv
from mia_rl.experiments.reinforce_tictactoe import train, evaluate_vs_random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de ReinforceAgent em Tic-Tac-Toe e Torneio contra oponente Aleatório."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50000,
        help="Número total de episódios para o treino.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=2000,
        help="Frequência (em episódios) com que se avalia o agente.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=500,
        help="Número de jogos em cada checkpoint de avaliação.",
    )
    parser.add_argument(
        "--initial-alpha",
        type=float,
        default=0.01,
        help="Taxa de aprendizagem (alpha) inicial.",
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.001,
        help="Taxa de aprendizagem mínima após o decaimento linear.",
    )
    parser.add_argument(
        "--entropy-beta",
        type=float,
        default=0.01,
        help="Coeficiente de regularização por entropia.",
    )
    parser.add_argument(
        "--random-fraction",
        type=float,
        default=0.3,
        help="Fração de jogos de treino jogados contra um oponente aleatório (vs self-play).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tictactoe_reinforce",
        help="Diretório onde os gráficos de treino serão guardados.",
    )
    return parser.parse_args()


def plot_training_results(history: dict, output_path: Path) -> None:
    """Gera e guarda gráficos com a evolução do treino do agente."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Gráfico da Função de Perda (Loss) suavizada
    losses = history["losses"]
    # Média móvel simples para suavizar a curva de erro
    window = 500
    if len(losses) >= window:
        smoothed_losses = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax1.plot(smoothed_losses, color="crimson", label="Loss (Média Móvel)")
    else:
        ax1.plot(losses, color="crimson", label="Loss por Episódio")
    ax1.set_title("Evolução da Policy Loss do REINFORCE")
    ax1.set_xlabel("Episódios de Treino")
    ax1.set_ylabel("Loss Média")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Gráfico das Taxas de Vitória e Empate nos Checkpoints
    checkpoints = history["eval_checkpoints"]
    ax2.plot(checkpoints, history["win_rates_as_x"], "o-", label="Vitórias como X", color="tab:blue")
    ax2.plot(checkpoints, history["win_rates_as_o"], "s-", label="Vitórias como O", color="tab:cyan")
    ax2.plot(checkpoints, history["draw_rates_as_x"], "--", label="Empates como X", color="tab:gray")
    
    ax2.set_title("Performance Intermédia vs Oponente Aleatório")
    ax2.set_xlabel("Episódios de Treino")
    ax2.set_ylabel("Proporção")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path / "tictactoe_training_curves.png", dpi=150)
    plt.close(fig)
    print(f"\n[INFO] Gráficos de treino guardados com sucesso em: {output_path}/")


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("      INICIANDO TREINO DO AGENTE REINFORCE EM TIC-TAC-TOE")
    print("=" * 70)
    print(f" -> Episódios Totais: {args.episodes}")
    print(f" -> Alpha Inicial: {args.initial_alpha} (Decaimento até: {args.min_alpha})")
    print(f" -> Coeficiente de Entropia (Beta): {args.entropy_beta}")
    print(f" -> Mistura Antagonista (Fraç. Aleatória): {args.random_fraction * 100}%")
    print("-" * 70)

    # 1. Inicializar o Agente REINFORCE
    agent = ReinforceAgent(
        alpha=args.initial_alpha,
        gamma=1.0,
        entropy_beta=args.entropy_beta,
        seed=args.seed,
    )

    # 2. Executar o Ciclo de Treino Central
    history = train(
        agent=agent,
        num_episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_games,
        random_opp_fraction=args.random_fraction,
        seed=args.seed,
        initial_alpha=args.initial_alpha,
        min_alpha=args.min_alpha,
    )

    # 3. Criar a pasta de outputs e gerar os gráficos de treino
    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_results(history, output_dir)

    # 4. TORNEIO FINAL DE AVALIAÇÃO (Postura Determinística / Greedy)
    print("\n" + "=" * 70)
    print("         TORNEIO FINAL DO AGENTE CONCLUÍDO VS ALEATÓRIO")
    print("=" * 70)
    env = TicTacToeEnv()
    
    # Avaliação exaustiva (1000 jogos por cada lado para robustez estatística)
    n_tournament_games = 1000
    
    print(f"A jogar {n_tournament_games} partidas como Jogador X (Começa primeiro)...")
    wins_x, draws_x, losses_x = evaluate_vs_random(
        env, agent, n_games=n_tournament_games, as_player=1
    )
    
    print(f"A jogar {n_tournament_games} partidas como Jogador O (Joga em segundo)...")
    wins_o, draws_o, losses_o = evaluate_vs_random(
        env, agent, n_games=n_tournament_games, as_player=-1
    )

    print("-" * 70)
    print(f"RESULTADOS COMO JOGADOR X (+1):")
    print(f"  - Taxa de Vitória: {wins_x * 100:.1f}%")
    print(f"  - Taxa de Empate:  {draws_x * 100:.1f}%")
    print(f"  - Taxa de Derrota: {losses_x * 100:.1f}%")
    print("-" * 70)
    print(f"RESULTADOS COMO JOGADOR O (-1):")
    print(f"  - Taxa de Vitória: {wins_o * 100:.1f}%")
    print(f"  - Taxa de Empate:  {draws_o * 100:.1f}%")
    print(f"  - Taxa de Derrota: {losses_o * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()