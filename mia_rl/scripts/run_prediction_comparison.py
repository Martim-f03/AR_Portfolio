from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MC, TD(0), and n-step TD prediction on Blackjack.")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes for each algorithm.")
    parser.add_argument("--td-alpha", type=float, default=0.05, help="Step-size for TD(0) and n-step TD.")
    parser.add_argument("--n-steps", type=int, default=4, help="The value of 'n' for n-step TD prediction.")
    parser.add_argument("--threshold", type=int, default=20, help="Policy threshold: hit below this sum.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/blackjack_comparison", help="Directory where plots will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Importação dos agentes existentes e do teu novo agente de n-passos
    from mia_rl.agents.prediction import FirstVisitMonteCarloPrediction, TD0Prediction
    from mia_rl.agents.prediction.n_step_td import NStepTDPrediction  # Ajusta o caminho se guardaste noutro local
    from mia_rl.envs.blackjack import BlackjackEnv
    from mia_rl.experiments.training import train_prediction_agent
    from mia_rl.plots.blackjack import plot_value_difference, plot_value_function
    from mia_rl.policies.blackjack import ThresholdPolicy

    policy = ThresholdPolicy(threshold=args.threshold)

    try:
        # Inicialização dos ambientes com a mesma semente para uma comparação justa
        mc_env = BlackjackEnv(seed=args.seed)
        td0_env = BlackjackEnv(seed=args.seed)
        n_step_env = BlackjackEnv(seed=args.seed)

        # Inicialização dos 3 agentes
        mc_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
        td0_agent = TD0Prediction(alpha=args.td_alpha, gamma=1.0)
        n_step_agent = NStepTDPrediction(n=args.n_steps, alpha=args.td_alpha, gamma=1.0)

        checkpoints = sorted({cp for cp in (1000, 5000, args.episodes) if cp <= args.episodes})

        # 1. Treino do First-Visit Monte Carlo
        print(f"Training First-Visit Monte Carlo for {args.episodes} episodes...")
        mc_history = train_prediction_agent(mc_env, policy, mc_agent, args.episodes, checkpoints=checkpoints)
        final_mc = mc_history[args.episodes]

        # 2. Treino do TD(0)
        print(f"Training TD(0) for {args.episodes} episodes...")
        td0_history = train_prediction_agent(td0_env, policy, td0_agent, args.episodes, checkpoints=checkpoints)
        final_td0 = td0_history[args.episodes]

        # 3. Treino do n-step TD
        print(f"Training {args.n_steps}-step TD for {args.episodes} episodes...")
        n_step_history = train_prediction_agent(n_step_env, policy, n_step_agent, args.episodes, checkpoints=checkpoints)
        final_n_step = n_step_history[args.episodes]

        # --- Geração de Gráficos ---
        print("\nGenerating plots...")
        
        # Função de valor estimada pelo teu novo agente de n-passos
        fig_nstep, _ = plot_value_function(
            final_n_step, title=f"{args.n_steps}-step TD after {args.episodes} episodes", vmin=-1.0, vmax=1.0
        )
        
        # Diferença: n-step TD vs Monte Carlo (Ideal para ver se o n-step aproxima bem o MC)
        fig_diff_mc, _ = plot_value_difference(
            final_n_step, final_mc, title=f"{args.n_steps}-step TD - First-Visit MC", vmin=-0.5, vmax=0.5
        )

        # Diferença: n-step TD vs TD(0) (Para ver o impacto do 'n' maior que 1)
        fig_diff_td0, _ = plot_value_difference(
            final_n_step, final_td0, title=f"{args.n_steps}-step TD - TD(0)", vmin=-0.5, vmax=0.5
        )

        # Guardar resultados
        output_dir = PACKAGE_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_nstep.savefig(output_dir / f"blackjack_{args.n_steps}step_td.png", dpi=150, bbox_inches="tight")
        fig_diff_mc.savefig(output_dir / f"blackjack_{args.n_steps}step_minus_mc.png", dpi=150, bbox_inches="tight")
        fig_diff_td0.savefig(output_dir / f"blackjack_{args.n_steps}step_minus_td0.png", dpi=150, bbox_inches="tight")
        
        print(f"Saved comparison plots to {output_dir}")

        if args.no_show:
            plt.close("all")
        else:
            plt.show()

    except NotImplementedError as exc:
        print(f"\nErro de implementação detetado: {exc}")
        return


if __name__ == "__main__":
    main()