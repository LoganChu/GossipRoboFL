"""
ablation_heterogeneity.py — Data heterogeneity and client heterogeneity ablation.

Two ablations in one script:

  1. Non-IID degree: vary Dirichlet alpha ∈ {0.1, 0.5, 1.0, 10.0}
     → How does data skew affect gossip convergence?

  2. Client heterogeneity: homogeneous vs heterogeneous local epochs
     → Impact of straggler robots on swarm performance?

Produces:
  results/ablation_noniid_convergence.png
  results/ablation_heterogeneity_comparison.png
  results/ablation_heterogeneity_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALPHA_VALUES = [0.1, 0.5, 1.0, 10.0]
ROUNDS = 200
NUM_CLIENTS = 20


def run_alpha(alpha: float, rounds: int, num_clients: int, heterogeneous: bool = False) -> list[dict]:
    from src.config import load_config
    from src.simulator import build_gossip_simulator

    tag = f"noniid_a{alpha}_het{heterogeneous}"
    overrides = {
        "experiment": {"name": tag, "rounds": rounds},
        "data": {"num_clients": num_clients, "alpha": alpha},
        "client": {"heterogeneous": heterogeneous},
        "gossip": {"aggregation": "ssclip"},
        "attack": {"enabled": False},
        "logging": {"backend": "json", "eval_every": max(1, rounds // 20)},
    }
    config = load_config("configs/default.yaml", overrides)
    simulator, metrics = build_gossip_simulator(config)
    simulator.run()
    metrics.close()
    return metrics.history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    args = parser.parse_args()

    # --- Ablation 1: Non-IID degree ---
    print("\n=== ABLATION 1: Non-IID Degree (varying alpha) ===")
    alpha_histories: dict[float, list[dict]] = {}
    alpha_summary: list[dict] = []

    for alpha in ALPHA_VALUES:
        print(f"\n>>> alpha={alpha}")
        t0 = time.time()
        history = run_alpha(alpha, args.rounds, args.num_clients, heterogeneous=False)
        elapsed = time.time() - t0
        alpha_histories[alpha] = history

        acc_entries = [e for e in history if "honest_global_accuracy" in e]
        final_acc = acc_entries[-1]["honest_global_accuracy"] if acc_entries else 0.0
        print(f"    final_acc={final_acc:.4f} ({elapsed:.1f}s)")
        alpha_summary.append({"alpha": alpha, "final_accuracy": final_acc})

    # --- Ablation 2: Client heterogeneity ---
    print("\n=== ABLATION 2: Client Heterogeneity ===")
    het_histories: dict[str, list[dict]] = {}

    for het in [False, True]:
        label = "heterogeneous" if het else "homogeneous"
        print(f"\n>>> {label} clients")
        t0 = time.time()
        history = run_alpha(0.5, args.rounds, args.num_clients, heterogeneous=het)
        elapsed = time.time() - t0
        het_histories[label] = history
        acc_entries = [e for e in history if "honest_global_accuracy" in e]
        final_acc = acc_entries[-1]["honest_global_accuracy"] if acc_entries else 0.0
        print(f"    final_acc={final_acc:.4f} ({elapsed:.1f}s)")

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {"alpha_ablation": alpha_summary, "heterogeneity_ablation": {
        k: ([e for e in v if "honest_global_accuracy" in e][-1]["honest_global_accuracy"]
            if any("honest_global_accuracy" in e for e in v) else 0.0)
        for k, v in het_histories.items()
    }}
    with open("results/ablation_heterogeneity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot 1: non-IID convergence curves
    from src.metrics import plot_non_iid_impact
    fig = plot_non_iid_impact(
        alpha_histories,
        save_path="results/ablation_noniid_convergence.png",
    )
    plt.close(fig)

    # Plot 2: alpha vs final accuracy bar chart
    fig2, ax = plt.subplots(figsize=(8, 5))
    alphas = [s["alpha"] for s in alpha_summary]
    accs = [s["final_accuracy"] for s in alpha_summary]
    bars = ax.bar([str(a) for a in alphas], accs, color="#3498db", edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
    ax.set_xlabel("Dirichlet α (higher = more IID)", fontsize=12)
    ax.set_ylabel("Final Test Accuracy", fontsize=12)
    ax.set_title(f"Final Accuracy vs Non-IID Degree (N={args.num_clients})", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig2.savefig("results/ablation_noniid_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: heterogeneity comparison
    from src.metrics import plot_convergence_curves
    fig3 = plot_convergence_curves(
        het_histories,
        metric="honest_global_accuracy",
        save_path="results/ablation_heterogeneity_comparison.png",
        title=f"Client Heterogeneity Impact (α=0.5, N={args.num_clients})",
    )
    plt.close(fig3)

    print("\nAll results saved to results/ablation_*.png|json")


if __name__ == "__main__":
    main()
