"""
ablation_topology.py — Topology density ablation study.

Sweeps communication range r ∈ {0.2, 0.3, 0.4, 0.5, 0.6} for the
random geometric graph topology. Measures:
  - Convergence speed (rounds to first exceed 60% accuracy)
  - Final accuracy
  - Average spectral gap (algebraic connectivity)

Produces:
  results/ablation_topology_convergence.png
  results/ablation_topology_results.json
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

COMM_RANGES = [0.2, 0.3, 0.4, 0.5, 0.6]
ROUNDS = 200
NUM_CLIENTS = 20
TARGET_ACC = 0.60  # rounds-to-target threshold


def run_single(comm_range: float, rounds: int, num_clients: int) -> list[dict]:
    """Run one topology experiment; return full history."""
    from src.config import load_config
    from src.simulator import build_gossip_simulator

    overrides = {
        "experiment": {"name": f"topo_abl_r{int(comm_range*100)}", "rounds": rounds},
        "data": {"num_clients": num_clients},
        "topology": {"comm_range": comm_range, "update_every": 5},
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

    all_histories: dict[float, list[dict]] = {}
    summary: list[dict] = []

    for r in COMM_RANGES:
        print(f"\n>>> comm_range={r}")
        t0 = time.time()
        history = run_single(r, args.rounds, args.num_clients)
        elapsed = time.time() - t0

        all_histories[r] = history
        acc_entries = [e for e in history if "honest_global_accuracy" in e]
        final_acc = acc_entries[-1]["honest_global_accuracy"] if acc_entries else 0.0

        # Rounds to target
        rounds_to_target = args.rounds
        for e in acc_entries:
            if e["honest_global_accuracy"] >= TARGET_ACC:
                rounds_to_target = e["round"]
                break

        # Mean spectral gap
        gaps = [e.get("graph_spectral_gap", 0) for e in history]
        mean_gap = float(np.mean(gaps)) if gaps else 0.0

        print(f"    final_acc={final_acc:.4f}, rounds_to_{int(TARGET_ACC*100)}%={rounds_to_target}, "
              f"mean_spectral_gap={mean_gap:.4f} ({elapsed:.1f}s)")

        summary.append({
            "comm_range": r,
            "final_accuracy": final_acc,
            "rounds_to_target": rounds_to_target,
            "mean_spectral_gap": mean_gap,
        })

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_topology_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot convergence curves by topology
    from src.metrics import plot_convergence_curves
    histories_named = {f"r={r}": all_histories[r] for r in COMM_RANGES}
    fig = plot_convergence_curves(
        histories_named,
        metric="honest_global_accuracy",
        save_path="results/ablation_topology_convergence.png",
        title=f"Topology Density Ablation (N={args.num_clients})",
    )
    plt.close(fig)

    # Plot spectral gap vs final accuracy
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rs = [s["comm_range"] for s in summary]
    final_accs = [s["final_accuracy"] for s in summary]
    rtt = [s["rounds_to_target"] for s in summary]
    gaps = [s["mean_spectral_gap"] for s in summary]

    ax1.plot(rs, final_accs, marker="o", color="#3498db", linewidth=2)
    ax1.set_xlabel("Communication Range r", fontsize=11)
    ax1.set_ylabel("Final Test Accuracy", fontsize=11)
    ax1.set_title("Final Accuracy vs Topology Density")
    ax1.grid(alpha=0.3)

    ax2.plot(rs, rtt, marker="s", color="#e74c3c", linewidth=2)
    ax2.set_xlabel("Communication Range r", fontsize=11)
    ax2.set_ylabel(f"Rounds to {int(TARGET_ACC*100)}% Accuracy", fontsize=11)
    ax2.set_title("Convergence Speed vs Topology Density")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig2.savefig("results/ablation_topology_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("\nAll results saved to results/ablation_topology_*.png|json")


if __name__ == "__main__":
    main()
