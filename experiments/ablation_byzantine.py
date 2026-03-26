"""
ablation_byzantine.py — Byzantine robustness ablation study.

Sweeps Byzantine fraction f ∈ {0, 0.1, 0.2, 0.3} across three aggregation
methods {mean, clipped_gossip, ssclip} with sign_flip attack.

Produces:
  results/ablation_byzantine_accuracy_vs_f.png
  results/ablation_byzantine_results.json

Runtime: ~N_methods × N_fractions × rounds local-SGD epochs.
For quick iteration use: --rounds 50 --num_clients 10
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

BYZ_FRACTIONS = [0.0, 0.1, 0.2, 0.3]
METHODS = ["mean", "clipped_gossip", "ssclip"]
ATTACK_TYPE = "sign_flip"
ROUNDS = 200         # reduce for quick test
NUM_CLIENTS = 20


def run_single(method: str, fraction: float, rounds: int, num_clients: int) -> float:
    """Run one (method, fraction) experiment and return final honest accuracy."""
    from src.config import load_config
    from src.simulator import build_gossip_simulator

    overrides = {
        "experiment": {"name": f"byz_abl_{method}_f{int(fraction*100)}", "rounds": rounds},
        "data": {"num_clients": num_clients},
        "gossip": {"aggregation": method},
        "attack": {
            "enabled": fraction > 0,
            "type": ATTACK_TYPE,
            "fraction": fraction,
        },
        "logging": {
            "backend": "json",
            "eval_every": max(1, rounds // 20),
        },
    }
    config = load_config("configs/default.yaml", overrides)
    simulator, metrics = build_gossip_simulator(config)
    simulator.run()
    metrics.close()

    # Extract final honest accuracy
    history = metrics.history
    acc_entries = [e for e in history if "honest_global_accuracy" in e]
    if acc_entries:
        return float(acc_entries[-1]["honest_global_accuracy"])
    return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    args = parser.parse_args()

    results: dict[float, dict[str, float]] = {f: {} for f in BYZ_FRACTIONS}

    for method in METHODS:
        for fraction in BYZ_FRACTIONS:
            print(f"\n>>> Method={method}, f={fraction:.1f}")
            t0 = time.time()
            acc = run_single(method, fraction, args.rounds, args.num_clients)
            elapsed = time.time() - t0
            results[fraction][method] = acc
            print(f"    Final accuracy: {acc:.4f} ({elapsed:.1f}s)")

    # Save raw results
    os.makedirs("results", exist_ok=True)
    out_path = "results/ablation_byzantine_results.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate plot
    from src.metrics import plot_accuracy_vs_byzantine_fraction
    fig = plot_accuracy_vs_byzantine_fraction(
        results,
        save_path="results/ablation_byzantine_accuracy_vs_f.png",
        title=f"Byzantine Robustness — {ATTACK_TYPE} attack, N={args.num_clients}",
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("Plot saved to results/ablation_byzantine_accuracy_vs_f.png")

    # Print summary table
    print("\n=== RESULTS TABLE ===")
    header = f"{'f':>6} | " + " | ".join(f"{m:>15}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for f in BYZ_FRACTIONS:
        row = f"{f:>6.1f} | " + " | ".join(
            f"{results[f].get(m, float('nan')):>15.4f}" for m in METHODS
        )
        print(row)


if __name__ == "__main__":
    main()
