"""
ablation_encounter.py — Compare synchronous vs encounter-based gossip.

Runs 5 conditions, all with SSClip aggregation and no Byzantine attack:

  1. sync          — existing GossipSimulator (forced connectivity, global barrier)
  2. encounter_p00 — EncounterGossipSimulator, drop=0.0 (perfect wireless)
  3. encounter_p10 — EncounterGossipSimulator, drop=0.1
  4. encounter_p30 — EncounterGossipSimulator, drop=0.3
  5. encounter_async — EncounterGossipSimulator, drop=0, async training

Output files (in results/):
  ablation_encounter_convergence.png   — accuracy vs round, all 5 conditions
  ablation_encounter_isolation.png     — mean isolation steps over time
  ablation_encounter_components.png    — number of graph components over time
  ablation_encounter_results.json      — summary table

Usage:
    conda run -n ml_env python experiments/ablation_encounter.py
    conda run -n ml_env python experiments/ablation_encounter.py --rounds 100 --num_clients 20
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.encounter_gossip import build_encounter_simulator
from src.simulator import build_gossip_simulator


# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------

def make_conditions(rounds: int, num_clients: int) -> list[dict]:
    return [
        {
            "name": "sync",
            "label": "Sync Gossip (baseline)",
            "mode": "sync",
            "overrides": {
                "experiment": {"name": "enc_abl_sync", "rounds": rounds},
                "data": {"num_clients": num_clients},
                "gossip": {"aggregation": "ssclip"},
                "attack": {"enabled": False},
            },
        },
        {
            "name": "encounter_p00",
            "label": "Encounter drop=0%",
            "mode": "encounter",
            "overrides": {
                "experiment": {"name": "enc_abl_p00", "rounds": rounds},
                "data": {"num_clients": num_clients},
                "gossip": {"aggregation": "ssclip", "steps_per_round": 1,
                           "train_every_steps": 1, "async_train": False},
                "topology": {"drop_prob": 0.0},
                "attack": {"enabled": False},
            },
        },
        {
            "name": "encounter_p10",
            "label": "Encounter drop=10%",
            "mode": "encounter",
            "overrides": {
                "experiment": {"name": "enc_abl_p10", "rounds": rounds},
                "data": {"num_clients": num_clients},
                "gossip": {"aggregation": "ssclip", "steps_per_round": 1,
                           "train_every_steps": 1, "async_train": False},
                "topology": {"drop_prob": 0.1},
                "attack": {"enabled": False},
            },
        },
        {
            "name": "encounter_p30",
            "label": "Encounter drop=30%",
            "mode": "encounter",
            "overrides": {
                "experiment": {"name": "enc_abl_p30", "rounds": rounds},
                "data": {"num_clients": num_clients},
                "gossip": {"aggregation": "ssclip", "steps_per_round": 1,
                           "train_every_steps": 1, "async_train": False},
                "topology": {"drop_prob": 0.3},
                "attack": {"enabled": False},
            },
        },
        {
            "name": "encounter_async",
            "label": "Encounter async training",
            "mode": "encounter",
            "overrides": {
                "experiment": {"name": "enc_abl_async", "rounds": rounds},
                "data": {"num_clients": num_clients},
                "gossip": {"aggregation": "ssclip", "steps_per_round": 1,
                           "train_every_steps": 3, "async_train": True},
                "topology": {"drop_prob": 0.0},
                "attack": {"enabled": False},
            },
        },
    ]


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------

def run_condition(condition: dict, config_path: str) -> dict:
    config = load_config(config_path, overrides=condition["overrides"])

    if condition["mode"] == "sync":
        simulator, metrics = build_gossip_simulator(config)
    else:
        simulator, metrics = build_encounter_simulator(config)

    simulator.run()

    # history is list[dict], each entry has "round" key
    history: list[dict] = metrics.history if hasattr(metrics, "history") else []

    rounds = [e["round"] for e in history]
    acc_curve = [
        e.get("honest_global_accuracy", e.get("global_accuracy", 0.0))
        for e in history
    ]
    isolation_curve = [e.get("mean_isolation_steps", 0.0) for e in history]
    components_curve = [e.get("graph_num_components", 1.0) for e in history]

    final_acc = acc_curve[-1] if acc_curve else 0.0
    print(f"  [{condition['name']}] final_accuracy={final_acc:.4f}")

    return {
        "name": condition["name"],
        "label": condition["label"],
        "rounds": rounds,
        "acc_curve": acc_curve,
        "isolation_curve": isolation_curve,
        "components_curve": components_curve,
        "final_accuracy": final_acc,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = ["#2c3e50", "#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
LINESTYLES = ["-", "--", "--", "--", ":"]


def plot_convergence(results: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        if r["rounds"] and r["acc_curve"]:
            ax.plot(r["rounds"], r["acc_curve"],
                    color=COLORS[i], linestyle=LINESTYLES[i],
                    linewidth=2, label=r["label"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Honest Global Accuracy")
    ax.set_title("Sync vs Encounter-Based Gossip: Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_isolation(results: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(results):
        if r["name"] == "sync":
            continue  # isolation not tracked for sync
        if r["rounds"] and any(x > 0 for x in r["isolation_curve"]):
            ax.plot(r["rounds"], r["isolation_curve"],
                    color=COLORS[i], linestyle=LINESTYLES[i],
                    linewidth=2, label=r["label"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Isolation Steps")
    ax.set_title("Robot Isolation Duration (steps without any neighbor)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_components(results: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(results):
        if r["name"] == "sync":
            continue
        if r["rounds"] and r["components_curve"]:
            ax.plot(r["rounds"], r["components_curve"],
                    color=COLORS[i], linestyle=LINESTYLES[i],
                    linewidth=2, label=r["label"])
    ax.axhline(1, color="gray", linewidth=1, linestyle=":", alpha=0.6, label="Fully connected")
    ax.set_xlabel("Round")
    ax.set_ylabel("Connected Components")
    ax.set_title("Swarm Partition Frequency (components > 1 = partition)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Encounter gossip ablation")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out_dir", type=str, default="results/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    conditions = make_conditions(args.rounds, args.num_clients)

    print(f"\n=== Encounter Ablation: {len(conditions)} conditions, "
          f"{args.rounds} rounds, N={args.num_clients} ===\n")

    all_results = []
    for cond in conditions:
        print(f"Running: {cond['name']}")
        result = run_condition(cond, args.config)
        all_results.append(result)

    # Plots
    plot_convergence(all_results,
                     os.path.join(args.out_dir, "ablation_encounter_convergence.png"))
    plot_isolation(all_results,
                   os.path.join(args.out_dir, "ablation_encounter_isolation.png"))
    plot_components(all_results,
                    os.path.join(args.out_dir, "ablation_encounter_components.png"))

    # Summary JSON
    summary = [
        {"name": r["name"], "label": r["label"], "final_accuracy": r["final_accuracy"]}
        for r in all_results
    ]
    summary_path = os.path.join(args.out_dir, "ablation_encounter_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    print("\n=== Final Accuracies ===")
    for r in all_results:
        print(f"  {r['label']:<40} {r['final_accuracy']:.4f}")


if __name__ == "__main__":
    main()
