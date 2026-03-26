"""
main.py — Entry point for GossipRoboFL experiments.

Usage:
    # Default 20-client gossip run (SSClip, no attack):
    python main.py

    # Override config values:
    python main.py --config configs/default.yaml --data.num_clients 50
    python main.py --attack.enabled true --attack.type sign_flip --attack.fraction 0.2

    # Run FedAvg baseline:
    python main.py --mode fedavg

    # With Hydra (if installed):
    python main.py +experiment=gossip_n50 +attack=sign_flip_f20

Hydra is used when installed; falls back to argparse + manual YAML loading.
"""
from __future__ import annotations

import argparse
import sys


def _parse_overrides(override_args: list[str]) -> dict:
    """Parse key=value pairs like 'data.num_clients=50' into nested dicts."""
    overrides = {}
    for kv in override_args:
        if "=" not in kv:
            print(f"[warning] Ignoring malformed override: '{kv}' (expected key=value)")
            continue
        key, value = kv.split("=", 1)
        # Parse value: try int, float, bool, then fallback to str
        parsed_val: object
        try:
            parsed_val = int(value)
        except ValueError:
            try:
                parsed_val = float(value)
            except ValueError:
                if value.lower() in ("true", "yes"):
                    parsed_val = True
                elif value.lower() in ("false", "no"):
                    parsed_val = False
                elif value.lower() in ("null", "none"):
                    parsed_val = None
                else:
                    parsed_val = value

        # Build nested dict from dotted key
        parts = key.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = parsed_val

    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="GossipRoboFL: Byzantine-Resilient Gossip Federated Learning"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--mode", choices=["gossip", "fedavg"], default="gossip",
        help="Simulator mode: 'gossip' (decentralised) or 'fedavg' (centralised baseline)",
    )
    parser.add_argument(
        "--ray", action="store_true", default=False,
        help="Use Ray for parallel local training (requires Ray installation)",
    )
    # Capture remaining args as key=value overrides
    args, remaining = parser.parse_known_args()
    overrides = _parse_overrides(remaining)

    from src.config import load_config
    config = load_config(args.config, overrides)

    print(f"[main] Mode={args.mode}, N={config.data.num_clients}, "
          f"dataset={config.data.dataset}, rounds={config.experiment.rounds}, "
          f"agg={config.gossip.aggregation}, "
          f"attack={config.attack.type if config.attack.enabled else 'none'}")

    if args.mode == "gossip":
        from src.simulator import build_gossip_simulator
        simulator, metrics = build_gossip_simulator(config, use_ray=args.ray)
        simulator.run()
        plot_paths = metrics.generate_plots()
        metrics.close()
        print(f"\n[main] Plots saved: {plot_paths}")

        # Generate topology GIF if snapshots were created
        from src.metrics import generate_topology_gif
        snap_paths = simulator.get_topology_gif_paths()
        if snap_paths:
            gif_path = f"{config.logging.log_dir}/{config.experiment.name}_topology.gif"
            generate_topology_gif(snap_paths, gif_path, fps=2)

    else:  # fedavg
        from src.simulator import build_fedavg_simulator
        simulator, metrics = build_fedavg_simulator(config)
        simulator.run()
        plot_paths = metrics.generate_plots()
        metrics.close()
        print(f"\n[main] Plots saved: {plot_paths}")


if __name__ == "__main__":
    main()
