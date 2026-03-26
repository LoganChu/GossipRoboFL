"""
run_gossip.py — Run a single gossip FL experiment.

Examples:
    python experiments/run_gossip.py                          # defaults
    python experiments/run_gossip.py data.num_clients=50
    python experiments/run_gossip.py attack.enabled=true attack.type=sign_flip
    python experiments/run_gossip.py gossip.aggregation=mean
"""
from __future__ import annotations

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.config import load_config
    from src.simulator import build_gossip_simulator
    from src.metrics import generate_topology_gif

    # Parse key=value overrides from CLI (same helper as main.py)
    overrides = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            parts = key.split(".")
            d = overrides
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            try:
                d[parts[-1]] = int(val)
            except ValueError:
                try:
                    d[parts[-1]] = float(val)
                except ValueError:
                    d[parts[-1]] = val if val.lower() not in ("true", "false") else val.lower() == "true"

    config = load_config("configs/default.yaml", overrides)
    simulator, metrics = build_gossip_simulator(config)
    simulator.run()
    metrics.generate_plots()
    metrics.close()

    snap_paths = simulator.get_topology_gif_paths()
    if snap_paths:
        gif_path = os.path.join(config.logging.log_dir, f"{config.experiment.name}_topology.gif")
        generate_topology_gif(snap_paths, gif_path, fps=2)
        print(f"Topology GIF: {gif_path}")


if __name__ == "__main__":
    main()
