"""
run_fedavg_baseline.py — Run the centralised FedAvg baseline.

Use this to compare against gossip FL results. Both simulators log to
the same results/ directory so plots can be overlaid in analysis.ipynb.

Examples:
    python experiments/run_fedavg_baseline.py
    python experiments/run_fedavg_baseline.py data.num_clients=50
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.config import load_config
    from src.simulator import build_fedavg_simulator

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
    # Force experiment name to distinguish from gossip run
    config.experiment.name = f"fedavg_{config.experiment.name}"

    simulator, metrics = build_fedavg_simulator(config)
    simulator.run()
    metrics.generate_plots()
    metrics.close()
    print(f"FedAvg baseline complete. Logs in {config.logging.log_dir}")


if __name__ == "__main__":
    main()
