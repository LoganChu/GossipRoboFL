"""Quick smoke test: 2 rounds to verify GPU is used."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.simulator import build_gossip_simulator

config = load_config("configs/default.yaml", {
    "experiment": {"rounds": 2, "name": "gpu_smoketest"},
    "logging": {"eval_every": 2, "save_model_every": 0},
})

simulator, metrics = build_gossip_simulator(config)
print(f"Device in use: {simulator.clients[0].device}")
simulator.run()
metrics.close()
print("Smoke test PASSED")
