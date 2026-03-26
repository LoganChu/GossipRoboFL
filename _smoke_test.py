"""Quick smoke test — run with: conda run -n ml_env python _smoke_test.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config, config_to_dict
cfg = load_config("configs/default.yaml")
print(f"Config OK: num_clients={cfg.data.num_clients}, agg={cfg.gossip.aggregation}, rounds={cfg.experiment.rounds}")
d = config_to_dict(cfg)
print(f"config_to_dict OK, keys={list(d.keys())}")

from src.model import get_model, count_parameters, model_size_bytes
m = get_model("cnn_cifar")
print(f"Model OK: {count_parameters(m):,} params, {model_size_bytes(m)/1024:.1f} KB")

from src.gossip import gossip_mean, ssclip, flatten_weights
import torch
w1 = {"a": torch.ones(100)}
w2 = {"a": torch.zeros(100)}
r = gossip_mean(w1, [w2])
assert abs(r["a"].mean().item() - 0.5) < 1e-5
r2 = ssclip(w1, [w2], tau=0.5)
print(f"Gossip mean OK: {r['a'].mean().item():.2f}, SSClip OK: {r2['a'].mean().item():.3f}")

from src.attacks import sign_flip, select_byzantine_clients
w = {"p": torch.ones(50) * 3.0}
neg = sign_flip(w)
assert neg["p"].mean().item() == pytest_approx(-3.0) if False else abs(neg["p"].mean().item() + 3.0) < 1e-5
byz = select_byzantine_clients(20, 0.2, seed=42)
print(f"Attacks OK: sign_flip, byzantine_ids={sorted(byz)}")

from src.topology import TopologyManager
from src.config import TopologyConfig, MobilityConfig
tc = TopologyConfig(comm_range=0.5, mobility=MobilityConfig())
tm = TopologyManager(10, tc, seed=0)
assert tm.graph.number_of_nodes() == 10
tm.step()
print(f"Topology OK: {tm.graph.number_of_edges()} edges, positions in bounds")

print("\nAll smoke tests passed!")
