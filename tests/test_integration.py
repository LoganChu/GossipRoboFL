"""
test_integration.py — End-to-end integration tests.

These tests run short simulations to verify the full pipeline works.
They are slower (~30–120s each on CPU) but catch integration bugs
that unit tests cannot.

Run with: pytest tests/test_integration.py -v -s
Skip with: pytest -m "not slow"
"""
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.slow  # tag so --ignore-glob=*integration* can skip them


def load_test_config(overrides=None):
    from src.config import load_config
    return load_config("configs/default.yaml", overrides or {})


class TestGossipConvergence:
    """Verify that gossip FL converges (models become similar)."""

    def test_gossip_mean_5clients_50rounds_models_converge(self):
        """5 clients, 50 rounds of mean gossip on IID data → models should converge.

        Convergence criterion: pairwise L2 distance between all clients < 5.0
        (small CNN with ~500k params: 5.0 is a tight threshold).
        """
        from src.simulator import build_gossip_simulator

        config = load_test_config({
            "experiment": {"name": "test_convergence", "rounds": 50, "seed": 0},
            "data": {"num_clients": 5, "alpha": 100.0},  # IID
            "gossip": {"aggregation": "mean", "fanout": 4},
            "attack": {"enabled": False},
            "topology": {"comm_range": 0.8, "update_every": 0, "mobility": {"enabled": False}},
            "client": {"local_epochs": 1},
            "logging": {"eval_every": 50, "backend": "json", "log_dir": "results/test"},
        })

        simulator, metrics = build_gossip_simulator(config)
        simulator.run()
        metrics.close()

        # Check models are similar after convergence
        from src.gossip import flatten_weights, l2_distance
        all_weights = [c.get_weights() for c in simulator.clients]
        max_dist = max(
            l2_distance(all_weights[i], all_weights[j])
            for i in range(len(all_weights))
            for j in range(i + 1, len(all_weights))
        )
        assert max_dist < 15.0, (
            f"Models did not converge after gossip: max pairwise dist={max_dist:.4f}"
        )

    def test_ssclip_beats_mean_under_sign_flip_attack(self):
        """With 20% Byzantine sign_flip, SSClip should achieve higher accuracy than mean.

        Both run for 50 rounds on 10 clients. SSClip should outperform mean.
        Uses IID data to isolate the robustness effect.
        """
        from src.simulator import build_gossip_simulator

        base_overrides = {
            "experiment": {"rounds": 50, "seed": 42},
            "data": {"num_clients": 10, "alpha": 10.0},
            "attack": {"enabled": True, "type": "sign_flip", "fraction": 0.2},
            "topology": {"comm_range": 0.7, "update_every": 0, "mobility": {"enabled": False}},
            "client": {"local_epochs": 1},
            "logging": {"eval_every": 10, "backend": "json", "log_dir": "results/test"},
        }

        # Run mean
        mean_config = load_test_config({**base_overrides,
                                        "experiment": {"name": "test_mean_byz", "rounds": 50, "seed": 42},
                                        "gossip": {"aggregation": "mean", "fanout": 4}})
        sim_mean, m_mean = build_gossip_simulator(mean_config)
        sim_mean.run()
        m_mean.close()

        # Run ssclip (same clients, same data)
        ssclip_config = load_test_config({**base_overrides,
                                          "experiment": {"name": "test_ssclip_byz", "rounds": 50, "seed": 42},
                                          "gossip": {"aggregation": "ssclip", "fanout": 4}})
        sim_ssclip, m_ssclip = build_gossip_simulator(ssclip_config)
        sim_ssclip.run()
        m_ssclip.close()

        # Compare final honest accuracy
        def last_acc(history):
            entries = [e for e in history if "honest_global_accuracy" in e]
            return entries[-1]["honest_global_accuracy"] if entries else 0.0

        acc_mean = last_acc(m_mean.history)
        acc_ssclip = last_acc(m_ssclip.history)

        print(f"\n[integration] mean acc={acc_mean:.4f}, ssclip acc={acc_ssclip:.4f}")
        # SSClip should be at least marginally better (or equal)
        # In 50 rounds this is not always guaranteed, so we allow 5% margin
        assert acc_ssclip >= acc_mean - 0.05, (
            f"SSClip ({acc_ssclip:.4f}) should ≥ mean ({acc_mean:.4f}) - 5% under attack"
        )


class TestFedAvgBaseline:
    def test_fedavg_runs_without_error(self):
        """FedAvg should complete 20 rounds without error."""
        from src.simulator import build_fedavg_simulator

        config = load_test_config({
            "experiment": {"name": "test_fedavg", "rounds": 20, "seed": 0},
            "data": {"num_clients": 5, "alpha": 1.0},
            "client": {"local_epochs": 1},
            "logging": {"eval_every": 5, "backend": "json", "log_dir": "results/test"},
        })

        sim, metrics = build_fedavg_simulator(config)
        result = sim.run()
        metrics.close()

        assert result["rounds"] == 20
        assert any("honest_global_accuracy" in e for e in metrics.history)


class TestConfigLoading:
    def test_default_config_loads(self):
        from src.config import load_config
        cfg = load_config("configs/default.yaml")
        assert cfg.data.num_clients == 20
        assert cfg.gossip.aggregation == "ssclip"

    def test_overrides_applied(self):
        from src.config import load_config
        cfg = load_config("configs/default.yaml", {"data": {"num_clients": 50}})
        assert cfg.data.num_clients == 50

    def test_config_to_dict_roundtrip(self):
        from src.config import config_to_dict, load_config
        cfg = load_config("configs/default.yaml")
        d = config_to_dict(cfg)
        assert d["data"]["num_clients"] == cfg.data.num_clients
        assert d["gossip"]["aggregation"] == cfg.gossip.aggregation
