"""
test_topology.py — Unit tests for src/topology.py.
"""
import sys
import os

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MobilityConfig, TopologyConfig
from src.topology import TopologyManager


def make_config(**kwargs) -> TopologyConfig:
    defaults = dict(
        type="random_geometric",
        comm_range=0.5,
        k_nearest=4,
        er_prob=0.3,
        update_every=5,
        mobility=MobilityConfig(enabled=True, step_size=0.05, boundary="reflect"),
    )
    defaults.update(kwargs)
    return TopologyConfig(**defaults)


class TestTopologyManager:
    def test_initial_positions_in_bounds(self):
        tm = TopologyManager(10, make_config(), seed=42)
        for node_id, (x, y) in tm.positions.items():
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_initial_graph_node_count(self):
        N = 15
        tm = TopologyManager(N, make_config(comm_range=0.5), seed=0)
        assert tm.graph.number_of_nodes() == N

    def test_random_geometric_always_connected(self):
        """With large comm_range, graph should almost always be connected."""
        # Use a large range to guarantee connectivity
        tm = TopologyManager(10, make_config(comm_range=0.8), seed=42)
        assert nx.is_connected(tm.graph)

    def test_ensure_connected_works(self):
        """Even with small range, ensure_connected guarantees connectivity."""
        for seed in range(10):
            tm = TopologyManager(20, make_config(comm_range=0.2), seed=seed)
            assert nx.is_connected(tm.graph), f"seed={seed}: graph not connected after ensure_connected"

    def test_knn_graph_connectivity(self):
        tm = TopologyManager(15, make_config(type="knn", k_nearest=3), seed=5)
        assert nx.is_connected(tm.graph)

    def test_erdos_renyi_connectivity(self):
        tm = TopologyManager(12, make_config(type="erdos_renyi", er_prob=0.4), seed=99)
        assert nx.is_connected(tm.graph)

    def test_mobility_reflect_stays_in_bounds(self):
        tm = TopologyManager(20, make_config(
            mobility=MobilityConfig(enabled=True, step_size=0.1, boundary="reflect")
        ), seed=0)
        for _ in range(100):
            tm.step()
        for node_id, (x, y) in tm.positions.items():
            assert 0.0 <= x <= 1.0, f"Node {node_id} x={x} out of bounds"
            assert 0.0 <= y <= 1.0, f"Node {node_id} y={y} out of bounds"

    def test_mobility_wrap_stays_in_bounds(self):
        tm = TopologyManager(20, make_config(
            mobility=MobilityConfig(enabled=True, step_size=0.3, boundary="wrap")
        ), seed=1)
        for _ in range(100):
            tm.step()
        for node_id, (x, y) in tm.positions.items():
            assert 0.0 <= x < 1.0 + 1e-9, f"Node {node_id} x={x}"
            assert 0.0 <= y < 1.0 + 1e-9, f"Node {node_id} y={y}"

    def test_no_mobility_positions_unchanged(self):
        tm = TopologyManager(5, make_config(
            mobility=MobilityConfig(enabled=False, step_size=0.1)
        ), seed=42)
        initial = dict(tm.positions)
        tm.step()
        assert tm.positions == initial

    def test_update_rebuilds_at_correct_rounds(self):
        tm = TopologyManager(10, make_config(update_every=5, comm_range=0.6), seed=0)
        for r in range(20):
            rebuilt = tm.update(r)
            if r % 5 == 0:
                assert rebuilt, f"Expected rebuild at round {r}"
            else:
                assert not rebuilt, f"Unexpected rebuild at round {r}"

    def test_get_neighbors_subset_of_nodes(self):
        tm = TopologyManager(10, make_config(), seed=0)
        for node in range(10):
            nbrs = tm.get_neighbors(node)
            assert all(0 <= n < 10 for n in nbrs)
            assert node not in nbrs

    def test_graph_metrics_keys(self):
        tm = TopologyManager(10, make_config(), seed=0)
        metrics = tm.get_graph_metrics()
        for key in ("is_connected", "num_components", "avg_degree", "clustering", "spectral_gap"):
            assert key in metrics

    def test_spectral_gap_positive_connected(self):
        """Connected graph must have positive spectral gap."""
        tm = TopologyManager(10, make_config(comm_range=0.7), seed=0)
        assert nx.is_connected(tm.graph)
        metrics = tm.get_graph_metrics()
        assert metrics["spectral_gap"] > 0, "Connected graph should have positive spectral gap"

    def test_get_positions_returns_copy(self):
        """Mutating returned positions should not affect internal state."""
        tm = TopologyManager(5, make_config(), seed=0)
        pos = tm.get_positions()
        pos[0] = (99.0, 99.0)
        assert tm.positions[0] != (99.0, 99.0)
