"""
test_gossip.py — Unit tests for src/gossip.py.

Tests cover:
  - flatten/unflatten roundtrip
  - gossip_mean correctness
  - ClippedGossip bounds
  - SSClip drift guarantee (output within tau of own weights)
  - tau auto-selection
  - Byzantine resilience: SSClip vs mean under sign_flip attack
"""
import copy
import math
import sys
import os

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gossip import (
    clipped_gossip,
    compute_tau_auto,
    flatten_weights,
    gossip_mean,
    l2_distance,
    pairwise_distances,
    ssclip,
    unflatten_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_weights(seed=0, size=100) -> dict[str, torch.Tensor]:
    """Create a simple state_dict for testing."""
    torch.manual_seed(seed)
    return {
        "layer1.weight": torch.randn(10, size),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
        "layer2.bias": torch.randn(5),
    }


def make_identical_weights(n=5, seed=0) -> list[dict[str, torch.Tensor]]:
    w = make_weights(seed=seed)
    return [copy.deepcopy(w) for _ in range(n)]


# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------

class TestFlattenUnflatten:
    def test_roundtrip_identical(self):
        w = make_weights()
        flat = flatten_weights(w)
        recovered = unflatten_weights(flat, w)
        for k in w:
            assert torch.allclose(w[k].float(), recovered[k].float(), atol=1e-6)

    def test_flat_is_1d(self):
        w = make_weights()
        flat = flatten_weights(w)
        assert flat.dim() == 1

    def test_flat_length(self):
        w = make_weights(size=10)
        flat = flatten_weights(w)
        expected = sum(v.numel() for v in w.values())
        assert flat.numel() == expected

    def test_deterministic_ordering(self):
        """Same keys in different insertion order → same flat vector."""
        w = make_weights(size=10)
        w2 = {k: w[k] for k in reversed(list(w.keys()))}
        assert torch.allclose(flatten_weights(w), flatten_weights(w2))

    def test_unflatten_preserves_dtype(self):
        w = {k: v.to(torch.float16) for k, v in make_weights(size=5).items()}
        flat = flatten_weights(w)
        recovered = unflatten_weights(flat, w)
        for k in w:
            assert recovered[k].dtype == torch.float16


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

class TestDistanceHelpers:
    def test_l2_distance_zero_self(self):
        w = make_weights()
        assert l2_distance(w, w) == pytest.approx(0.0, abs=1e-5)

    def test_l2_distance_symmetric(self):
        w1 = make_weights(seed=0)
        w2 = make_weights(seed=1)
        assert l2_distance(w1, w2) == pytest.approx(l2_distance(w2, w1), rel=1e-4)

    def test_pairwise_distances_shape(self):
        ws = [make_weights(seed=i, size=5) for i in range(4)]
        dists = pairwise_distances(ws)
        assert dists.shape == (4, 4)

    def test_pairwise_distances_diagonal_zero(self):
        ws = [make_weights(seed=i, size=5) for i in range(3)]
        dists = pairwise_distances(ws)
        diag = torch.diagonal(dists)
        assert torch.all(diag < 1e-4)

    def test_pairwise_distances_symmetric(self):
        ws = [make_weights(seed=i, size=5) for i in range(4)]
        dists = pairwise_distances(ws)
        assert torch.allclose(dists, dists.T, atol=1e-5)


# ---------------------------------------------------------------------------
# gossip_mean
# ---------------------------------------------------------------------------

class TestGossipMean:
    def test_identical_weights_unchanged(self):
        ws = make_identical_weights(n=5)
        result = gossip_mean(ws[0], ws[1:])
        for k in ws[0]:
            assert torch.allclose(ws[0][k].float(), result[k].float(), atol=1e-5)

    def test_average_two_known(self):
        """Mean of [1, 3] should be 2."""
        w1 = {"a": torch.ones(5) * 1.0}
        w2 = {"a": torch.ones(5) * 3.0}
        result = gossip_mean(w1, [w2])
        assert torch.allclose(result["a"], torch.ones(5) * 2.0, atol=1e-5)

    def test_average_three_known(self):
        """Mean of [0, 1, 2] should be 1."""
        ws = [{"a": torch.ones(3) * float(i)} for i in range(3)]
        result = gossip_mean(ws[0], ws[1:])
        assert torch.allclose(result["a"], torch.ones(3) * 1.0, atol=1e-5)

    def test_no_neighbors(self):
        """With no neighbours, result should equal own weights."""
        w = make_weights()
        result = gossip_mean(w, [])
        for k in w:
            assert torch.allclose(w[k].float(), result[k].float())

    def test_does_not_mutate_inputs(self):
        w1 = {"a": torch.ones(5)}
        w2 = {"a": torch.zeros(5)}
        _ = gossip_mean(w1, [w2])
        assert torch.all(w1["a"] == 1.0)


# ---------------------------------------------------------------------------
# ClippedGossip
# ---------------------------------------------------------------------------

class TestClippedGossip:
    def test_honest_clients_similar_to_mean(self):
        """With no Byzantines, ClippedGossip ≈ mean (within numerical tolerance)."""
        ws = [make_weights(seed=i, size=10) for i in range(5)]
        result_mean = gossip_mean(ws[0], ws[1:])
        result_cg = clipped_gossip(ws[0], ws[1:], tau=10.0)
        # Should be close but not necessarily identical (clipping is active at large tau)
        dist = l2_distance(result_mean, result_cg)
        # Distance should be small relative to model norm
        own_norm = flatten_weights(ws[0]).norm().item()
        assert dist < own_norm * 0.5, f"CG diverged from mean: dist={dist:.4f}"

    def test_sign_flip_byzantine_resilience(self):
        """1 Byzantine sign_flip among 4 honest → CG closer to honest mean than plain mean."""
        honest = [make_weights(seed=i, size=20) for i in range(4)]
        byzantine = {"layer1.weight": -10.0 * torch.ones(10, 20),
                     "layer1.bias": -10.0 * torch.ones(10),
                     "layer2.weight": -10.0 * torch.ones(5, 10),
                     "layer2.bias": -10.0 * torch.ones(5)}

        honest_mean_w = gossip_mean(honest[0], honest[1:])

        # With Byzantine
        neighbors_with_byz = honest[1:] + [byzantine]
        result_mean = gossip_mean(honest[0], neighbors_with_byz)
        result_cg = clipped_gossip(honest[0], neighbors_with_byz, tau=None, tau_percentile=50)

        dist_mean = l2_distance(result_mean, honest_mean_w)
        dist_cg = l2_distance(result_cg, honest_mean_w)

        assert dist_cg < dist_mean, (
            f"ClippedGossip (dist={dist_cg:.4f}) should be closer to honest mean "
            f"than plain mean (dist={dist_mean:.4f})"
        )

    def test_clipping_with_explicit_tau(self):
        """Ensure clipping does something when Byzantine weight is huge."""
        own = {"a": torch.zeros(100)}
        byz = {"a": torch.ones(100) * 1000.0}
        tau = 1.0

        result = clipped_gossip(own, [byz], tau=tau)
        # The result should have moved at most tau from own (delta clipped)
        moved = (result["a"] - own["a"]).norm().item()
        assert moved <= tau + 1e-4, f"Moved {moved:.4f} > tau={tau}"

    def test_auto_tau_positive(self):
        """Auto-tau should always be positive."""
        ws = [make_weights(seed=i, size=10) for i in range(5)]
        tau = compute_tau_auto(ws)
        assert tau > 0

    def test_no_neighbors_returns_own(self):
        w = make_weights()
        result = clipped_gossip(w, [])
        for k in w:
            assert torch.allclose(w[k].float(), result[k].float())


# ---------------------------------------------------------------------------
# SSClip
# ---------------------------------------------------------------------------

class TestSSClip:
    def test_output_within_tau_of_own(self):
        """SSClip guarantee: output always within tau of own_weights (L2)."""
        own = make_weights(seed=0, size=10)
        neighbors = [make_weights(seed=i, size=10) for i in range(5)]
        # Make one neighbour very far away (Byzantine-like)
        neighbors[0] = {"layer1.weight": torch.ones(10, 10) * 500,
                        "layer1.bias": torch.ones(10) * 500,
                        "layer2.weight": torch.ones(5, 10) * 500,
                        "layer2.bias": torch.ones(5) * 500}

        tau = 5.0
        result = ssclip(own, neighbors, tau=tau)
        dist_to_own = l2_distance(result, own)
        # SSClip averages own with clipped neighbors, so result is between own and neighbors
        # Distance must be ≤ tau (the clipping radius)
        assert dist_to_own <= tau + 1e-3, (
            f"SSClip output is {dist_to_own:.4f} away from own, exceeds tau={tau}"
        )

    def test_better_than_mean_under_sign_flip(self):
        """SSClip should outperform plain mean under sign_flip attack."""
        torch.manual_seed(7)
        honest = [{"a": torch.randn(50) * 0.1 + 1.0} for _ in range(4)]  # honest near 1
        byzantine = {"a": torch.ones(50) * (-100.0)}  # sign_flip: very negative

        honest_mean_w = gossip_mean(honest[0], honest[1:])
        neighbors = honest[1:] + [byzantine]

        result_mean = gossip_mean(honest[0], neighbors)
        result_ss = ssclip(honest[0], neighbors, tau=None, tau_percentile=50)

        dist_mean = l2_distance(result_mean, honest_mean_w)
        dist_ss = l2_distance(result_ss, honest_mean_w)

        assert dist_ss < dist_mean, (
            f"SSClip dist={dist_ss:.4f} should < mean dist={dist_mean:.4f}"
        )

    def test_identical_honest_no_change(self):
        """With identical honest weights and no Byzantines, SSClip = input."""
        ws = make_identical_weights(n=5)
        result = ssclip(ws[0], ws[1:])
        for k in ws[0]:
            assert torch.allclose(ws[0][k].float(), result[k].float(), atol=1e-5)

    def test_no_neighbors_returns_own(self):
        w = make_weights()
        result = ssclip(w, [])
        for k in w:
            assert torch.allclose(w[k].float(), result[k].float())
