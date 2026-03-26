"""
test_attacks.py — Unit tests for src/attacks.py.
"""
import copy
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks import (
    apply_attack,
    gaussian_perturbation,
    partial_knowledge,
    random_noise,
    select_byzantine_clients,
    sign_flip,
)
from src.config import AttackConfig


def make_weights(val=1.0, size=20) -> dict[str, torch.Tensor]:
    return {
        "w": torch.ones(size) * val,
        "b": torch.ones(5) * val,
    }


def default_attack_config(**kwargs) -> AttackConfig:
    cfg = AttackConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


class TestSignFlip:
    def test_negates_all_params(self):
        w = make_weights(val=1.0)
        result = sign_flip(w)
        for k, v in result.items():
            assert torch.all(v == -1.0), f"Key {k}: expected -1.0, got {v}"

    def test_does_not_mutate_input(self):
        w = make_weights(val=2.0)
        _ = sign_flip(w)
        assert torch.all(w["w"] == 2.0)

    def test_scale(self):
        w = make_weights(val=1.0)
        result = sign_flip(w, scale=3.0)
        for v in result.values():
            assert torch.all(v == -3.0)


class TestRandomNoise:
    def test_shape_preserved(self):
        w = make_weights()
        result = random_noise(w, scale=1.0, seed=0)
        for k in w:
            assert result[k].shape == w[k].shape

    def test_differs_from_input(self):
        w = make_weights(val=0.0)
        result = random_noise(w, scale=10.0, seed=1)
        any_different = any(not torch.allclose(result[k], w[k]) for k in w)
        assert any_different

    def test_large_scale_large_values(self):
        w = make_weights(val=0.0)
        result = random_noise(w, scale=100.0, seed=0)
        max_val = max(v.abs().max().item() for v in result.values())
        assert max_val > 10.0, "Large noise scale should produce large values"

    def test_reproducible_with_seed(self):
        w = make_weights()
        r1 = random_noise(w, scale=5.0, seed=42)
        r2 = random_noise(w, scale=5.0, seed=42)
        for k in w:
            assert torch.allclose(r1[k], r2[k])


class TestGaussianPerturbation:
    def test_close_to_input_small_scale(self):
        w = make_weights(val=5.0)
        result = gaussian_perturbation(w, scale=0.001, seed=0)
        for k in w:
            assert torch.allclose(result[k], w[k], atol=0.1), "Small noise should stay close"

    def test_shape_preserved(self):
        w = make_weights()
        result = gaussian_perturbation(w, scale=1.0, seed=0)
        for k in w:
            assert result[k].shape == w[k].shape


class TestPartialKnowledge:
    def test_moves_away_from_centroid(self):
        """Output should be farther from centroid than input is."""
        own = make_weights(val=0.0)
        honest = [make_weights(val=1.0) for _ in range(3)]
        result = partial_knowledge(own, honest, scale=2.0)
        # Centroid is at val=1.0; result should be at val ≈ 0 - 2*(1-0) = -2
        assert torch.mean(result["w"]).item() < 0.0

    def test_falls_back_to_sign_flip_empty(self):
        """With no observed weights, should fall back to sign_flip."""
        own = make_weights(val=2.0)
        result = partial_knowledge(own, [], scale=5.0)
        for k in result:
            assert torch.all(result[k] < 0), "Fallback sign_flip should negate"

    def test_does_not_mutate_inputs(self):
        own = make_weights(val=1.0)
        honest = [make_weights(val=3.0)]
        _ = partial_knowledge(own, honest)
        assert torch.all(own["w"] == 1.0)


class TestApplyAttack:
    def test_sign_flip_dispatch(self):
        w = make_weights(val=1.0)
        cfg = default_attack_config(type="sign_flip", sign_scale=1.0)
        result = apply_attack(w, "sign_flip", cfg)
        assert torch.all(result["w"] == -1.0)

    def test_random_noise_dispatch(self):
        w = make_weights()
        cfg = default_attack_config(type="random_noise", noise_scale=10.0)
        result = apply_attack(w, "random_noise", cfg, seed=0)
        assert result["w"].shape == w["w"].shape

    def test_label_flip_returns_honest_weights(self):
        """Label flip corrupts training, not weights — apply_attack should pass through."""
        w = make_weights(val=5.0)
        cfg = default_attack_config(type="label_flip")
        result = apply_attack(w, "label_flip", cfg)
        assert torch.allclose(result["w"], w["w"])

    def test_unknown_attack_raises(self):
        w = make_weights()
        cfg = default_attack_config()
        with pytest.raises(ValueError, match="Unknown attack type"):
            apply_attack(w, "made_up_attack", cfg)


class TestSelectByzantineClients:
    def test_correct_count(self):
        byz = select_byzantine_clients(20, fraction=0.2, seed=0)
        assert len(byz) == 4  # floor(0.2 * 20) = 4

    def test_within_valid_range(self):
        byz = select_byzantine_clients(20, fraction=0.3, seed=0)
        assert all(0 <= b < 20 for b in byz)

    def test_reproducible(self):
        byz1 = select_byzantine_clients(20, fraction=0.2, seed=42)
        byz2 = select_byzantine_clients(20, fraction=0.2, seed=42)
        assert byz1 == byz2

    def test_zero_fraction_empty(self):
        byz = select_byzantine_clients(20, fraction=0.0, seed=0)
        assert len(byz) == 0

    def test_different_seeds_different(self):
        byz1 = select_byzantine_clients(20, fraction=0.2, seed=1)
        byz2 = select_byzantine_clients(20, fraction=0.2, seed=2)
        # Very likely to differ
        assert byz1 != byz2
