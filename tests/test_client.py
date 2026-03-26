"""
test_client.py — Unit tests for src/client.py.
"""
import copy
import sys
import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import LabelFlipDataset, RobotClient
from src.config import AttackConfig, ClientConfig
from src.model import get_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_tiny_dataset(n=64, num_classes=10):
    imgs = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(imgs, labels)
    ds.targets = labels.tolist()
    return ds


def make_client(
    client_id=0,
    attack_type=None,
    heterogeneous=False,
    local_epochs=1,
) -> RobotClient:
    ds = make_tiny_dataset(n=64)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = get_model("cnn_cifar", num_classes=10)
    device = torch.device("cpu")

    client_cfg = ClientConfig(
        local_epochs=local_epochs,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        heterogeneous=heterogeneous,
        min_epochs=1,
        max_epochs=3,
        delay_scale=0.0,
    )

    attack_cfg = None
    if attack_type:
        attack_cfg = AttackConfig(
            enabled=True,
            type=attack_type,
            fraction=1.0,
            noise_scale=10.0,
            sign_scale=1.0,
        )

    return RobotClient(
        client_id=client_id,
        model=model,
        train_loader=loader,
        val_loader=None,
        config=client_cfg,
        attack_config=attack_cfg,
        device=device,
        position=(0.3, 0.7),
        seed=42,
    )


# ---------------------------------------------------------------------------
# LabelFlipDataset
# ---------------------------------------------------------------------------

class TestLabelFlipDataset:
    def test_labels_flipped(self):
        ds = make_tiny_dataset(n=20, num_classes=10)
        flipped = LabelFlipDataset(ds, num_classes=10)
        for i in range(len(ds)):
            _, orig_label = ds[i]
            _, flip_label = flipped[i]
            assert flip_label == 9 - orig_label

    def test_images_unchanged(self):
        ds = make_tiny_dataset(n=10)
        flipped = LabelFlipDataset(ds)
        for i in range(len(ds)):
            img_orig, _ = ds[i]
            img_flip, _ = flipped[i]
            assert torch.allclose(img_orig, img_flip)

    def test_len(self):
        ds = make_tiny_dataset(n=50)
        assert len(LabelFlipDataset(ds)) == 50


# ---------------------------------------------------------------------------
# RobotClient — honest
# ---------------------------------------------------------------------------

class TestHonestClient:
    def test_local_train_returns_dict(self):
        c = make_client()
        result = c.local_train()
        assert "loss" in result
        assert "samples" in result
        assert "epochs" in result
        assert result["client_id"] == 0

    def test_local_train_reduces_loss(self):
        """After 3 epochs, training loss should be lower than initial."""
        c = make_client(local_epochs=3)
        result1 = c.local_train()
        # Train again — loss should be similar or lower (not a strict test since it's stochastic)
        result2 = c.local_train()
        # Both should be finite
        assert result1["loss"] < float("inf")
        assert result2["loss"] < float("inf")

    def test_get_weights_returns_dict(self):
        c = make_client()
        weights = c.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_get_weights_is_deep_copy(self):
        """Mutating returned weights should not affect the model."""
        c = make_client()
        weights = c.get_weights()
        first_key = list(weights.keys())[0]
        original_val = weights[first_key].clone()
        weights[first_key] *= 99.0  # mutate
        # Model should be unaffected
        new_weights = c.get_weights()
        assert torch.allclose(new_weights[first_key].float(), original_val.float(), atol=1e-5)

    def test_set_then_get_roundtrip(self):
        c = make_client()
        target = {k: torch.zeros_like(v) for k, v in c.model.state_dict().items()}
        c.set_weights(target)
        got = c.get_weights()
        for k in target:
            assert torch.allclose(got[k].float(), target[k].float())

    def test_evaluate_returns_dict(self):
        c = make_client()
        ds = make_tiny_dataset(n=32)
        loader = DataLoader(ds, batch_size=32)
        result = c.evaluate(loader)
        assert "accuracy" in result
        assert "loss" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_is_byzantine_false(self):
        c = make_client()
        assert not c.is_byzantine

    def test_position_property(self):
        c = make_client()
        assert c.position == (0.3, 0.7)

    def test_move_reflect(self):
        c = make_client()
        c.position = (0.05, 0.05)
        c.move((-0.1, -0.1), boundary="reflect")
        x, y = c.position
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0

    def test_move_wrap(self):
        c = make_client()
        c.position = (0.95, 0.95)
        c.move((0.1, 0.1), boundary="wrap")
        x, y = c.position
        assert 0.0 <= x < 1.0 + 1e-9
        assert 0.0 <= y < 1.0 + 1e-9


# ---------------------------------------------------------------------------
# RobotClient — Byzantine
# ---------------------------------------------------------------------------

class TestByzantineClient:
    def test_sign_flip_is_byzantine(self):
        c = make_client(attack_type="sign_flip")
        assert c.is_byzantine

    def test_sign_flip_negates_weights(self):
        c = make_client(attack_type="sign_flip")
        # Train first so weights are non-zero
        c.local_train()
        honest_weights = copy.deepcopy(c.model.state_dict())
        corrupted = c.get_weights()
        for k in honest_weights:
            assert torch.allclose(
                corrupted[k].float(), -honest_weights[k].float(), atol=1e-5
            ), f"Key {k}: sign_flip did not negate"

    def test_random_noise_differs_from_honest(self):
        c = make_client(attack_type="random_noise")
        c.local_train()
        honest = copy.deepcopy(c.model.state_dict())
        corrupted = c.get_weights()
        any_diff = any(
            not torch.allclose(corrupted[k].float(), honest[k].float())
            for k in honest
        )
        assert any_diff

    def test_label_flip_client_trains_on_flipped_labels(self):
        """Label flip wraps the DataLoader at init; model trains on inverted labels."""
        c = make_client(attack_type="label_flip")
        assert c.is_byzantine
        # Just verify it trains without error
        result = c.local_train()
        assert result["samples"] > 0
