"""
test_data.py — Unit tests for src/data.py.

Tests cover:
  - Dirichlet partition coverage (all indices assigned exactly once)
  - min_samples floor enforcement
  - IID vs non-IID class distribution
  - DataLoader creation
  - Partition stats
"""
import sys
import os

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    dirichlet_partition,
    get_partition_stats,
    make_client_dataloaders,
    make_test_dataloader,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_toy_dataset(n=1000, num_classes=10, seed=0):
    """Create a TensorDataset with random images and labels."""
    torch.manual_seed(seed)
    imgs = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(imgs, labels)
    ds.targets = labels.tolist()  # mimic torchvision API
    return ds


# ---------------------------------------------------------------------------
# Dirichlet partition
# ---------------------------------------------------------------------------

class TestDirichletPartition:
    def test_coverage_all_indices_once(self):
        ds = make_toy_dataset(n=500)
        index_lists = dirichlet_partition(ds, num_clients=10, alpha=0.5, seed=42, min_samples=5)
        all_assigned = sorted(idx for idxs in index_lists for idx in idxs)
        assert all_assigned == list(range(500)), "Not all indices covered exactly once"

    def test_num_clients(self):
        ds = make_toy_dataset(n=200)
        index_lists = dirichlet_partition(ds, num_clients=5, alpha=1.0, seed=0)
        assert len(index_lists) == 5

    def test_min_samples_floor(self):
        ds = make_toy_dataset(n=500)
        min_s = 10
        index_lists = dirichlet_partition(ds, num_clients=20, alpha=0.1, seed=42, min_samples=min_s)
        for i, idxs in enumerate(index_lists):
            assert len(idxs) >= min_s, f"Client {i} has only {len(idxs)} samples (min={min_s})"

    def test_high_alpha_near_iid(self):
        """With alpha=1000, each client should get ~equal class distribution."""
        ds = make_toy_dataset(n=2000, num_classes=5)
        index_lists = dirichlet_partition(ds, num_clients=5, alpha=1000.0, seed=0, min_samples=1)
        stats = get_partition_stats(index_lists, ds)
        dist = stats["label_distributions"]
        # Each class should be ~20% for each client
        for i in range(len(index_lists)):
            for c in range(5):
                assert abs(dist[i, c] - 0.2) < 0.15, (
                    f"Client {i}, class {c}: {dist[i, c]:.3f} far from 0.2 (IID expected)"
                )

    def test_low_alpha_non_iid(self):
        """With alpha=0.01, clients should have very skewed distributions (high entropy << log(C))."""
        ds = make_toy_dataset(n=2000, num_classes=10)
        index_lists = dirichlet_partition(ds, num_clients=10, alpha=0.01, seed=0, min_samples=1)
        stats = get_partition_stats(index_lists, ds)
        max_entropy = np.log(10)  # uniform distribution entropy
        mean_entropy = np.mean(stats["entropy"])
        assert mean_entropy < max_entropy * 0.7, (
            f"Expected non-IID entropy (< {max_entropy*0.7:.3f}), got {mean_entropy:.3f}"
        )

    def test_reproducibility(self):
        """Same seed → identical partition."""
        ds = make_toy_dataset(n=300)
        idx1 = dirichlet_partition(ds, num_clients=5, alpha=0.5, seed=7)
        idx2 = dirichlet_partition(ds, num_clients=5, alpha=0.5, seed=7)
        for i in range(5):
            assert sorted(idx1[i]) == sorted(idx2[i])

    def test_different_seeds_different(self):
        ds = make_toy_dataset(n=300)
        idx1 = dirichlet_partition(ds, num_clients=5, alpha=0.5, seed=1)
        idx2 = dirichlet_partition(ds, num_clients=5, alpha=0.5, seed=2)
        # Should differ for at least one client
        same = all(sorted(idx1[i]) == sorted(idx2[i]) for i in range(5))
        assert not same, "Different seeds produced identical partitions (unlikely)"


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------

class TestDataLoaders:
    def test_creates_correct_number(self):
        ds = make_toy_dataset(n=200)
        index_lists = dirichlet_partition(ds, num_clients=4, alpha=1.0, seed=0)
        loaders = make_client_dataloaders(ds, index_lists, batch_size=16)
        assert len(loaders) == 4

    def test_loaders_iterable(self):
        ds = make_toy_dataset(n=200)
        index_lists = dirichlet_partition(ds, num_clients=4, alpha=1.0, seed=0)
        loaders = make_client_dataloaders(ds, index_lists, batch_size=16)
        for train_loader, val_loader in loaders:
            for batch in train_loader:
                assert len(batch) == 2
                imgs, labels = batch
                assert imgs.shape[0] <= 16
                break

    def test_val_split_creates_val_loader(self):
        ds = make_toy_dataset(n=500)
        index_lists = dirichlet_partition(ds, num_clients=5, alpha=1.0, seed=0)
        loaders = make_client_dataloaders(ds, index_lists, batch_size=32, val_split=0.2)
        for _, val_loader in loaders:
            assert val_loader is not None

    def test_no_val_split_none(self):
        ds = make_toy_dataset(n=200)
        index_lists = dirichlet_partition(ds, num_clients=4, alpha=1.0, seed=0)
        loaders = make_client_dataloaders(ds, index_lists, batch_size=16, val_split=0.0)
        for _, val_loader in loaders:
            assert val_loader is None

    def test_test_dataloader_covers_all(self):
        ds = make_toy_dataset(n=300)
        loader = make_test_dataloader(ds, batch_size=64)
        total = sum(imgs.shape[0] for imgs, _ in loader)
        assert total == 300


# ---------------------------------------------------------------------------
# Partition statistics
# ---------------------------------------------------------------------------

class TestPartitionStats:
    def test_stats_keys(self):
        ds = make_toy_dataset(n=100)
        index_lists = dirichlet_partition(ds, num_clients=3, alpha=1.0, seed=0)
        stats = get_partition_stats(index_lists, ds)
        for key in ("label_distributions", "shard_sizes", "entropy", "num_clients", "num_classes"):
            assert key in stats

    def test_label_distributions_sum_to_one(self):
        ds = make_toy_dataset(n=500, num_classes=5)
        index_lists = dirichlet_partition(ds, num_clients=5, alpha=0.5, seed=0)
        stats = get_partition_stats(index_lists, ds)
        dist = stats["label_distributions"]
        for i in range(5):
            row_sum = dist[i].sum()
            assert abs(row_sum - 1.0) < 1e-5, f"Row {i} sums to {row_sum}"

    def test_shard_sizes_consistent(self):
        ds = make_toy_dataset(n=200)
        index_lists = dirichlet_partition(ds, num_clients=4, alpha=1.0, seed=0)
        stats = get_partition_stats(index_lists, ds)
        for i, (size, idxs) in enumerate(zip(stats["shard_sizes"], index_lists)):
            assert size == len(idxs)
