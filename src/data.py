"""
data.py — Dataset loading and non-IID partitioning for GossipRoboFL.

Physical AI framing: each robot has its own local sensor data (camera images)
from a different environment, lighting condition, or object distribution. This
heterogeneity is modeled via Dirichlet(alpha) partitioning of CIFAR-10/FashionMNIST.

Low alpha (0.1) → highly non-IID: each robot sees only 1-2 classes (like a
robot that only ever sees indoor scenes vs outdoor). High alpha (10+) → nearly IID.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_FASHION_MEAN = (0.2860,)
_FASHION_STD = (0.3530,)


def load_raw_dataset(
    name: str,
    root: str = "data/",
    download: bool = True,
) -> tuple[Dataset, Dataset]:
    """Load a dataset and return (train_dataset, test_dataset).

    Args:
        name: "cifar10" | "fashion_mnist".
        root: Local directory for dataset storage.
        download: Auto-download if not present.

    Returns:
        Tuple of (train, test) torchvision Datasets with standard normalisation.
    """
    os.makedirs(root, exist_ok=True)

    if name == "cifar10":
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
        train_ds = torchvision.datasets.CIFAR10(root, train=True, download=download, transform=train_transform)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, download=download, transform=test_transform)

    elif name == "fashion_mnist":
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_FASHION_MEAN, _FASHION_STD),
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(_FASHION_MEAN, _FASHION_STD),
        ])
        train_ds = torchvision.datasets.FashionMNIST(root, train=True, download=download, transform=train_transform)
        test_ds = torchvision.datasets.FashionMNIST(root, train=False, download=download, transform=test_transform)

    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'cifar10' or 'fashion_mnist'.")

    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Non-IID partitioning via Dirichlet distribution
# ---------------------------------------------------------------------------

def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples: int = 10,
) -> list[list[int]]:
    """Partition dataset indices across clients using Dirichlet(alpha).

    Algorithm:
      1. Group all indices by class label.
      2. For each class c, sample proportions p_c ~ Dir(alpha * 1_N).
      3. Assign floor(p_c[i] * |class_c|) indices to client i.
      4. Redistribute leftover indices round-robin to maintain coverage.
      5. Enforce min_samples floor by stealing from the largest clients.

    Args:
        dataset: Source dataset (must have .targets attribute or iterable labels).
        num_clients: Number of clients N.
        alpha: Dirichlet concentration. Lower → more non-IID.
        seed: Random seed for reproducibility.
        min_samples: Minimum samples per client (prevents empty shards).

    Returns:
        List of N lists, where each inner list contains dataset indices.
    """
    rng = np.random.default_rng(seed)

    # Extract labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])  # type: ignore[arg-type]

    num_classes = int(labels.max()) + 1
    class_indices: list[list[int]] = [
        np.where(labels == c)[0].tolist() for c in range(num_classes)
    ]

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idxs = class_indices[c]
        rng.shuffle(idxs)  # type: ignore[arg-type]
        # Draw Dirichlet proportions for this class across all clients
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # Compute how many samples each client gets from class c
        counts = (proportions * len(idxs)).astype(int)
        # Fix rounding: add leftovers to random clients
        leftover = len(idxs) - counts.sum()
        if leftover > 0:
            lucky = rng.choice(num_clients, size=leftover, replace=False)
            counts[lucky] += 1
        # Assign
        pos = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idxs[pos : pos + cnt])
            pos += cnt

    # Enforce minimum samples per client
    _enforce_min_samples(client_indices, min_samples, rng)

    # Verify coverage
    all_assigned = sorted(idx for idxs in client_indices for idx in idxs)
    all_expected = list(range(len(dataset)))
    assert all_assigned == all_expected, (
        f"Partition mismatch: {len(all_assigned)} assigned vs {len(all_expected)} expected"
    )

    return client_indices


def _enforce_min_samples(
    client_indices: list[list[int]],
    min_samples: int,
    rng: np.random.Generator,
) -> None:
    """Steal samples from largest clients to satisfy per-client floor (in-place)."""
    for i in range(len(client_indices)):
        deficit = min_samples - len(client_indices[i])
        if deficit <= 0:
            continue
        # Find donor clients (those with more than min_samples + deficit)
        donors = [
            j for j in range(len(client_indices))
            if j != i and len(client_indices[j]) > min_samples + deficit
        ]
        if not donors:
            # Best effort: steal from largest available
            donors = sorted(
                [j for j in range(len(client_indices)) if j != i],
                key=lambda j: -len(client_indices[j]),
            )
        for j in donors:
            take = min(deficit, len(client_indices[j]) - min_samples)
            if take <= 0:
                continue
            stolen_indices = rng.choice(len(client_indices[j]), size=take, replace=False)
            stolen = [client_indices[j].pop(idx) for idx in sorted(stolen_indices, reverse=True)]
            client_indices[i].extend(stolen)
            deficit -= take
            if deficit <= 0:
                break


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------

def make_client_dataloaders(
    dataset: Dataset,
    index_lists: list[list[int]],
    batch_size: int,
    val_split: float = 0.0,
    num_workers: int = 0,       # 0 is required on Windows for DataLoader in subprocesses
    pin_memory: bool = False,
) -> list[tuple[DataLoader, Optional[DataLoader]]]:
    """Create per-client (train_loader, val_loader) pairs.

    Args:
        dataset: Full training dataset.
        index_lists: Per-client index lists from dirichlet_partition().
        batch_size: Mini-batch size.
        val_split: Fraction held out for local validation (0 = no val set).
        num_workers: DataLoader workers (0 for Windows compatibility).
        pin_memory: Pin tensors to CUDA memory for faster transfer.

    Returns:
        List of (train_loader, val_loader | None) tuples, one per client.
    """
    loaders = []
    for idxs in index_lists:
        if val_split > 0.0:
            n_val = max(1, int(len(idxs) * val_split))
            val_idxs = idxs[:n_val]
            train_idxs = idxs[n_val:]
        else:
            train_idxs = idxs
            val_idxs = []

        train_loader = DataLoader(
            Subset(dataset, train_idxs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        val_loader: Optional[DataLoader] = None
        if val_idxs:
            val_loader = DataLoader(
                Subset(dataset, val_idxs),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        loaders.append((train_loader, val_loader))

    return loaders


def make_test_dataloader(
    test_dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Return a DataLoader over the global test set (shared by all clients)."""
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# Partition statistics and visualisation
# ---------------------------------------------------------------------------

def get_partition_stats(
    index_lists: list[list[int]],
    dataset: Dataset,
) -> dict:
    """Compute per-client class distribution statistics.

    Returns:
        {
          "label_distributions": np.ndarray (N, C) — fraction of each class per client,
          "shard_sizes": list[int] — number of samples per client,
          "entropy": list[float] — Shannon entropy of each client's class distribution,
          "num_clients": int,
          "num_classes": int,
        }
    """
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])  # type: ignore[arg-type]

    num_classes = int(labels.max()) + 1
    N = len(index_lists)
    dist = np.zeros((N, num_classes))

    for i, idxs in enumerate(index_lists):
        client_labels = labels[idxs]
        for c in range(num_classes):
            dist[i, c] = np.sum(client_labels == c)
        row_sum = dist[i].sum()
        if row_sum > 0:
            dist[i] /= row_sum

    entropies = [float(scipy_entropy(row + 1e-12)) for row in dist]  # type: ignore[arg-type]

    return {
        "label_distributions": dist,
        "shard_sizes": [len(idxs) for idxs in index_lists],
        "entropy": entropies,
        "num_clients": N,
        "num_classes": num_classes,
    }


def visualize_partition(
    index_lists: list[list[int]],
    dataset: Dataset,
    save_path: Optional[str] = None,
    max_clients: int = 30,
) -> plt.Figure:
    """Stacked bar chart showing class distribution per client.

    Args:
        index_lists: Per-client index lists.
        dataset: Source dataset.
        save_path: If provided, save figure to this path.
        max_clients: Cap display at this many clients for readability.

    Returns:
        Matplotlib Figure.
    """
    stats = get_partition_stats(index_lists, dataset)
    dist = stats["label_distributions"][:max_clients]
    N, C = dist.shape

    cmap = plt.cm.get_cmap("tab10", C)  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(max(10, N * 0.4), 5))

    bottom = np.zeros(N)
    for c in range(C):
        ax.bar(range(N), dist[:, c], bottom=bottom, color=cmap(c), label=f"Class {c}", width=0.8)
        bottom += dist[:, c]

    ax.set_xlabel("Robot / Client ID")
    ax.set_ylabel("Fraction of local data")
    ax.set_title(f"Non-IID Data Distribution (Dirichlet α, showing {N}/{len(index_lists)} clients)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
