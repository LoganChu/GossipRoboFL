"""
client.py — RobotClient: the core agent in the GossipRoboFL swarm.

Physical AI framing: each RobotClient represents a mobile robot with:
  - Its own onboard model (starts from shared random init, diverges via local SGD)
  - Local sensor data (non-IID CIFAR-10 shard representing its environment)
  - A 2D position in the arena (managed externally by TopologyManager)
  - Optional Byzantine behaviour (corrupts weights before broadcasting)

The separation of concerns is intentional:
  - client.py: per-robot training + weight management
  - gossip.py: aggregation algorithms (stateless, pure functions)
  - topology.py: graph and movement
  - attacks.py: corruption functions (called by get_weights() here)
"""
from __future__ import annotations

import copy
import random
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import AttackConfig, ClientConfig


# ---------------------------------------------------------------------------
# LabelFlipDataset — wraps a dataset and flips labels for Byzantine training
# ---------------------------------------------------------------------------

class LabelFlipDataset(Dataset):
    """Dataset wrapper that flips labels: class c → (num_classes - 1 - c).

    Used by Byzantine clients whose attack type is "label_flip": they train
    on poisoned data so their local model learns inverted associations, then
    broadcast these corrupted weights to neighbours.
    """

    def __init__(self, dataset: Dataset, num_classes: int = 10) -> None:
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        img, label = self.dataset[idx]
        flipped_label = self.num_classes - 1 - int(label)
        return img, flipped_label


# ---------------------------------------------------------------------------
# RobotClient
# ---------------------------------------------------------------------------

class RobotClient:
    """Simulated swarm robot that participates in gossip federated learning.

    Lifecycle per round:
      1. local_train()        — run E SGD epochs on local shard
      2. get_weights()        — return (possibly corrupted) model state_dict
      3. apply_gossip_update() — receive aggregated weights from neighbours
      4. evaluate()           — score on global test set (for logging)

    Byzantine injection:
      - "label_flip": LabelFlipDataset wraps the DataLoader at __init__ time,
        poisoning the training signal. Corrupted weights are then broadcast.
      - All other attack types: training proceeds honestly. Corruption is
        applied lazily inside get_weights() by calling attacks.apply_attack().
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: ClientConfig,
        attack_config: Optional[AttackConfig],
        device: torch.device,
        position: tuple[float, float],
        seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.attack_config = attack_config
        self._position = list(position)
        self.seed = seed

        # Heterogeneous client: sample local epochs once at init
        if config.heterogeneous:
            rng = random.Random(seed + client_id)
            self._local_epochs = rng.randint(config.min_epochs, config.max_epochs)
        else:
            self._local_epochs = config.local_epochs

        # Byzantine flag and optional label-flip wrapping
        self.is_byzantine: bool = (
            attack_config is not None
            and attack_config.enabled
        )
        self._attack_type: str = attack_config.type if attack_config else ""

        if self.is_byzantine and self._attack_type == "label_flip":
            # Rewrap the DataLoader with poisoned labels
            poisoned_ds = LabelFlipDataset(
                train_loader.dataset,  # type: ignore[arg-type]
                num_classes=10,
            )
            self.train_loader = DataLoader(
                poisoned_ds,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
                drop_last=False,
            )
        else:
            self.train_loader = train_loader

        self.val_loader = val_loader

        # Optimiser — SGD with momentum and L2 regularisation
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        # Running stats
        self._round = 0
        self._last_train_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def local_train(self) -> dict:
        """Run local_epochs of SGD on local data.

        Returns:
            {"loss": float, "samples": int, "epochs": int, "client_id": int}
        """
        # Simulate communication delay (heterogeneous compute/network)
        if self.config.delay_scale > 0:
            delay = random.expovariate(1.0 / self.config.delay_scale)
            time.sleep(min(delay, 2.0))  # cap at 2s to avoid stalling simulation

        self.model.train()
        total_loss = 0.0
        total_samples = 0
        epochs = self._local_epochs

        for _ in range(epochs):
            for imgs, labels in self.train_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        self._last_train_loss = avg_loss
        self._round += 1

        return {
            "client_id": self.client_id,
            "loss": avg_loss,
            "samples": total_samples,
            "epochs": epochs,
            "is_byzantine": self.is_byzantine,
        }

    # ------------------------------------------------------------------
    # Weight access (Byzantine clients corrupt here)
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return current model weights (state_dict copy).

        For Byzantine clients (non-label_flip), this applies the attack
        transformation to the weights before returning. The internal model
        state is NOT modified — corruption only affects what is broadcast.

        Returns:
            Deep-copied (possibly corrupted) state_dict.
        """
        weights = copy.deepcopy(self.model.state_dict())

        if self.is_byzantine and self._attack_type != "label_flip":
            # Import here to avoid circular imports at module load time
            from src.attacks import apply_attack
            weights = apply_attack(
                weights=weights,
                attack_type=self._attack_type,
                attack_config=self.attack_config,  # type: ignore[arg-type]
                observed_weights=None,  # partial_knowledge needs external injection
                seed=self.seed + self._round,
            )

        return weights

    def get_model_delta(
        self,
        reference_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return difference between current weights and a reference.

        Useful for computing update norms (e.g., for tau auto-selection in gossip).

        Args:
            reference_weights: Baseline weights (e.g., weights at start of round).

        Returns:
            {key: current[key] - reference[key]} for all keys.
        """
        current = self.model.state_dict()
        return {k: current[k].float() - reference_weights[k].float() for k in current}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load a state_dict into the model (strict=True)."""
        self.model.load_state_dict(weights, strict=True)

    def apply_gossip_update(self, aggregated_weights: dict[str, torch.Tensor]) -> None:
        """Apply aggregated gossip weights to the model.

        This is called after each gossip round to synchronise the local model
        with the neighbourhood consensus.
        """
        self.set_weights(aggregated_weights)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate model on a DataLoader (typically the global test set).

        Args:
            test_loader: DataLoader over evaluation data.

        Returns:
            {"accuracy": float, "loss": float, "client_id": int}
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in test_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        return {
            "client_id": self.client_id,
            "accuracy": correct / max(total, 1),
            "loss": total_loss / max(total, 1),
        }

    # ------------------------------------------------------------------
    # Position / mobility (called by TopologyManager)
    # ------------------------------------------------------------------

    @property
    def position(self) -> tuple[float, float]:
        return (self._position[0], self._position[1])

    @position.setter
    def position(self, pos: tuple[float, float]) -> None:
        self._position = [pos[0], pos[1]]

    def move(self, step: tuple[float, float], boundary: str = "reflect") -> None:
        """Update 2D position with boundary handling.

        Args:
            step: (dx, dy) displacement.
            boundary: "reflect" mirrors off walls; "wrap" applies modular arithmetic.
        """
        for axis in range(2):
            self._position[axis] += step[axis]
            if boundary == "reflect":
                if self._position[axis] < 0.0:
                    self._position[axis] = -self._position[axis]
                elif self._position[axis] > 1.0:
                    self._position[axis] = 2.0 - self._position[axis]
                # Clamp after reflection in case of large steps
                self._position[axis] = max(0.0, min(1.0, self._position[axis]))
            else:  # wrap
                self._position[axis] = self._position[axis] % 1.0

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        byz_str = f", attack={self._attack_type}" if self.is_byzantine else ""
        return (
            f"RobotClient(id={self.client_id}, "
            f"byzantine={self.is_byzantine}{byz_str}, "
            f"pos=({self._position[0]:.2f},{self._position[1]:.2f}), "
            f"epochs={self._local_epochs})"
        )
