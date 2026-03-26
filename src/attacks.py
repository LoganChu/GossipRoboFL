"""
attacks.py — Byzantine attack generators for GossipRoboFL.

All functions are pure (stateless): they take honest weights and return
corrupted weights without modifying inputs. Attacks are injected by
RobotClient.get_weights() only for designated Byzantine robots.

Attack taxonomy:
  sign_flip         — Negate all parameters. Simple but effective against
                      vanilla gossip. Easily detectable by robust aggregators.

  random_noise      — Replace weights with large-scale Gaussian noise.
                      More sophisticated than sign_flip; harder to filter
                      when the noise scale matches Byzantine distance.

  gaussian_perturb  — Add moderate Gaussian noise to weights (weak attack,
                      useful for testing robustness margins).

  label_flip        — Corruption at training time (handled in client.py via
                      LabelFlipDataset). No function here — listed for docs.

  partial_knowledge — Adaptive attack: uses observed honest neighbour weights
                      to craft an update that maximally disrupts the consensus
                      centroid. Most powerful attack in this suite.

Reference: Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust
Federated Learning", USENIX Security 2020.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch

from src.config import AttackConfig


# ---------------------------------------------------------------------------
# Individual attack functions
# ---------------------------------------------------------------------------

def sign_flip(
    weights: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Negate (and optionally scale) all model parameters.

    The most basic Byzantine attack: a Byzantine robot broadcasts the negation
    of its locally trained weights. Without robust aggregation, this drives the
    aggregate in the opposite direction from the true optimum.

    Args:
        weights: Honest state_dict to corrupt.
        scale: Multiply negated weights by this factor (default 1.0 → pure negation).

    Returns:
        New state_dict where every tensor v → -scale * v.
    """
    return {k: -scale * v.float() for k, v in weights.items()}


def random_noise(
    weights: dict[str, torch.Tensor],
    scale: float = 10.0,
    seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Replace model weights with large Gaussian noise.

    The noise magnitude (scale) should be >> typical weight norm to be
    effective. ClippedGossip and SSClip clip these outliers when tau is
    correctly calibrated.

    Args:
        weights: Honest state_dict (used only for shape information).
        scale: Standard deviation of the replacement Gaussian noise.
        seed: Optional seed for reproducible noise generation.

    Returns:
        New state_dict filled with N(0, scale²) samples.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    return {
        k: torch.randn(v.shape, generator=rng).float() * scale
        for k, v in weights.items()
    }


def gaussian_perturbation(
    weights: dict[str, torch.Tensor],
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Add zero-mean Gaussian noise to honest weights (weak / gradual attack).

    Unlike random_noise which replaces weights entirely, this adds noise:
    w_corrupt = w_honest + N(0, scale²). With small scale, this represents
    sensor noise or gradient estimation error.

    Args:
        weights: Honest state_dict.
        scale: Std dev of additive noise.
        seed: Optional seed.

    Returns:
        Perturbed state_dict.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    return {
        k: v.float() + torch.randn(v.shape, generator=rng).float() * scale
        for k, v in weights.items()
    }


def partial_knowledge(
    own_weights: dict[str, torch.Tensor],
    observed_honest_weights: list[dict[str, torch.Tensor]],
    scale: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Adaptive attack using observed neighbour weights to maximise disruption.

    The attacker (Byzantine robot) observes some honest neighbours' weights
    via gossip, computes their centroid, and broadcasts a weight vector that
    moves away from the consensus by `scale` times the honest update norm.

    This is the strongest attack in the suite because it is adaptive to the
    current state of the network. Robust aggregators still clip it when
    tau ≤ honest update norm * scale.

    Algorithm:
      1. Compute centroid of observed honest weights.
      2. direction = centroid - own_weights (flattened).
      3. Broadcast: own_weights - scale × direction.
         (moves opposite to where honest clients are converging)

    Falls back to sign_flip if no honest weights are observed.

    Args:
        own_weights: This Byzantine robot's locally trained weights.
        observed_honest_weights: Weights received from honest neighbours
                                  during the previous gossip push.
        scale: How aggressively to move anti-centroid (default 5.0).

    Returns:
        Corrupted state_dict that points away from the honest consensus.
    """
    if not observed_honest_weights:
        return sign_flip(own_weights)

    from src.gossip import flatten_weights, unflatten_weights

    flat_own = flatten_weights(own_weights)
    flat_centroid = torch.stack(
        [flatten_weights(w) for w in observed_honest_weights]
    ).mean(dim=0)

    direction = flat_centroid - flat_own
    corrupted_flat = flat_own - scale * direction
    return unflatten_weights(corrupted_flat, own_weights)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_attack(
    weights: dict[str, torch.Tensor],
    attack_type: str,
    attack_config: AttackConfig,
    observed_weights: Optional[list[dict[str, torch.Tensor]]] = None,
    seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Apply a named attack to model weights.

    This is the single entry point called by RobotClient.get_weights().

    Args:
        weights: Honest model state_dict to corrupt.
        attack_type: One of the attack names (see module docstring).
        attack_config: AttackConfig holding attack hyperparameters.
        observed_weights: Required for "partial_knowledge" attack; ignored otherwise.
        seed: Optional seed for stochastic attacks.

    Returns:
        Corrupted state_dict. Input is not modified.
    """
    if attack_type == "sign_flip":
        return sign_flip(weights, scale=attack_config.sign_scale)

    elif attack_type == "random_noise":
        return random_noise(weights, scale=attack_config.noise_scale, seed=seed)

    elif attack_type == "gaussian_perturb":
        return gaussian_perturbation(weights, scale=attack_config.noise_scale, seed=seed)

    elif attack_type == "partial_knowledge":
        return partial_knowledge(
            own_weights=weights,
            observed_honest_weights=observed_weights or [],
            scale=attack_config.noise_scale,
        )

    elif attack_type == "label_flip":
        # Label flip corrupts training data, not weights — handled in client.py.
        # Returning honest weights here is correct: the corruption is already baked in.
        return copy.deepcopy(weights)

    else:
        raise ValueError(
            f"Unknown attack type '{attack_type}'. "
            "Choose from: sign_flip, random_noise, gaussian_perturb, "
            "partial_knowledge, label_flip."
        )


# ---------------------------------------------------------------------------
# Byzantine client selection
# ---------------------------------------------------------------------------

def select_byzantine_clients(
    num_clients: int,
    fraction: float,
    seed: int = 42,
) -> set[int]:
    """Randomly select a reproducible set of Byzantine client IDs.

    Args:
        num_clients: Total number of clients N.
        fraction: Fraction of clients to mark as Byzantine (0 ≤ f < 0.5).
        seed: Random seed for reproducibility.

    Returns:
        Set of Byzantine client IDs (subset of {0, ..., N-1}).
    """
    n_byzantine = int(fraction * num_clients)
    if n_byzantine == 0:
        return set()

    rng = np.random.default_rng(seed)
    chosen = rng.choice(num_clients, size=n_byzantine, replace=False)
    return set(int(c) for c in chosen)
