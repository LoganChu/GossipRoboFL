"""
gossip.py — Gossip aggregation algorithms for GossipRoboFL.

All functions are pure (no side effects) and operate on plain Python dicts
of {str: torch.Tensor} (model state_dicts). This makes them easy to unit
test and swap without touching the simulator.

Three aggregation modes:

  gossip_mean       — Standard gossip averaging. Baseline, no Byzantine resilience.

  clipped_gossip    — ClippedGossip (Farhadkhani et al., NeurIPS 2022).
                      Clips each neighbour's weight-delta to an L2 ball of
                      radius tau around own weights, then adds the mean clipped
                      delta. Byzantine drift ∝ f × tau.

  ssclip            — Self-Centred Clipping (Fraboni et al., 2022). Stronger:
                      reconstructs each accepted neighbour's weights after
                      clipping, then averages. Drift ≤ tau regardless of f
                      (as long as f < 0.5 of honest neighbours dominate).

Reference implementations:
  https://github.com/epfml/byzantine-robust-decentralized-optimizer
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

import networkx as nx
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Flatten / unflatten helpers
# ---------------------------------------------------------------------------

def flatten_weights(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate all state_dict tensors into a single 1-D float32 vector.

    Key ordering is deterministic (sorted alphabetically) so that two
    calls on the same keys always produce compatible flat vectors.
    """
    return torch.cat(
        [weights[k].float().reshape(-1) for k in sorted(weights.keys())]
    )


def unflatten_weights(
    flat: torch.Tensor,
    template: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Inverse of flatten_weights.

    Args:
        flat: 1-D tensor produced by flatten_weights.
        template: A state_dict whose shapes and key order are used for
                  reconstruction. Values are ignored.

    Returns:
        New state_dict with the same keys and shapes as template.
    """
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for k in sorted(template.keys()):
        shape = template[k].shape
        numel = template[k].numel()
        chunk = flat[offset : offset + numel].reshape(shape)
        # Preserve original dtype
        result[k] = chunk.to(template[k].dtype)
        offset += numel
    return result


def l2_distance(
    w1: dict[str, torch.Tensor],
    w2: dict[str, torch.Tensor],
) -> float:
    """L2 (Euclidean) distance between two flattened weight vectors."""
    diff = flatten_weights(w1) - flatten_weights(w2)
    return float(diff.norm(p=2).item())


def pairwise_distances(
    weight_list: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Compute pairwise L2 distances between a list of weight dicts.

    Uses torch.cdist for efficiency.

    Returns:
        (N, N) symmetric float32 tensor of pairwise distances.
    """
    if len(weight_list) == 0:
        return torch.zeros(0, 0)
    flat = torch.stack([flatten_weights(w) for w in weight_list])  # (N, D)
    return torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)  # (N, N)


def compute_tau_auto(
    weight_list: list[dict[str, torch.Tensor]],
    percentile: float = 50.0,
) -> float:
    """Estimate tau from the given percentile of all pairwise distances.

    The median-of-pairwise-distances heuristic is robust because, when the
    Byzantine fraction f < 0.5, the majority of pairwise distances are between
    honest nodes and thus the median reflects the honest update scale.

    Args:
        weight_list: All weights visible to a node (own + received neighbours).
        percentile: Which percentile to use (default 50 = median).

    Returns:
        tau as a positive float. Returns 1.0 as a fallback for degenerate cases.
    """
    if len(weight_list) < 2:
        return 1.0
    dists = pairwise_distances(weight_list)
    # Extract upper triangle (avoid double-counting and self-distances)
    N = dists.shape[0]
    upper = dists[torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)]
    if upper.numel() == 0:
        return 1.0
    tau = float(torch.quantile(upper.float(), percentile / 100.0).item())
    return max(tau, 1e-6)  # guard against zero-distance


# ---------------------------------------------------------------------------
# Gossip aggregation functions
# ---------------------------------------------------------------------------

def gossip_mean(
    own_weights: dict[str, torch.Tensor],
    neighbor_weights: list[dict[str, torch.Tensor]],
    include_self: bool = True,
) -> dict[str, torch.Tensor]:
    """Standard gossip: unweighted average of own and neighbour weights.

    No Byzantine resilience — one corrupted neighbour can skew the result
    arbitrarily.

    Args:
        own_weights: This node's current state_dict.
        neighbor_weights: List of received neighbour state_dicts.
        include_self: Whether to include own weights in the average (default True).

    Returns:
        Averaged state_dict with same keys/shapes as own_weights.
    """
    all_weights = ([own_weights] if include_self else []) + neighbor_weights
    if not all_weights:
        return copy.deepcopy(own_weights)

    flat_own = flatten_weights(own_weights)
    stacked = torch.stack([flatten_weights(w) for w in all_weights])  # (M, D)
    mean_flat = stacked.mean(dim=0)
    return unflatten_weights(mean_flat, own_weights)


def clipped_gossip(
    own_weights: dict[str, torch.Tensor],
    neighbor_weights: list[dict[str, torch.Tensor]],
    tau: Optional[float] = None,
    tau_percentile: float = 50.0,
) -> dict[str, torch.Tensor]:
    """ClippedGossip aggregation (Farhadkhani et al., NeurIPS 2022).

    Algorithm:
      1. Optionally compute tau from all weights (own + neighbours).
      2. For each neighbour j: compute delta_j = w_j - w_own.
      3. Clip delta_j to the L2 ball of radius tau:
             scale_j = min(1, tau / (||delta_j|| + eps))
             clipped_j = scale_j * delta_j
      4. result = w_own + mean(clipped_j for j in neighbours)

    Byzantine guarantee: each Byzantine neighbour can shift the aggregate by
    at most tau / (1 + |neighbours|) in any direction. With f Byzantine
    neighbours, total drift ≤ f × tau / (1 + |neighbours|).

    Args:
        own_weights: This node's current state_dict.
        neighbor_weights: List of received neighbour state_dicts.
        tau: Clipping radius. None → auto-computed from pairwise distances.
        tau_percentile: Percentile for auto-tau (default 50 = median).

    Returns:
        Aggregated state_dict.
    """
    if not neighbor_weights:
        return copy.deepcopy(own_weights)

    # Auto-select tau if not provided
    if tau is None:
        all_ws = [own_weights] + neighbor_weights
        tau = compute_tau_auto(all_ws, percentile=tau_percentile)

    flat_own = flatten_weights(own_weights)
    clipped_deltas = []

    for w_j in neighbor_weights:
        flat_j = flatten_weights(w_j)
        delta = flat_j - flat_own
        delta_norm = delta.norm(p=2).item()
        scale = min(1.0, tau / (delta_norm + 1e-8))
        clipped_deltas.append(scale * delta)

    if not clipped_deltas:
        return copy.deepcopy(own_weights)

    mean_clipped_delta = torch.stack(clipped_deltas).mean(dim=0)
    result_flat = flat_own + mean_clipped_delta
    return unflatten_weights(result_flat, own_weights)


def ssclip(
    own_weights: dict[str, torch.Tensor],
    neighbor_weights: list[dict[str, torch.Tensor]],
    tau: Optional[float] = None,
    tau_percentile: float = 50.0,
) -> dict[str, torch.Tensor]:
    """SSClip: Self-Centred Clipping (Fraboni et al., 2022).

    Stronger Byzantine guarantee than ClippedGossip: the output is always
    within tau of own_weights, independent of how many Byzantines there are
    (provided honest clients form the majority of the neighbourhood).

    Algorithm:
      1. Compute delta_j = w_j - w_own for each neighbour j.
      2. If ||delta_j|| > tau: clip delta_j to the sphere of radius tau.
             delta_j = tau × delta_j / ||delta_j||
      3. Reconstruct accepted neighbour: accepted_j = w_own + delta_j.
      4. Average own + all accepted: result = mean([w_own] + [accepted_j for j]).

    Key difference from ClippedGossip:
      - CG adds mean(clipped_delta) to own → output may drift far if many Byzantines
      - SSClip averages own with clipped reconstructions → bounded by construction

    Args:
        own_weights: This node's current state_dict.
        neighbor_weights: List of received neighbour state_dicts.
        tau: Clipping radius. None → auto-computed.
        tau_percentile: Percentile for auto-tau.

    Returns:
        Aggregated state_dict within tau of own_weights (L2 norm guarantee).
    """
    if not neighbor_weights:
        return copy.deepcopy(own_weights)

    if tau is None:
        all_ws = [own_weights] + neighbor_weights
        tau = compute_tau_auto(all_ws, percentile=tau_percentile)

    flat_own = flatten_weights(own_weights)
    accepted_flat = [flat_own]  # own weight is always included

    for w_j in neighbor_weights:
        flat_j = flatten_weights(w_j)
        delta = flat_j - flat_own
        delta_norm = delta.norm(p=2).item()
        if delta_norm > tau:
            delta = (tau / (delta_norm + 1e-8)) * delta
        accepted_flat.append(flat_own + delta)

    mean_flat = torch.stack(accepted_flat).mean(dim=0)
    return unflatten_weights(mean_flat, own_weights)


# ---------------------------------------------------------------------------
# Registry and selector
# ---------------------------------------------------------------------------

_AGGREGATION_FNS: dict[str, Callable] = {
    "mean": gossip_mean,
    "clipped_gossip": clipped_gossip,
    "ssclip": ssclip,
}


def select_gossip_fn(name: str) -> Callable:
    """Return the aggregation function by name.

    Args:
        name: One of "mean", "clipped_gossip", "ssclip".

    Returns:
        The aggregation callable.

    Raises:
        ValueError for unknown names.
    """
    if name not in _AGGREGATION_FNS:
        raise ValueError(
            f"Unknown aggregation '{name}'. Choose from {list(_AGGREGATION_FNS)}"
        )
    return _AGGREGATION_FNS[name]


# ---------------------------------------------------------------------------
# Neighbour selection
# ---------------------------------------------------------------------------

def compute_gossip_neighbors(
    client_id: int,
    graph: nx.Graph,
    fanout: int,
    rng: np.random.Generator,
) -> list[int]:
    """Select up to fanout neighbours for a gossip push round.

    Strategy:
      1. Use direct graph neighbours first (physical proximity).
      2. If fewer than fanout direct neighbours exist, expand to 2-hop
         neighbours to ensure minimum connectivity for sparse graphs.

    Args:
        client_id: Source robot's ID.
        graph: Current communication graph.
        fanout: Target number of neighbours (k).
        rng: Numpy random generator (for reproducible selection).

    Returns:
        List of up to fanout neighbour IDs (never includes client_id itself).
    """
    direct = [n for n in graph.neighbors(client_id)]
    if len(direct) >= fanout:
        chosen = rng.choice(direct, size=fanout, replace=False).tolist()
        return [int(x) for x in chosen]

    # Expand to 2-hop neighbours
    two_hop: set[int] = set()
    for n in direct:
        two_hop.update(graph.neighbors(n))
    two_hop.discard(client_id)
    two_hop -= set(direct)

    candidates = direct + list(two_hop)
    if not candidates:
        return []

    n_select = min(fanout, len(candidates))
    chosen = rng.choice(candidates, size=n_select, replace=False).tolist()
    return [int(x) for x in chosen]
