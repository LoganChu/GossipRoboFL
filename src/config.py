"""
config.py — Typed configuration dataclasses for GossipRoboFL.

All hyperparameters live in YAML files under configs/. This module provides
the Python-side type-safe mirror of those configs via dataclasses, loaded
with dacite + PyYAML. Hydra can optionally override values at runtime.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 42
    rounds: int = 200
    device: str = "cuda"


@dataclass
class DataConfig:
    dataset: str = "cifar10"       # "cifar10" | "fashion_mnist"
    num_clients: int = 20
    alpha: float = 0.5             # Dirichlet concentration
    batch_size: int = 64
    val_split: float = 0.1
    data_root: str = "data/"


@dataclass
class ClientConfig:
    local_epochs: int = 3
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    heterogeneous: bool = False    # vary epochs/lr per robot (straggler simulation)
    min_epochs: int = 1
    max_epochs: int = 5
    delay_scale: float = 0.0       # exponential delay mean in seconds (0 = disabled)


@dataclass
class GossipConfig:
    fanout: int = 4                # k neighbors to push to per round
    aggregation: str = "ssclip"   # "mean" | "clipped_gossip" | "ssclip"
    tau: Optional[float] = None   # None = auto-select from pairwise distances
    tau_percentile: float = 50.0  # percentile of pairwise dists used for auto-tau
    # Encounter-mode parameters (EncounterGossipSimulator only)
    steps_per_round: int = 1      # micro-steps per logical round
    train_every_steps: int = 1    # base training interval (steps)
    async_train: bool = False     # Poisson-sampled training intervals per robot


@dataclass
class MobilityConfig:
    enabled: bool = True
    step_size: float = 0.05       # max displacement per axis per round
    boundary: str = "reflect"     # "reflect" | "wrap"


@dataclass
class TopologyConfig:
    type: str = "random_geometric"  # "random_geometric" | "knn" | "erdos_renyi"
    comm_range: float = 0.4         # radius r for random geometric graph
    k_nearest: int = 5              # k for KNN graph
    er_prob: float = 0.2            # edge probability for Erdos-Renyi
    update_every: int = 5           # rebuild graph every N rounds (0 = static)
    drop_prob: float = 0.0          # message drop probability for encounter mode
    mobility: MobilityConfig = field(default_factory=MobilityConfig)


@dataclass
class AttackConfig:
    enabled: bool = False
    type: str = "sign_flip"        # "sign_flip"|"random_noise"|"label_flip"|"gaussian_perturb"|"partial_knowledge"
    fraction: float = 0.2          # fraction of clients that are Byzantine
    noise_scale: float = 10.0      # scale for random_noise / gaussian_perturb
    sign_scale: float = 1.0        # multiplier for sign_flip negation


@dataclass
class LoggingConfig:
    backend: str = "json"          # "json" | "mlflow" | "both"
    log_dir: str = "results/"
    eval_every: int = 5            # evaluate test accuracy every N rounds
    save_model_every: int = 50     # checkpoint every N rounds (0 = disabled)
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "gossiprobofl"
    topo_snap_every: int = 20       # snapshot topology every N rounds; 0 = disabled


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    gossip: GossipConfig = field(default_factory=GossipConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict (overrides win)."""
    result = copy.deepcopy(base)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _dict_to_config(d: dict) -> Config:
    """Convert a nested dict into a typed Config dataclass using dacite if available,
    otherwise fall back to manual construction."""
    try:
        import dacite
        return dacite.from_dict(
            data_class=Config,
            data=d,
            config=dacite.Config(strict=False),
        )
    except ImportError:
        # Manual fallback — construct sub-configs from nested dicts
        exp_d = d.get("experiment", {})
        data_d = d.get("data", {})
        client_d = d.get("client", {})
        gossip_d = d.get("gossip", {})
        topo_d = d.get("topology", {})
        mob_d = topo_d.pop("mobility", {}) if "mobility" in topo_d else {}
        attack_d = d.get("attack", {})
        log_d = d.get("logging", {})

        return Config(
            experiment=ExperimentConfig(**{k: v for k, v in exp_d.items() if hasattr(ExperimentConfig, k)}),
            data=DataConfig(**{k: v for k, v in data_d.items() if hasattr(DataConfig, k)}),
            client=ClientConfig(**{k: v for k, v in client_d.items() if hasattr(ClientConfig, k)}),
            gossip=GossipConfig(**{k: v for k, v in gossip_d.items() if hasattr(GossipConfig, k)}),
            topology=TopologyConfig(
                **{k: v for k, v in topo_d.items() if k != "mobility" and hasattr(TopologyConfig, k)},
                mobility=MobilityConfig(**{k: v for k, v in mob_d.items() if hasattr(MobilityConfig, k)}),
            ),
            attack=AttackConfig(**{k: v for k, v in attack_d.items() if hasattr(AttackConfig, k)}),
            logging=LoggingConfig(**{k: v for k, v in log_d.items() if hasattr(LoggingConfig, k)}),
        )


def load_config(path: str, overrides: Optional[dict] = None) -> Config:
    """Load a YAML config file and optionally apply override dict.

    Args:
        path: Path to the YAML config file (e.g., "configs/default.yaml").
        overrides: Optional flat or nested dict of overrides, e.g.,
                   {"data": {"num_clients": 50}, "attack": {"enabled": True}}.

    Returns:
        Fully populated Config dataclass.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    return _dict_to_config(raw)


def config_to_dict(cfg: Config) -> dict:
    """Convert Config dataclass back to a plain dict (for JSON serialization)."""
    import dataclasses
    return dataclasses.asdict(cfg)
