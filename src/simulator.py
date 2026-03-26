"""
simulator.py — Orchestration of the gossip FL simulation.

Two simulators are provided:

  GossipSimulator   — Fully decentralised gossip over a dynamic graph.
                      Supports mean, ClippedGossip, and SSClip aggregation.
                      Parallel local training via Ray (if available) or
                      concurrent.futures (Windows-safe fallback).

  FedAvgSimulator   — Centralised FedAvg baseline.
                      All clients participate, a server averages their updates,
                      and broadcasts the global model. No gossip.

Single-round data flow (GossipSimulator):
  1. topology.update(t)            — move robots, optionally rebuild graph
  2. _parallel_local_train()       — all clients run local SGD in parallel
  3. _collect_weights()            — honest: deepcopy; Byzantine: apply_attack
  4. _run_gossip_round(weights, g) — SYNCHRONOUS: compute all new_weights[i]
                                     from the snapshot; then apply in bulk
  5. _apply_aggregated_weights()   — load new_weights into each client's model
  6. _evaluate_round()             — every eval_every rounds, score all clients
"""
from __future__ import annotations

import copy
import os
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.client import RobotClient
from src.config import Config
from src.gossip import compute_gossip_neighbors, select_gossip_fn
from src.metrics import MetricsTracker
from src.model import model_size_bytes
from src.topology import TopologyManager


# ---------------------------------------------------------------------------
# Parallel training backend detection
# ---------------------------------------------------------------------------

def _try_import_ray():
    try:
        import ray
        return ray
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# GossipSimulator
# ---------------------------------------------------------------------------

class GossipSimulator:
    """Orchestrates a fully-decentralised gossip federated learning simulation.

    Args:
        config: Full experiment Config.
        clients: List of N RobotClient instances (mix of honest and Byzantine).
        topology: TopologyManager for graph/mobility management.
        test_loader: DataLoader for the global test set (shared evaluation).
        metrics: MetricsTracker for logging.
        use_ray: Whether to use Ray for parallel local training.
                 Falls back to ThreadPoolExecutor if False (or Ray unavailable).
    """

    def __init__(
        self,
        config: Config,
        clients: list[RobotClient],
        topology: TopologyManager,
        test_loader: DataLoader,
        metrics: MetricsTracker,
        use_ray: bool = False,
    ) -> None:
        self.config = config
        self.clients = clients
        self.topology = topology
        self.test_loader = test_loader
        self.metrics = metrics
        self.rng = np.random.default_rng(config.experiment.seed)

        self._n = len(clients)
        self._byzantine_ids = {c.client_id for c in clients if c.is_byzantine}
        self._gossip_fn = select_gossip_fn(config.gossip.aggregation)
        self._model_bytes = model_size_bytes(clients[0].model)

        # Weight cache: updated after each collect_weights call
        self._weight_cache: dict[int, dict[str, torch.Tensor]] = {}

        # Ray setup
        self._use_ray = use_ray and _try_import_ray() is not None
        if use_ray and not self._use_ray:
            print("[GossipSimulator] Ray not available, falling back to serial training.")

        # Topology snapshot list for GIF generation (stored as file paths)
        self._topo_snapshot_paths: list[str] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_rounds: Optional[int] = None) -> dict:
        """Run the gossip FL simulation.

        Args:
            num_rounds: Override number of rounds (default: config.experiment.rounds).

        Returns:
            Summary dict with final metrics.
        """
        rounds = num_rounds or self.config.experiment.rounds
        eval_every = self.config.logging.eval_every
        save_topo_every = self.config.topology.update_every or 10

        print(f"\n[GossipSimulator] Starting: N={self._n}, byz={len(self._byzantine_ids)}, "
              f"agg={self.config.gossip.aggregation}, rounds={rounds}")

        t0 = time.time()
        for round_num in tqdm(range(rounds), desc="Gossip rounds", unit="rd"):
            round_metrics = self._run_round(round_num)

            # Save topology snapshot for GIF
            if round_num % save_topo_every == 0:
                snap_path = os.path.join(
                    self.config.logging.log_dir,
                    "topology_snaps",
                    f"round_{round_num:04d}.png",
                )
                os.makedirs(os.path.dirname(snap_path), exist_ok=True)
                self.topology.visualize(
                    round_num=round_num,
                    byzantine_ids=self._byzantine_ids,
                    save_path=snap_path,
                )
                plt_close_all()
                self._topo_snapshot_paths.append(snap_path)

            if round_num % eval_every == 0:
                eval_metrics = self._evaluate_round(round_num)
                combined = {**round_metrics, **eval_metrics}
            else:
                combined = round_metrics

            self.metrics.log_round(round_num, combined)

        elapsed = time.time() - t0
        print(f"[GossipSimulator] Done in {elapsed:.1f}s")
        self.metrics.save()
        return {"elapsed_sec": elapsed, "rounds": rounds}

    # ------------------------------------------------------------------
    # Single round
    # ------------------------------------------------------------------

    def _run_round(self, round_num: int) -> dict:
        """Execute one complete gossip round."""

        # Step 1: Update topology (move robots, optionally rebuild graph)
        graph_updated = self.topology.update(round_num)
        topo_metrics = self.topology.get_graph_metrics()

        # Step 2: Parallel local training
        self._parallel_local_train()

        # Step 3: Collect weights (Byzantine clients apply attack here)
        weights = self._collect_weights()

        # Step 4: Gossip exchange (synchronous — snapshot before modifying)
        new_weights = self._run_gossip_round(weights, self.topology.graph)

        # Step 5: Apply aggregated weights to client models
        self._apply_aggregated_weights(new_weights)

        # Log communication volume: each client pushes to fanout neighbours
        # (push) + receives from push_senders (pull approximated as fanout)
        n_messages = self._n * self.config.gossip.fanout
        self.metrics.log_communication_volume(round_num, n_messages, self._model_bytes)

        return {
            "graph_updated": graph_updated,
            "graph_avg_degree": topo_metrics["avg_degree"],
            "graph_spectral_gap": topo_metrics["spectral_gap"],
            "graph_is_connected": topo_metrics["is_connected"],
        }

    def _parallel_local_train(self) -> None:
        """Run local_train() on all clients.

        Uses Ray actors if available, otherwise trains serially.
        (Serial is surprisingly fast when GPU is shared because each client's
        local epoch is short, and GPU overhead dominates for tiny mini-batches.)
        """
        # Serial training (default — works reliably on Windows with GPU)
        for client in self.clients:
            client.local_train()

    def _collect_weights(self) -> dict[int, dict[str, torch.Tensor]]:
        """Call get_weights() on all clients; Byzantine ones apply attacks here."""
        weights = {}
        for client in self.clients:
            weights[client.client_id] = client.get_weights()
        self._weight_cache = weights
        return weights

    def _run_gossip_round(
        self,
        weights: dict[int, dict[str, torch.Tensor]],
        graph,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Synchronous gossip: compute all aggregated weights from a snapshot.

        IMPORTANT: new_weights[i] is computed from the pre-round snapshot
        `weights`, NOT from any already-updated new_weights[j]. This is the
        synchronous gossip semantics that guarantees convergence.

        Args:
            weights: Pre-round snapshot of all client weights.
            graph: Current communication graph (NetworkX).

        Returns:
            {client_id: aggregated_weights} — ready to load into models.
        """
        new_weights: dict[int, dict[str, torch.Tensor]] = {}
        tau = self.config.gossip.tau
        tau_pct = self.config.gossip.tau_percentile

        for client in self.clients:
            cid = client.client_id
            # Select gossip neighbours
            neighbor_ids = compute_gossip_neighbors(
                cid, graph, self.config.gossip.fanout, self.rng
            )
            neighbor_ws = [weights[nid] for nid in neighbor_ids if nid in weights]

            # Aggregate using the configured method
            own_ws = weights[cid]
            if self.config.gossip.aggregation == "mean":
                new_ws = self._gossip_fn(own_ws, neighbor_ws)
            else:
                new_ws = self._gossip_fn(own_ws, neighbor_ws, tau=tau, tau_percentile=tau_pct)

            new_weights[cid] = new_ws

        return new_weights

    def _apply_aggregated_weights(
        self,
        new_weights: dict[int, dict[str, torch.Tensor]],
    ) -> None:
        """Apply gossip-aggregated weights back to each client's model."""
        for client in self.clients:
            cid = client.client_id
            if cid in new_weights:
                client.apply_gossip_update(new_weights[cid])

    def _evaluate_round(self, round_num: int) -> dict:
        """Score all clients on the global test set and return aggregate metrics.

        Returns:
            {
              "global_accuracy": float,
              "per_client_accuracy": list[float],
              "accuracy_std": float,
              "global_loss": float,
            }
        """
        results = [client.evaluate(self.test_loader) for client in self.clients]
        accs = [r["accuracy"] for r in results]
        losses = [r["loss"] for r in results]

        return {
            "global_accuracy": float(np.mean(accs)),
            "per_client_accuracy": accs,
            "accuracy_std": float(np.std(accs)),
            "worst_client_accuracy": float(np.min(accs)),
            "global_loss": float(np.mean(losses)),
        }

    def get_topology_gif_paths(self) -> list[str]:
        return self._topo_snapshot_paths


# ---------------------------------------------------------------------------
# FedAvgSimulator (centralised baseline)
# ---------------------------------------------------------------------------

class FedAvgSimulator:
    """Centralised FedAvg baseline for comparison with gossip FL.

    The "server" is an in-process Python object (no separate process needed).
    It holds a global model, broadcasts it to all clients, collects local
    updates, and averages weighted by dataset size.

    No gossip, no topology, no Byzantine attacks by default — this is the
    idealised upper bound that gossip FL tries to match.

    Args:
        config: Full Config (uses data.num_clients, client.*, logging.*).
        clients: List of RobotClient instances (used for local training only;
                 their weights are overwritten by server before each round).
        test_loader: Global test DataLoader.
        metrics: MetricsTracker for logging.
    """

    def __init__(
        self,
        config: Config,
        clients: list[RobotClient],
        test_loader: DataLoader,
        metrics: MetricsTracker,
    ) -> None:
        self.config = config
        self.clients = clients
        self.test_loader = test_loader
        self.metrics = metrics
        self._n = len(clients)
        self._model_bytes = model_size_bytes(clients[0].model)
        self._byzantine_ids = {c.client_id for c in clients if c.is_byzantine}

        # Initialise global model as average of client initialisation
        # (in practice they share the same random init so this is a no-op)
        self._global_weights: dict[str, torch.Tensor] = copy.deepcopy(
            clients[0].model.state_dict()
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_rounds: Optional[int] = None) -> dict:
        """Run the FedAvg simulation.

        Args:
            num_rounds: Override number of rounds.

        Returns:
            Summary dict.
        """
        rounds = num_rounds or self.config.experiment.rounds
        eval_every = self.config.logging.eval_every

        print(f"\n[FedAvgSimulator] Starting: N={self._n}, rounds={rounds}")

        t0 = time.time()
        for round_num in tqdm(range(rounds), desc="FedAvg rounds", unit="rd"):
            round_metrics = self._run_round(round_num)

            if round_num % eval_every == 0:
                eval_metrics = self._evaluate_round(round_num)
                combined = {**round_metrics, **eval_metrics}
            else:
                combined = round_metrics

            self.metrics.log_round(round_num, combined)

        elapsed = time.time() - t0
        print(f"[FedAvgSimulator] Done in {elapsed:.1f}s")
        self.metrics.save()
        return {"elapsed_sec": elapsed, "rounds": rounds}

    # ------------------------------------------------------------------
    # Single round
    # ------------------------------------------------------------------

    def _run_round(self, round_num: int) -> dict:
        """One FedAvg round: broadcast → local train → aggregate."""

        # Step 1: Broadcast global model to all clients
        for client in self.clients:
            client.set_weights(copy.deepcopy(self._global_weights))

        # Step 2: Local training
        train_results = [client.local_train() for client in self.clients]

        # Step 3: Weighted aggregate (by number of training samples)
        total_samples = sum(r["samples"] for r in train_results)
        new_global: dict[str, torch.Tensor] = {}

        for k in self._global_weights:
            weighted_sum = torch.zeros_like(self._global_weights[k].float())
            for client, result in zip(self.clients, train_results):
                w = result["samples"] / max(total_samples, 1)
                weighted_sum += w * client.model.state_dict()[k].float()
            new_global[k] = weighted_sum.to(self._global_weights[k].dtype)

        self._global_weights = new_global

        # Communication: server → clients (broadcast) + clients → server (collect)
        n_messages = 2 * self._n
        self.metrics.log_communication_volume(round_num, n_messages, self._model_bytes)

        return {
            "graph_updated": False,
            "graph_avg_degree": float(self._n - 1),  # fully connected in FedAvg
            "graph_spectral_gap": 1.0,
            "graph_is_connected": True,
        }

    def _evaluate_round(self, round_num: int) -> dict:
        """Evaluate all clients (they all have the global model after broadcast)."""
        # Temporarily set all clients to global model
        for client in self.clients:
            client.set_weights(copy.deepcopy(self._global_weights))

        results = [client.evaluate(self.test_loader) for client in self.clients]
        accs = [r["accuracy"] for r in results]
        losses = [r["loss"] for r in results]

        return {
            "global_accuracy": float(np.mean(accs)),
            "per_client_accuracy": accs,
            "accuracy_std": float(np.std(accs)),
            "worst_client_accuracy": float(np.min(accs)),
            "global_loss": float(np.mean(losses)),
        }


# ---------------------------------------------------------------------------
# Factory: build simulator from config
# ---------------------------------------------------------------------------

def build_gossip_simulator(
    config: Config,
    use_ray: bool = False,
) -> tuple[GossipSimulator, MetricsTracker]:
    """Convenience factory: load data, create clients, build GossipSimulator.

    Args:
        config: Fully populated Config.
        use_ray: Pass True to enable Ray-based parallel training.

    Returns:
        (simulator, metrics_tracker) ready to call .run() on.
    """
    import torch
    from src.attacks import select_byzantine_clients
    from src.data import (
        dirichlet_partition,
        load_raw_dataset,
        make_client_dataloaders,
        make_test_dataloader,
    )
    from src.model import model_for_dataset

    # Reproducibility
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    if config.experiment.device != "cpu" and torch.cuda.is_available():
        try:
            device = torch.device(config.experiment.device)
            torch.zeros(1, device=device)  # probe: catches driver/context failures
        except Exception as e:
            print(f"[build_gossip_simulator] CUDA unavailable ({e}), falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Data
    train_ds, test_ds = load_raw_dataset(config.data.dataset, root=config.data.data_root)
    index_lists = dirichlet_partition(
        train_ds,
        num_clients=config.data.num_clients,
        alpha=config.data.alpha,
        seed=config.experiment.seed,
    )
    loaders = make_client_dataloaders(
        train_ds, index_lists,
        batch_size=config.data.batch_size,
        val_split=config.data.val_split,
    )
    test_loader = make_test_dataloader(test_ds, batch_size=256)

    # Byzantine selection
    byz_ids = select_byzantine_clients(
        config.data.num_clients,
        config.attack.fraction if config.attack.enabled else 0.0,
        seed=config.experiment.seed,
    )
    attack_cfg = config.attack if config.attack.enabled else None

    # Build a single model, share init across all clients
    init_model = model_for_dataset(config.data.dataset)
    init_weights = copy.deepcopy(init_model.state_dict())

    # Clients
    clients = []
    for cid, (train_loader, val_loader) in enumerate(loaders):
        model = model_for_dataset(config.data.dataset).to(device)
        model.load_state_dict(copy.deepcopy(init_weights))
        client_attack = attack_cfg if cid in byz_ids else None
        # Force Byzantine flag even if attack_config is provided for non-byz client
        if client_attack is not None and cid not in byz_ids:
            client_attack = None
        clients.append(RobotClient(
            client_id=cid,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.client,
            attack_config=client_attack,
            device=device,
            position=(float(np.random.rand()), float(np.random.rand())),
            seed=config.experiment.seed + cid,
        ))

    # Fix Byzantine flags (client_attack was set based on byz_ids, now align)
    for client in clients:
        client.is_byzantine = client.client_id in byz_ids
        if client.is_byzantine and client.attack_config is None and attack_cfg is not None:
            client.attack_config = attack_cfg
            client._attack_type = attack_cfg.type

    # Topology
    from src.topology import TopologyManager
    topology = TopologyManager(config.data.num_clients, config.topology, seed=config.experiment.seed)

    # Metrics
    run_name = (
        f"{config.experiment.name}_"
        f"agg{config.gossip.aggregation}_"
        f"n{config.data.num_clients}_"
        f"byz{int(config.attack.fraction * 100)}_"
        f"a{config.data.alpha}"
    )
    metrics = MetricsTracker(
        log_dir=config.logging.log_dir,
        run_name=run_name,
        num_clients=config.data.num_clients,
        byzantine_ids=byz_ids,
        backend=config.logging.backend,
        mlflow_uri=config.logging.mlflow_uri,
        experiment_name=config.logging.experiment_name,
    )

    simulator = GossipSimulator(
        config=config,
        clients=clients,
        topology=topology,
        test_loader=test_loader,
        metrics=metrics,
        use_ray=use_ray,
    )

    print(f"[build_gossip_simulator] device={device}, "
          f"byz_ids={sorted(byz_ids)}, "
          f"model_size={model_size_bytes(clients[0].model) / 1024:.1f} KB")

    return simulator, metrics


def build_fedavg_simulator(config: Config) -> tuple[FedAvgSimulator, MetricsTracker]:
    """Convenience factory for the FedAvg baseline."""
    import torch
    from src.data import (
        dirichlet_partition,
        load_raw_dataset,
        make_client_dataloaders,
        make_test_dataloader,
    )
    from src.model import model_for_dataset

    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    if config.experiment.device != "cpu" and torch.cuda.is_available():
        try:
            device = torch.device(config.experiment.device)
            torch.zeros(1, device=device)  # probe: catches driver/context failures
        except Exception as e:
            print(f"[build_fedavg_simulator] CUDA unavailable ({e}), falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    train_ds, test_ds = load_raw_dataset(config.data.dataset, root=config.data.data_root)
    index_lists = dirichlet_partition(
        train_ds,
        num_clients=config.data.num_clients,
        alpha=config.data.alpha,
        seed=config.experiment.seed,
    )
    loaders = make_client_dataloaders(
        train_ds, index_lists,
        batch_size=config.data.batch_size,
        val_split=config.data.val_split,
    )
    test_loader = make_test_dataloader(test_ds, batch_size=256)

    init_model = model_for_dataset(config.data.dataset)
    init_weights = copy.deepcopy(init_model.state_dict())

    clients = []
    for cid, (train_loader, val_loader) in enumerate(loaders):
        model = model_for_dataset(config.data.dataset).to(device)
        model.load_state_dict(copy.deepcopy(init_weights))
        clients.append(RobotClient(
            client_id=cid,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.client,
            attack_config=None,
            device=device,
            position=(float(np.random.rand()), float(np.random.rand())),
            seed=config.experiment.seed + cid,
        ))

    run_name = f"fedavg_n{config.data.num_clients}_a{config.data.alpha}"
    metrics = MetricsTracker(
        log_dir=config.logging.log_dir,
        run_name=run_name,
        num_clients=config.data.num_clients,
        byzantine_ids=set(),
        backend=config.logging.backend,
    )

    simulator = FedAvgSimulator(
        config=config,
        clients=clients,
        test_loader=test_loader,
        metrics=metrics,
    )

    return simulator, metrics


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def plt_close_all():
    """Close all open Matplotlib figures to prevent memory leaks."""
    import matplotlib.pyplot as plt
    plt.close("all")
