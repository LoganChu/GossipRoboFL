"""
encounter_gossip.py — DTN-style encounter-based gossip simulator.

Unlike GossipSimulator (synchronous rounds with forced connectivity), here:
  - Robots move at every micro-step on their own.
  - Gossip only triggers when two robots physically enter comm_range (an encounter).
  - Training happens on each robot's own schedule (async).
  - The proximity graph is never forced connected; partitions are natural.
  - Messages can be dropped with probability drop_prob (unreliable wireless).

This models the real-world robot swarm scenario:
  robots communicate opportunistically over lossy wireless, with no global clock.
  When isolated (no neighbors), a robot carries its model weights until it drifts
  back into contact range — the store-carry-forward property of DTN.

A "logical round" = steps_per_round micro-steps, used only for evaluation cadence.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.client import RobotClient
from src.config import Config
from src.gossip import select_gossip_fn
from src.metrics import MetricsTracker
from src.model import model_size_bytes
from src.topology import TopologyManager


class EncounterGossipSimulator:
    """
    Encounter-based gossip FL for robot swarms.

    Each micro-step:
      1. All robots move one step (topology.step).
      2. An encounter graph is built from raw proximity — no forced connectivity.
      3. Robots whose training timer has fired run local SGD.
      4. Weight snapshots are taken (Byzantine clients corrupt here).
      5. For each robot with encounter neighbors: gossip aggregation using snapshot.
      6. Robots with no encounters keep their current weights (store-carry-forward).

    New metrics vs GossipSimulator:
      - graph_num_components: how many disconnected clusters exist
      - graph_largest_component_frac: fraction of robots in the largest cluster
      - mean_isolation_steps / max_isolation_steps: how long robots go without contact
    """

    def __init__(
        self,
        config: Config,
        clients: list[RobotClient],
        topology: TopologyManager,
        test_loader: DataLoader,
        metrics: MetricsTracker,
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

        self._steps_per_round = config.gossip.steps_per_round
        self._train_every = config.gossip.train_every_steps
        self._drop_prob = config.topology.drop_prob

        # Per-robot training schedule: step number when each robot next trains
        if config.gossip.async_train:
            # Stagger initial training times so robots don't all train on step 0
            self._next_train: list[int] = [
                int(self.rng.integers(0, max(1, self._train_every)))
                for _ in range(self._n)
            ]
        else:
            self._next_train = [0] * self._n

        # Steps since last encounter per robot (0 = had an encounter this step)
        self._steps_isolated: list[int] = [0] * self._n

        self._total_messages = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_rounds: Optional[int] = None) -> dict:
        rounds = num_rounds or self.config.experiment.rounds
        eval_every = self.config.logging.eval_every
        save_topo_every = self.config.logging.topo_snap_every
        total_steps = rounds * self._steps_per_round

        print(
            f"\n[EncounterGossipSimulator] Starting: N={self._n}, "
            f"byz={len(self._byzantine_ids)}, agg={self.config.gossip.aggregation}, "
            f"rounds={rounds}, steps/round={self._steps_per_round}, "
            f"drop_prob={self._drop_prob:.2f}, async_train={self.config.gossip.async_train}"
        )

        t0 = time.time()
        for step in tqdm(range(total_steps), desc="Encounter steps", unit="step"):
            round_num = step // self._steps_per_round

            encounter_graph, messages = self._run_step(step)
            self._total_messages += messages

            # Only log at the first step of each logical round
            if step % self._steps_per_round == 0:
                topo_metrics = self._get_encounter_metrics(encounter_graph)
                self.metrics.log_communication_volume(round_num, messages, self._model_bytes)

                if save_topo_every > 0 and round_num % save_topo_every == 0:
                    self._save_topology_snapshot(encounter_graph, round_num)

                if round_num % eval_every == 0:
                    eval_metrics = self._evaluate_round(round_num)
                    combined = {**topo_metrics, **eval_metrics}
                else:
                    combined = topo_metrics

                self.metrics.log_round(round_num, combined)

        elapsed = time.time() - t0
        print(f"[EncounterGossipSimulator] Done in {elapsed:.1f}s, "
              f"total_messages={self._total_messages}")
        self.metrics.save()
        return {"elapsed_sec": elapsed, "rounds": rounds, "total_messages": self._total_messages}

    # ------------------------------------------------------------------
    # Single micro-step
    # ------------------------------------------------------------------

    def _run_step(self, step: int) -> tuple[nx.Graph, int]:
        """Execute one micro-step. Returns (encounter_graph, messages_sent)."""

        # Step 1: Move robots (no graph rebuild — encounter graph is built fresh below)
        self.topology.step()

        # Step 2: Build raw proximity graph (no _ensure_connected)
        encounter_graph = self._build_encounter_graph()

        # Step 3: Async training — each robot trains when its timer fires
        for client in self.clients:
            cid = client.client_id
            if step >= self._next_train[cid]:
                client.local_train()
                self._next_train[cid] = step + self._sample_train_interval(cid)

        # Step 4: Snapshot all weights (Byzantine clients corrupt on get_weights)
        weight_snapshot: dict[int, dict[str, torch.Tensor]] = {
            c.client_id: c.get_weights() for c in self.clients
        }

        # Step 5: Gossip for robots that have encounter neighbors
        new_weights, messages = self._process_encounters(encounter_graph, weight_snapshot)

        # Step 6: Apply aggregated weights; isolated robots keep their current weights
        for client in self.clients:
            cid = client.client_id
            if cid in new_weights:
                client.apply_gossip_update(new_weights[cid])

        # Step 7: Update isolation counters
        encountered: set[int] = set()
        for u, v in encounter_graph.edges():
            encountered.add(u)
            encountered.add(v)
        for cid in range(self._n):
            if cid in encountered:
                self._steps_isolated[cid] = 0
            else:
                self._steps_isolated[cid] += 1

        return encounter_graph, messages

    # ------------------------------------------------------------------
    # Encounter graph (no forced connectivity)
    # ------------------------------------------------------------------

    def _build_encounter_graph(self) -> nx.Graph:
        """Build proximity graph from current positions without forcing connectivity.

        Robots that are out of comm_range of all others are genuinely isolated.
        This is the key difference from TopologyManager._build_random_geometric,
        which calls _ensure_connected() afterward.
        """
        g = nx.Graph()
        g.add_nodes_from(range(self._n))

        pos = self.topology.positions
        coords = np.array([pos[i] for i in range(self._n)])
        r = self.topology.config.comm_range
        tree = cKDTree(coords)
        for i, j in tree.query_pairs(r):
            g.add_edge(i, j)
        return g

    # ------------------------------------------------------------------
    # Gossip on encounters
    # ------------------------------------------------------------------

    def _process_encounters(
        self,
        encounter_graph: nx.Graph,
        weight_snapshot: dict[int, dict[str, torch.Tensor]],
    ) -> tuple[dict[int, dict[str, torch.Tensor]], int]:
        """Compute gossip aggregations for all robots that have encounter neighbors.

        Uses the weight snapshot taken before this step so that all aggregations
        are based on the same pre-step state (no intra-step cascade).

        Args:
            encounter_graph: Raw proximity graph for this step.
            weight_snapshot: {client_id: weights} captured before gossip.

        Returns:
            ({client_id: new_weights} for robots with at least one live neighbor,
             total messages sent)
        """
        new_weights: dict[int, dict[str, torch.Tensor]] = {}
        tau = self.config.gossip.tau
        tau_pct = self.config.gossip.tau_percentile
        messages = 0

        for client in self.clients:
            cid = client.client_id
            neighbors = list(encounter_graph.neighbors(cid))
            if not neighbors:
                continue  # isolated: carry current weights, no gossip

            # Fanout limit: randomly select up to fanout neighbors
            fanout = self.config.gossip.fanout
            if len(neighbors) > fanout:
                neighbors = self.rng.choice(neighbors, size=fanout, replace=False).tolist()

            # Apply message drop probability (unreliable wireless channel)
            live_neighbors = [
                nid for nid in neighbors
                if self._drop_prob == 0.0 or self.rng.random() >= self._drop_prob
            ]
            messages += len(live_neighbors)

            if not live_neighbors:
                continue  # all messages dropped; robot keeps its weights

            neighbor_ws = [weight_snapshot[nid] for nid in live_neighbors]
            own_ws = weight_snapshot[cid]

            if self.config.gossip.aggregation == "mean":
                new_weights[cid] = self._gossip_fn(own_ws, neighbor_ws)
            else:
                new_weights[cid] = self._gossip_fn(
                    own_ws, neighbor_ws, tau=tau, tau_percentile=tau_pct
                )

        return new_weights, messages

    # ------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------

    def _sample_train_interval(self, client_id: int) -> int:
        """Return steps until this robot trains again."""
        base = self._train_every
        if self.config.gossip.async_train:
            # Poisson inter-arrival: exponential distribution, mean = base
            return max(1, int(self.rng.exponential(base)))
        return base

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _get_encounter_metrics(self, encounter_graph: nx.Graph) -> dict:
        """Compute topology metrics from the natural (unforced) encounter graph."""
        g = encounter_graph
        num_comp = nx.number_connected_components(g)
        is_conn = num_comp <= 1
        avg_degree = float(np.mean([d for _, d in g.degree()]))

        largest_frac = (
            len(max(nx.connected_components(g), key=len)) / self._n
            if self._n > 0 else 0.0
        )

        if is_conn and g.number_of_nodes() > 1:
            try:
                L = nx.laplacian_matrix(g).toarray().astype(float)
                eigs = np.sort(np.linalg.eigvalsh(L))
                spec_gap = float(eigs[1])
            except Exception:
                spec_gap = 0.0
        else:
            spec_gap = 0.0

        return {
            "graph_updated": True,
            "graph_avg_degree": avg_degree,
            "graph_spectral_gap": spec_gap,
            "graph_is_connected": is_conn,
            "graph_num_components": num_comp,
            "graph_largest_component_frac": largest_frac,
            "mean_isolation_steps": float(np.mean(self._steps_isolated)),
            "max_isolation_steps": float(np.max(self._steps_isolated)),
        }

    def _evaluate_round(self, round_num: int) -> dict:
        results = [client.evaluate(self.test_loader) for client in self.clients]
        accs = [r["accuracy"] for r in results]
        losses = [r["loss"] for r in results]
        honest_accs = [
            r["accuracy"] for r in results
            if r["client_id"] not in self._byzantine_ids
        ]
        return {
            "global_accuracy": float(np.mean(accs)),
            "honest_global_accuracy": float(np.mean(honest_accs)) if honest_accs else 0.0,
            "per_client_accuracy": accs,
            "accuracy_std": float(np.std(accs)),
            "worst_client_accuracy": float(np.min(accs)),
            "global_loss": float(np.mean(losses)),
        }

    # ------------------------------------------------------------------
    # Topology snapshot
    # ------------------------------------------------------------------

    def _save_topology_snapshot(self, encounter_graph: nx.Graph, round_num: int) -> None:
        """Save a visualization of the encounter graph, colored by component."""
        snap_path = os.path.join(
            self.config.logging.log_dir,
            "topology_snaps",
            self.config.experiment.name,
            f"round_{round_num:04d}.png",
        )
        os.makedirs(os.path.dirname(snap_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        pos = self.topology.positions

        components = list(nx.connected_components(encounter_graph))
        component_colors = plt.cm.tab10(np.linspace(0, 1, max(len(components), 1)))

        for u, v in encounter_graph.edges():
            xu, yu = pos[u]
            xv, yv = pos[v]
            ax.plot([xu, xv], [yu, yv], color="#cccccc", linewidth=0.7, zorder=1)

        for comp_idx, comp in enumerate(components):
            color = "#e74c3c" if comp & self._byzantine_ids else component_colors[comp_idx % 10]
            for cid in comp:
                x, y = pos[cid]
                marker = "x" if cid in self._byzantine_ids else "o"
                ax.scatter(x, y, s=80, c=[color], marker=marker,
                           zorder=2, edgecolors="white", linewidths=0.8)
                ax.text(x, y + 0.025, str(cid), fontsize=5,
                        ha="center", va="bottom", zorder=3)

        num_comp = len(components)
        largest_frac = len(max(components, key=len)) / self._n if components else 0.0
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(
            f"Encounter Round {round_num} | components={num_comp} | "
            f"largest={largest_frac:.0%} | drop={self._drop_prob:.2f}"
        )
        fig.savefig(snap_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_encounter_simulator(
    config: Config,
) -> tuple[EncounterGossipSimulator, MetricsTracker]:
    """Build an EncounterGossipSimulator from a Config.

    Mirrors build_gossip_simulator but sets up encounter-mode defaults.
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

    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)

    if config.experiment.device != "cpu" and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        client_devices = []
        for i in range(num_gpus):
            try:
                d = torch.device(f"cuda:{i}")
                torch.zeros(1, device=d)
                client_devices.append(d)
            except Exception:
                pass
        if not client_devices:
            client_devices = [torch.device("cpu")]
    else:
        client_devices = [torch.device("cpu")]

    num_devices = len(client_devices)

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

    byz_ids = select_byzantine_clients(
        config.data.num_clients,
        config.attack.fraction if config.attack.enabled else 0.0,
        seed=config.experiment.seed,
    )
    attack_cfg = config.attack if config.attack.enabled else None

    init_model = model_for_dataset(config.data.dataset)
    init_weights = copy.deepcopy(init_model.state_dict())

    clients = []
    rng = np.random.default_rng(config.experiment.seed)
    for cid, (train_loader, val_loader) in enumerate(loaders):
        device = client_devices[cid % num_devices]
        model = model_for_dataset(config.data.dataset).to(device)
        model.load_state_dict(copy.deepcopy(init_weights))
        client_attack = attack_cfg if cid in byz_ids else None
        clients.append(RobotClient(
            client_id=cid,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.client,
            attack_config=client_attack,
            device=device,
            position=(float(rng.random()), float(rng.random())),
            seed=config.experiment.seed + cid,
        ))

    for client in clients:
        client.is_byzantine = client.client_id in byz_ids
        if client.is_byzantine and client.attack_config is None and attack_cfg is not None:
            client.attack_config = attack_cfg
            client._attack_type = attack_cfg.type

    topology = TopologyManager(
        config.data.num_clients, config.topology, seed=config.experiment.seed
    )

    run_name = (
        f"{config.experiment.name}_"
        f"encounter_agg{config.gossip.aggregation}_"
        f"n{config.data.num_clients}_"
        f"drop{int(config.topology.drop_prob * 100)}_"
        f"async{int(config.gossip.async_train)}"
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

    simulator = EncounterGossipSimulator(
        config=config,
        clients=clients,
        topology=topology,
        test_loader=test_loader,
        metrics=metrics,
    )

    print(f"[build_encounter_simulator] devices={[str(d) for d in client_devices]}, "
          f"byz_ids={sorted(byz_ids)}, drop_prob={config.topology.drop_prob}")

    return simulator, metrics
