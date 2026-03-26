"""
topology.py — Dynamic communication graph for the GossipRoboFL robot swarm.

Physical AI framing: robots move on a 2D arena [0,1]². Communication links
exist only between robots within range r (random geometric graph) or the k
nearest neighbours (KNN graph). As robots move, the graph is rebuilt every
update_every rounds, simulating varying proximity.

Key design decisions:
  - Connectivity is always enforced: a disconnected graph prevents global
    gossip convergence. Missing edges are added between closest nodes across
    components.
  - The spectral gap (2nd eigenvalue of the graph Laplacian) is tracked as a
    proxy for gossip mixing speed. Higher spectral gap → faster convergence.
  - Position updates (step()) and graph rebuilds (build_graph()) are separate
    so the simulator can control their frequency independently.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from src.config import TopologyConfig


# ---------------------------------------------------------------------------
# TopologyManager
# ---------------------------------------------------------------------------

class TopologyManager:
    """Manages robot positions and the communication graph.

    Usage in the simulation loop (handled by GossipSimulator):

        topology = TopologyManager(N, config, seed=42)
        for round_t in range(num_rounds):
            graph_updated = topology.update(round_t)
            neighbors_of_i = topology.get_neighbors(i)
    """

    def __init__(
        self,
        num_clients: int,
        config: TopologyConfig,
        seed: int = 42,
    ) -> None:
        self.num_clients = num_clients
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Initialise positions uniformly in [0,1]²
        self.positions: dict[int, tuple[float, float]] = {
            i: (float(self.rng.uniform()), float(self.rng.uniform()))
            for i in range(num_clients)
        }

        # Build initial graph
        self.graph: nx.Graph = self.build_graph()
        self._round_history: list[dict] = []  # for GIF generation

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> nx.Graph:
        """Build a new communication graph from current positions.

        Dispatches to the method selected by config.type, then enforces
        connectivity. Sets self.graph and returns it.
        """
        builders = {
            "random_geometric": self._build_random_geometric,
            "knn": self._build_knn,
            "erdos_renyi": self._build_erdos_renyi,
        }
        if self.config.type not in builders:
            raise ValueError(
                f"Unknown topology type '{self.config.type}'. "
                f"Choose from {list(builders)}"
            )
        g = builders[self.config.type]()
        g = self._ensure_connected(g)
        self.graph = g
        return g

    def _build_random_geometric(self) -> nx.Graph:
        """Random geometric graph: edge iff Euclidean distance ≤ comm_range."""
        pos = self.positions
        g = nx.Graph()
        g.add_nodes_from(range(self.num_clients))
        # Store positions as node attributes
        for i, (x, y) in pos.items():
            g.nodes[i]["pos"] = (x, y)

        r = self.config.comm_range
        nodes = list(pos.keys())
        coords = np.array([pos[i] for i in nodes])

        # Use KD-tree for O(N log N) range query
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r)
        for i_idx, j_idx in pairs:
            g.add_edge(nodes[i_idx], nodes[j_idx])

        return g

    def _build_knn(self) -> nx.Graph:
        """KNN graph: each node connects to its k nearest neighbours."""
        pos = self.positions
        g = nx.Graph()
        g.add_nodes_from(range(self.num_clients))
        for i, (x, y) in pos.items():
            g.nodes[i]["pos"] = (x, y)

        k = min(self.config.k_nearest, self.num_clients - 1)
        nodes = list(pos.keys())
        coords = np.array([pos[i] for i in nodes])

        tree = cKDTree(coords)
        # k+1 because query returns the point itself as first result
        dists, idxs = tree.query(coords, k=k + 1)
        for i_idx, neighbors in enumerate(idxs):
            for j_idx in neighbors[1:]:  # skip self
                g.add_edge(nodes[i_idx], nodes[j_idx])

        return g

    def _build_erdos_renyi(self) -> nx.Graph:
        """Erdős-Rényi random graph (ignores positions, used for ablations)."""
        seed_int = int(self.rng.integers(0, 2**31))
        g = nx.erdos_renyi_graph(self.num_clients, self.config.er_prob, seed=seed_int)
        # Attach positions for visualisation
        for i, (x, y) in self.positions.items():
            g.nodes[i]["pos"] = (x, y)
        return g

    def _ensure_connected(self, graph: nx.Graph) -> nx.Graph:
        """Add minimum edges to make graph connected.

        Algorithm: find connected components, link each non-main component to
        the main component via the shortest cross-component edge.
        """
        if nx.is_connected(graph):
            return graph

        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        main_comp = list(components[0])
        main_coords = np.array([self.positions[n] for n in main_comp])
        main_tree = cKDTree(main_coords)

        for comp in components[1:]:
            comp_nodes = list(comp)
            comp_coords = np.array([self.positions[n] for n in comp_nodes])
            # Find the closest pair between this component and main
            dists, idxs = main_tree.query(comp_coords, k=1)
            best_idx = int(np.argmin(dists))
            u = comp_nodes[best_idx]
            v = main_comp[int(idxs[best_idx])]
            graph.add_edge(u, v)
            # Merge into main component for next iteration
            main_comp.extend(comp_nodes)
            main_coords = np.vstack([main_coords, comp_coords])
            main_tree = cKDTree(main_coords)

        return graph

    # ------------------------------------------------------------------
    # Mobility
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Move all robots by one random-walk step (does NOT rebuild graph).

        Each robot moves by a displacement sampled from
        Uniform([-step_size, step_size])² per axis.
        Boundary conditions: "reflect" or "wrap".
        """
        if not self.config.mobility.enabled:
            return

        s = self.config.mobility.step_size
        boundary = self.config.mobility.boundary

        for i in range(self.num_clients):
            dx = float(self.rng.uniform(-s, s))
            dy = float(self.rng.uniform(-s, s))
            x, y = self.positions[i]
            x += dx
            y += dy

            if boundary == "reflect":
                x = _reflect(x, 0.0, 1.0)
                y = _reflect(y, 0.0, 1.0)
            else:  # wrap
                x = x % 1.0
                y = y % 1.0

            self.positions[i] = (x, y)

    def update(self, round_num: int) -> bool:
        """Step robots and optionally rebuild the graph.

        Args:
            round_num: Current simulation round (0-indexed).

        Returns:
            True if the graph was rebuilt this round, False otherwise.
        """
        self.step()

        rebuild = (
            self.config.update_every > 0
            and round_num % self.config.update_every == 0
        )
        if rebuild:
            self.build_graph()
        return rebuild

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def get_neighbors(self, client_id: int) -> list[int]:
        """Return list of current graph neighbours for a given client."""
        return list(self.graph.neighbors(client_id))

    def get_graph_metrics(self) -> dict:
        """Compute and return graph-level connectivity metrics.

        Returns:
            {
              "is_connected": bool,
              "num_components": int,
              "diameter": int | None,        # None if disconnected
              "avg_degree": float,
              "clustering": float,           # average clustering coefficient
              "spectral_gap": float,         # algebraic connectivity (2nd Laplacian eigenvalue)
            }
        """
        g = self.graph
        is_conn = nx.is_connected(g)
        num_comp = nx.number_connected_components(g)
        diameter = nx.diameter(g) if is_conn else None
        avg_degree = float(np.mean([d for _, d in g.degree()]))
        clustering = float(nx.average_clustering(g))

        # Spectral gap = algebraic connectivity (Fiedler value).
        # Use numpy eigenvalue decomposition directly — avoids LAPACK crashes on
        # Windows that occur with nx.algebraic_connectivity's tracemin_lu method.
        try:
            L = nx.laplacian_matrix(g).toarray().astype(float)
            eigenvalues = np.linalg.eigvalsh(L)
            eigenvalues_sorted = np.sort(eigenvalues)
            # Second-smallest eigenvalue (first is always ~0 for connected graph)
            spec_gap = float(eigenvalues_sorted[1]) if len(eigenvalues_sorted) > 1 else 0.0
        except Exception:
            spec_gap = 0.0

        return {
            "is_connected": is_conn,
            "num_components": num_comp,
            "diameter": diameter,
            "avg_degree": avg_degree,
            "clustering": clustering,
            "spectral_gap": spec_gap,
        }

    def get_positions(self) -> dict[int, tuple[float, float]]:
        """Return a copy of the current position dict."""
        return dict(self.positions)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        round_num: int = 0,
        byzantine_ids: Optional[set[int]] = None,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Draw the current robot positions and communication graph.

        Args:
            round_num: Used in the title.
            byzantine_ids: Plotted in red (honest robots in blue).
            save_path: If provided, save figure to this path.
            ax: Existing Axes to draw on (creates new figure if None).

        Returns:
            Matplotlib Figure.
        """
        byz = byzantine_ids or set()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        pos = self.positions
        g = self.graph

        # Draw edges
        for u, v in g.edges():
            xu, yu = pos[u]
            xv, yv = pos[v]
            ax.plot([xu, xv], [yu, yv], color="#cccccc", linewidth=0.7, zorder=1)

        # Draw nodes
        node_colors = ["#e74c3c" if i in byz else "#3498db" for i in range(self.num_clients)]
        degrees = dict(g.degree())
        for i in range(self.num_clients):
            x, y = pos[i]
            size = 80 + degrees[i] * 20
            ax.scatter(x, y, s=size, c=node_colors[i], zorder=2, edgecolors="white", linewidths=0.8)
            ax.text(x, y + 0.025, str(i), fontsize=5, ha="center", va="bottom", zorder=3, color="black")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        metrics = self.get_graph_metrics()
        ax.set_title(
            f"Round {round_num} | N={self.num_clients} | "
            f"Byz={len(byz)} | deg={metrics['avg_degree']:.1f} | "
            f"gap={metrics['spectral_gap']:.3f}"
        )

        # Legend
        handles = [
            plt.scatter([], [], s=80, c="#3498db", label="Honest"),
            plt.scatter([], [], s=80, c="#e74c3c", label="Byzantine"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight")

        return fig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _reflect(val: float, lo: float, hi: float) -> float:
    """Reflect a value back into [lo, hi] if it overshoots."""
    if val < lo:
        val = lo + (lo - val)
    elif val > hi:
        val = hi - (val - hi)
    # Clamp for large overshoots
    return max(lo, min(hi, val))
