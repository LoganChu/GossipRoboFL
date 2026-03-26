"""
metrics.py — Evaluation tracking, JSON/MLflow logging, and plot generation.

MetricsTracker is injected into both GossipSimulator and FedAvgSimulator.
It accumulates per-round metrics in memory, flushes to disk after every
logged round, and provides plot generation at experiment end.

Tracked metrics per round:
  global_accuracy     — mean test accuracy across all honest clients
  per_client_accuracy — list of per-client test accuracies
  accuracy_std        — std across client accuracies (measures consensus)
  worst_client        — lowest accuracy among honest clients
  global_loss         — mean test loss
  comm_volume_bytes   — cumulative bytes exchanged in this round's gossip
  graph_avg_degree    — average node degree (topology metric)
  graph_spectral_gap  — algebraic connectivity (gossip mixing speed proxy)
  graph_updated       — whether the topology was rebuilt this round
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Accumulates, logs, and visualises simulation metrics.

    Args:
        log_dir: Directory for JSON logs and plots.
        run_name: Unique identifier for this experiment run.
        num_clients: Total client count (for context).
        byzantine_ids: Set of Byzantine client IDs (excluded from honest metrics).
        backend: "json" | "mlflow" | "both".
        mlflow_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        log_dir: str,
        run_name: str,
        num_clients: int,
        byzantine_ids: set[int],
        backend: str = "json",
        mlflow_uri: str = "http://localhost:5000",
        experiment_name: str = "gossiprobofl",
    ) -> None:
        self.log_dir = log_dir
        self.run_name = run_name
        self.num_clients = num_clients
        self.byzantine_ids = byzantine_ids
        self.backend = backend
        self.history: list[dict] = []
        self._cumulative_comm_bytes: int = 0
        self._start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)
        self._json_path = os.path.join(log_dir, f"{run_name}_metrics.json")

        # Optional MLflow
        self._mlflow_run = None
        if backend in ("mlflow", "both"):
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(experiment_name)
                self._mlflow_run = mlflow.start_run(run_name=run_name)
            except Exception as e:
                print(f"[MetricsTracker] MLflow unavailable, falling back to JSON only: {e}")
                self.backend = "json"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_round(self, round_num: int, metrics: dict) -> None:
        """Log per-round metrics.

        Expected keys in metrics:
          global_accuracy, per_client_accuracy (list), global_loss,
          graph_avg_degree, graph_spectral_gap, graph_updated.
          Optional: comm_volume_bytes (added via log_communication_volume).

        Args:
            round_num: 0-indexed round number.
            metrics: Dict of metric name → value.
        """
        entry = {
            "round": round_num,
            "elapsed_sec": time.time() - self._start_time,
            "cumulative_comm_bytes": self._cumulative_comm_bytes,
            **metrics,
        }
        # Compute honest-only stats if per_client_accuracy present
        per_client = metrics.get("per_client_accuracy", [])
        if per_client:
            honest = [
                acc for i, acc in enumerate(per_client)
                if i not in self.byzantine_ids
            ]
            if honest:
                entry["honest_global_accuracy"] = float(np.mean(honest))
                entry["honest_accuracy_std"] = float(np.std(honest))
                entry["honest_worst_accuracy"] = float(np.min(honest))

        self.history.append(entry)
        self._flush_json()

        if self.backend in ("mlflow", "both") and self._mlflow_run:
            try:
                import mlflow
                flat_metrics = {
                    k: float(v)
                    for k, v in entry.items()
                    if isinstance(v, (int, float)) and k != "round"
                }
                mlflow.log_metrics(flat_metrics, step=round_num)
            except Exception:
                pass

    def log_communication_volume(
        self,
        round_num: int,
        num_messages: int,
        model_size_bytes: int,
    ) -> None:
        """Accumulate bytes exchanged in one gossip round.

        One gossip "message" = one full model broadcast to one neighbour.

        Args:
            round_num: Current round (for future per-round logging).
            num_messages: Number of model transmissions this round.
            model_size_bytes: Bytes per model (from model.model_size_bytes()).
        """
        self._cumulative_comm_bytes += num_messages * model_size_bytes

    def log_byzantine_detection(
        self,
        round_num: int,
        detected_ids: set[int],
        true_byzantine_ids: set[int],
    ) -> dict:
        """Compute detection precision/recall/F1.

        "Detected" here means inferred from clipping behaviour (e.g., any
        neighbour whose clipped delta norm == tau is a potential Byzantine).

        Args:
            round_num: Current round.
            detected_ids: Clients flagged as suspicious by the aggregator.
            true_byzantine_ids: Ground-truth Byzantine set.

        Returns:
            {"precision": float, "recall": float, "f1": float}
        """
        if not true_byzantine_ids and not detected_ids:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        tp = len(detected_ids & true_byzantine_ids)
        fp = len(detected_ids - true_byzantine_ids)
        fn = len(true_byzantine_ids - detected_ids)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _flush_json(self) -> None:
        """Write current history to JSON (called after each log_round)."""
        with open(self._json_path, "w") as f:
            json.dump(self.history, f, indent=2, default=_json_default)

    def save(self) -> str:
        """Flush JSON and return the path."""
        self._flush_json()
        return self._json_path

    def close(self) -> None:
        """Flush and close any open logging backends."""
        self._flush_json()
        if self._mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def generate_plots(self, save_dir: Optional[str] = None) -> list[str]:
        """Generate and save all standard experiment plots.

        Args:
            save_dir: Directory for plots. Defaults to self.log_dir.

        Returns:
            List of saved file paths.
        """
        save_dir = save_dir or self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        saved = []

        if not self.history:
            return saved

        # 1. Convergence curve
        fig = plot_convergence_curves(
            {self.run_name: self.history},
            metric="honest_global_accuracy",
        )
        path = os.path.join(save_dir, f"{self.run_name}_convergence.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        # 2. Communication overhead
        fig = plot_communication_overhead(self.history)
        path = os.path.join(save_dir, f"{self.run_name}_comm_overhead.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        return saved


# ---------------------------------------------------------------------------
# Standalone plot functions
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    histories: dict[str, list[dict]],
    metric: str = "honest_global_accuracy",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot test accuracy vs round for multiple experiment runs.

    Args:
        histories: {run_name: [round_dict, ...]} — one entry per experiment.
        metric: Which metric to plot on the Y axis.
        save_path: If provided, save to this path.
        title: Plot title override.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(histories), 1)))  # type: ignore[attr-defined]

    for (run_name, history), color in zip(histories.items(), colors):
        rounds = [e["round"] for e in history if metric in e]
        vals = [e[metric] for e in history if metric in e]
        if not rounds:
            continue
        ax.plot(rounds, vals, label=run_name, color=color, linewidth=2)

        # Shade ±std if accuracy_std is present alongside honest_global_accuracy
        std_key = "honest_accuracy_std" if "honest" in metric else "accuracy_std"
        stds = [e.get(std_key, 0) for e in history if metric in e]
        if any(s > 0 for s in stds):
            vals_arr = np.array(vals)
            stds_arr = np.array(stds)
            ax.fill_between(rounds, vals_arr - stds_arr, vals_arr + stds_arr,
                            alpha=0.15, color=color)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or "Gossip FL Convergence", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_accuracy_vs_byzantine_fraction(
    results: dict[float, dict[str, float]],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Robustness curve: final accuracy vs Byzantine fraction f.

    Args:
        results: {f: {method_name: final_accuracy}} mapping.
        save_path: Optional save path.
        title: Optional title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    fractions = sorted(results.keys())
    methods = list(next(iter(results.values())).keys())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(methods), 1)))  # type: ignore[attr-defined]

    for method, color in zip(methods, colors):
        accs = [results[f].get(method, np.nan) for f in fractions]
        ax.plot([f * 100 for f in fractions], accs, marker="o", label=method,
                color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Byzantine Fraction f (%)", fontsize=12)
    ax.set_ylabel("Final Test Accuracy", fontsize=12)
    ax.set_title(title or "Byzantine Robustness Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_communication_overhead(
    history: list[dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Dual-axis plot: accuracy (left) and cumulative comm bytes (right) vs round.

    Args:
        history: List of round dicts from MetricsTracker.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    rounds = [e["round"] for e in history if "honest_global_accuracy" in e]
    accs = [e["honest_global_accuracy"] for e in history if "honest_global_accuracy" in e]
    comm = [e.get("cumulative_comm_bytes", 0) / 1e6 for e in history if "honest_global_accuracy" in e]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_acc = "#3498db"
    color_comm = "#e74c3c"

    ax1.plot(rounds, accs, color=color_acc, linewidth=2, label="Accuracy")
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Test Accuracy", color=color_acc, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(rounds, comm, color=color_comm, linewidth=2, linestyle="--", label="Comm (MB)")
    ax2.set_ylabel("Cumulative Comm. Volume (MB)", color=color_comm, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_comm)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    fig.suptitle("Accuracy vs Communication Overhead", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_non_iid_impact(
    results: dict[float, list[dict]],
    metric: str = "honest_global_accuracy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Convergence curves faceted by Dirichlet alpha.

    Args:
        results: {alpha: history_list} mapping.
        metric: Metric to plot.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(results), 1)))  # type: ignore[attr-defined]

    for (alpha, history), color in zip(sorted(results.items()), colors):
        rounds = [e["round"] for e in history if metric in e]
        vals = [e[metric] for e in history if metric in e]
        ax.plot(rounds, vals, label=f"α={alpha}", color=color, linewidth=2)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Impact of Non-IID Degree (Dirichlet α)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_topology_gif(
    snapshot_paths: list[str],
    save_path: str,
    fps: int = 2,
) -> None:
    """Create an animated GIF from topology snapshot image files.

    Args:
        snapshot_paths: Ordered list of paths to PNG snapshots.
        save_path: Output .gif path.
        fps: Frames per second.
    """
    try:
        import imageio
    except ImportError:
        print("[metrics] imageio not installed — skipping GIF generation.")
        return

    if not snapshot_paths:
        return

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    frames = [imageio.imread(p) for p in snapshot_paths]
    imageio.mimwrite(save_path, frames, fps=fps, loop=0)
    print(f"[metrics] Topology GIF saved to {save_path}")


def load_history(json_path: str) -> list[dict]:
    """Load a saved metrics JSON file.

    Args:
        json_path: Path to JSON file written by MetricsTracker.

    Returns:
        List of round dicts.
    """
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle non-serialisable types (numpy scalars, etc.)."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
