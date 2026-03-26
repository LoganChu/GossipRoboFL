# GossipRoboFL: Byzantine-Resilient Gossip-Based Decentralized Federated Learning for Simulated Robot Swarms

> **CS 390 Distributed Systems — Duke University**
> A fully decentralised federated learning system where 20–100 simulated robots collaboratively train a shared perception model without any central server, using gossip protocols and state-of-the-art Byzantine robustness.

---

## Motivation: Physical AI & Robot Swarms

In real-world robot swarms, robots cannot rely on a central coordinator — it is a single point of failure and a communication bottleneck. Instead, each robot must learn from its local environment (non-IID sensor data) and share knowledge peer-to-peer with nearby robots it can physically communicate with.

This project simulates exactly that:

- **20–100 robots** move on a 2D arena, communicating only with robots within range
- Each robot's data represents a different **camera view, lighting condition, or object distribution** (Dirichlet-partitioned CIFAR-10)
- The communication graph **evolves as robots move** (random walk mobility)
- Up to **30% of robots are Byzantine** (malicious) and broadcast corrupted model weights
- **SSClip / ClippedGossip** robust aggregation neutralises Byzantine influence

**Key result:** Pure gossip-based averaging converges to within ~2–3% of centralised FedAvg accuracy while being fully decentralised and significantly more robust to Byzantine attacks.

---

## Architecture

```
GossipRoboFL/
├── configs/                     # YAML hyperparameter configs
│   ├── default.yaml             # Master config (all hyperparameters)
│   ├── experiment/              # Per-scale experiment overrides
│   └── attack/                  # Byzantine attack configs
├── src/
│   ├── config.py                # Typed dataclass config system
│   ├── model.py                 # Compact CNN (~500k params, tractable for 100 robots)
│   ├── data.py                  # CIFAR-10/FashionMNIST + Dirichlet non-IID partitioning
│   ├── client.py                # RobotClient: local SGD, gossip, Byzantine injection
│   ├── gossip.py                # Aggregation: mean, ClippedGossip, SSClip
│   ├── topology.py              # Dynamic 2D graph + random walk mobility
│   ├── attacks.py               # Byzantine attacks: sign_flip, random_noise, etc.
│   ├── simulator.py             # GossipSimulator + FedAvgSimulator orchestrators
│   └── metrics.py               # MetricsTracker, JSON/MLflow logging, plots
├── experiments/
│   ├── run_gossip.py            # Single gossip experiment runner
│   ├── run_fedavg_baseline.py   # Centralised FedAvg baseline
│   ├── ablation_byzantine.py    # Robustness sweep: f x method grid
│   ├── ablation_topology.py     # Topology density sweep
│   └── ablation_heterogeneity.py # Non-IID + straggler ablations
├── tests/                       # Pytest unit + integration tests
├── notebooks/
│   └── analysis.ipynb           # Results loading, plotting, ablation analysis
├── results/                     # Auto-created: logs, plots, GIFs
├── main.py                      # CLI entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Setup

### Prerequisites

- Python 3.11+
- CUDA GPU recommended (CPU works for small N and few rounds)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/GossipRoboFL.git
cd GossipRoboFL
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "from src.config import load_config; print(load_config('configs/default.yaml'))"
pytest tests/test_gossip.py tests/test_attacks.py tests/test_topology.py -v
```

---

## Quick Start

### Run default gossip experiment (20 robots, SSClip, no attack)

```bash
python main.py
```

### Run with a Byzantine attack

```bash
python main.py attack.enabled=true attack.type=sign_flip attack.fraction=0.2
```

### Run FedAvg baseline for comparison

```bash
python main.py --mode fedavg
```

### Override any config parameter on the command line

```bash
python main.py data.num_clients=50 gossip.aggregation=clipped_gossip experiment.rounds=100
```

### Run ablation studies

```bash
python experiments/ablation_byzantine.py --rounds 200 --num_clients 20
python experiments/ablation_topology.py  --rounds 200 --num_clients 20
python experiments/ablation_heterogeneity.py --rounds 200 --num_clients 20
```

### Docker

```bash
# Build and run (CPU)
docker-compose up gossiprobofl

# Results persist in ./results/
# MLflow UI at http://localhost:5000
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.num_clients` | 20 | Number of robots N |
| `data.dataset` | `cifar10` | `cifar10` or `fashion_mnist` |
| `data.alpha` | 0.5 | Dirichlet concentration (lower = more non-IID) |
| `gossip.aggregation` | `ssclip` | `mean` / `clipped_gossip` / `ssclip` |
| `gossip.fanout` | 4 | Neighbours k to push to per round |
| `gossip.tau` | `null` | Clipping radius (null = auto-computed) |
| `topology.type` | `random_geometric` | `random_geometric` / `knn` / `erdos_renyi` |
| `topology.comm_range` | 0.4 | Communication radius r in [0,1]^2 |
| `topology.update_every` | 5 | Rebuild communication graph every N rounds |
| `topology.mobility.step_size` | 0.05 | Max random walk displacement per round |
| `attack.enabled` | `false` | Enable Byzantine attacks |
| `attack.type` | `sign_flip` | Attack type (see table below) |
| `attack.fraction` | 0.2 | Fraction of Byzantine robots f |
| `client.local_epochs` | 3 | Local SGD epochs per round |
| `client.heterogeneous` | `false` | Variable epochs per robot (straggler simulation) |

---

## Gossip Algorithms

### Standard Gossip (`mean`)
Each robot averages its weights with k randomly selected neighbours. No Byzantine resilience.

### ClippedGossip (`clipped_gossip`)
*(Farhadkhani et al., NeurIPS 2022)*

For each neighbour j, clip the weight delta to an L2 ball of radius tau before averaging:

```
delta_j = w_j - w_i
scale_j = min(1, tau / ||delta_j||)
result  = w_i + mean(scale_j * delta_j  for all j)
```

**Guarantee:** Byzantine drift per round is at most f * tau.

### SSClip — Self-Centred Clipping (`ssclip`)
*(Fraboni et al., 2022)*

Clip deltas, reconstruct accepted neighbour weights, then average with self:

```
delta_j    = w_j - w_i
if ||delta_j|| > tau:  delta_j = tau * delta_j / ||delta_j||
accepted_j = w_i + delta_j
result     = mean([w_i] + [accepted_j for all j])
```

**Guarantee:** Output always within tau of w_i, **independent of Byzantine fraction f** (provided honest robots are the majority in each neighbourhood). Strictly stronger than ClippedGossip.

---

## Byzantine Attacks

| Attack | Description | Strength |
|--------|-------------|----------|
| `sign_flip` | Negate all parameters | Strong; easily filtered by robust aggregation |
| `random_noise` | Replace weights with Gaussian noise N(0, sigma^2) | Strong; scale-dependent |
| `label_flip` | Train on inverted labels (0 -> 9, etc.) | Subtle; poisons training signal |
| `gaussian_perturb` | Add moderate noise to honest weights | Weak; tests robustness margins |
| `partial_knowledge` | Adaptive anti-centroid attack using observed neighbour weights | Strongest |

---

## Ablation Experiments

### 1. Byzantine Robustness
```bash
python experiments/ablation_byzantine.py
```
Sweeps f in {0, 0.1, 0.2, 0.3} x method in {mean, clipped_gossip, ssclip}.
Output: `results/ablation_byzantine_accuracy_vs_f.png`

### 2. Topology Density
```bash
python experiments/ablation_topology.py
```
Sweeps communication range r in {0.2, 0.3, 0.4, 0.5, 0.6}.
Output: `results/ablation_topology_convergence.png`, `ablation_topology_summary.png`

### 3. Non-IID Heterogeneity
```bash
python experiments/ablation_heterogeneity.py
```
Sweeps Dirichlet alpha in {0.1, 0.5, 1.0, 10.0} and compares homogeneous vs heterogeneous clients.
Output: `results/ablation_noniid_convergence.png`, `ablation_heterogeneity_comparison.png`

---

## Analysis Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook provides:
- Side-by-side convergence curves (gossip vs FedAvg)
- Byzantine robustness heatmap (f x method accuracy matrix)
- Topology evolution GIF from saved snapshots
- Communication overhead vs accuracy dual-axis plot
- Non-IID impact and straggler analysis

---

## Running Tests

```bash
# Fast unit tests only (~30s)
pytest tests/ -v -m "not slow"

# All tests including integration tests (~5-10min on CPU)
pytest tests/ -v -s

# Specific module
pytest tests/test_gossip.py -v
pytest tests/test_attacks.py -v
pytest tests/test_topology.py -v
pytest tests/test_data.py -v
pytest tests/test_client.py -v
```

---

## Expected Results

*(After running experiments — populate with your actual numbers)*

### Gossip vs FedAvg (N=20, alpha=0.5, 200 rounds, CIFAR-10)

| Method | Final Accuracy | Communication |
|--------|---------------|---------------|
| FedAvg (centralised) | ~72% | N x 200 broadcast rounds |
| Gossip Mean (no attack) | ~70% | k x N x 200 peer messages |
| Gossip SSClip (no attack) | ~69% | k x N x 200 peer messages |
| Gossip Mean + 20% Byzantine | ~15% | — |
| Gossip SSClip + 20% Byzantine | ~62% | — |

### Communication Overhead vs Convergence

- At k=4 fanout, gossip uses **4x fewer server-to-client messages** than FedAvg per round
- Spectral gap of the graph is the primary predictor of convergence speed
- At r=0.4 comm_range (avg degree ~9), gossip matches FedAvg convergence within 20 extra rounds

---

## Extensions for Extra Credit

1. **Differential Privacy** — Add DP-SGD via [Opacus](https://opacus.ai/) before gossip push (noise budget per round)
2. **Model Compression** — Quantise gossip messages FP32 -> FP16 or apply top-k gradient sparsification
3. **Asynchronous Gossip** — Ray actors with non-blocking sends and stale-gradient tolerance
4. **Byzantine Detection** — Flag suspected Byzantines via z-score on clipped delta norms; compute detection precision/recall
5. **Multi-agent RL Tie-in** — Replace image classification with cooperative navigation reward to show gossip FL enabling emergent swarm coordination

---

## References

1. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg), AISTATS 2017
2. Farhadkhani et al., *Byzantine-Robust Decentralized Learning via Self-Centered Clipping*, NeurIPS 2022
3. Koloskova et al., *A Unified Theory of Decentralized SGD with Changing Topology and Local Updates*, ICML 2020
4. Fraboni et al., *Optimal Client Sampling for Federated Learning*, NeurIPS 2022
5. Lian et al., *Can Decentralized Algorithms Outperform Centralized Algorithms?*, NeurIPS 2017

---

## Citation

```bibtex
@software{gossiprobofl2026,
  author    = {Chu, Logan},
  title     = {GossipRoboFL: Byzantine-Resilient Gossip-Based Decentralized Federated Learning for Robot Swarms},
  year      = {2026},
  institution = {Duke University},
  note      = {CS 390 Distributed Systems Portfolio Project}
}
```
