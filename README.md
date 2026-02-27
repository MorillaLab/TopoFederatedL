<div align="center">

# 🔐 TopoFederatedL

### Topology-Enhanced Federated Learning with Guaranteed Data Privacy

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://choosealicense.com/licenses/gpl-3.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TDA](https://img.shields.io/badge/TDA-giotto--tda-8B5CF6)](https://giotto-ai.github.io/gtda-docs/)
[![FL](https://img.shields.io/badge/Framework-Federated_Learning-10B981)](https://github.com/MorillaLab/TopoFederatedL)
[![Status](https://img.shields.io/badge/Status-Active_Research-orange)](https://github.com/MorillaLab/TopoFederatedL)

**TopoFederatedL** introduces a new federated learning methodology where clients share **topological descriptors** instead of raw gradients — preserving data privacy by design while improving aggregation, robustness to adversarial attacks, and personalisation across heterogeneous clients.

[🧭 Overview](#-overview) · [🚀 Quick Start](#-quick-start) · [🔬 Five Research Directions](#-five-research-directions) · [🏗️ Architecture](#️-architecture) · [🔗 Related Work](#-related-work)

</div>

---

## 🔍 Overview

Standard Federated Learning (FedAvg) treats all clients as interchangeable — averaging their gradients regardless of how different their local data distributions are. This breaks down badly in real-world settings: hospitals with different patient populations, edge devices with heterogeneous sensor data, banks in different economic contexts.

**TopoFederatedL** solves this by applying Topological Data Analysis (TDA) at every stage of the FL pipeline:

<p align="center">
  <img src="Topo_FL.png" alt="Topology-enhanced Federated Learning framework" width="820"/>
  <br/>
  <em>TopoFederatedL: clients share topological summaries (persistent homology features) rather than raw gradients,
  enabling privacy-preserving, topology-aware aggregation across heterogeneous data.</em>
</p>

| Problem in standard FL | TopoFederatedL solution |
|---|---|
| Non-IID data → poor FedAvg | Cluster clients by topological similarity before aggregation |
| Gradient sharing leaks private data | Share persistent homology descriptors instead of gradients |
| Adversarial / poisoned updates | Detect anomalies via topological irregularities |
| Catastrophic forgetting in continual FL | Track topological signatures over rounds to preserve structure |
| One-size-fits-all global model | Personalise via shared topological features |

---

## 🔬 Five Research Directions

### 1 — Enhanced Data Representations (non-IID clients)

TDA captures **global and local structures** simultaneously — critical when FL clients have non-IID data. Persistent homology extracts multi-scale topological features from each client's local data, ensuring local models learn **better structural representations** before aggregation rather than overfitting to their local distribution.

### 2 — Topology-Aware Aggregation (beyond FedAvg)

Standard FedAvg naively averages all client models. TopoFederatedL improves this by:
- Computing **Wasserstein distance** between persistent diagrams of client data
- Clustering clients with similar topological features before aggregation
- Weighting aggregation by topological similarity — avoiding catastrophic averaging of fundamentally dissimilar models
- Enabling **personalised federated learning** grounded in structural data similarity

```
Client A (hospital) ──▶ PH diagram_A ──┐
Client B (clinic)   ──▶ PH diagram_B ──┼──▶ Wasserstein clustering ──▶ Topology-weighted FedAvg
Client C (lab)      ──▶ PH diagram_C ──┘
```

### 3 — Robustness to Adversarial Attacks

TDA detects **anomalies in data distributions** by identifying topological irregularities. Clients whose topological signature deviates significantly from the cluster can be flagged and down-weighted before their updates reach the global model.

> **Healthcare example:** TDA identifies outlier hospitals whose data distribution deviates from the expected topology, preventing their updates from degrading global model performance — even when the deviation is due to data poisoning rather than genuine patient population differences.

Tools: persistent entropy, Betti curves, topological complexity measures.

### 4 — Continual Federated Learning (catastrophic forgetting)

In continual FL, topological methods track **structural changes in data representations over round**. Persistent homology signatures of features guide dynamic learning rate adjustment, preserving important topological structures across tasks as new data arrives.

### 5 — Privacy-Preserving Learning via Topological Abstraction

The core privacy guarantee: clients share **topological summaries** that are **data-agnostic** — they encode the shape of the data without revealing specific values.

```
❌ Standard FL:  Client ──▶ share gradients ──▶ (can reconstruct training data)
✅ TopoFederatedL: Client ──▶ share PH descriptors ──▶ (shape only, no raw data)
```

Persistent homology features encode essential structural information about the local distribution while being provably uninvertible to recover individual records.

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────┐
                    │        FL Server            │
                    │  Topology-Aware Aggregation │
                    │  (Wasserstein-weighted avg) │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ Client A │    │ Client B │    │ Client C │
        │          │    │          │    │          │
        │ Local    │    │ Local    │    │ Local    │
        │ Training │    │ Training │    │ Training │
        │    +     │    │    +     │    │    +     │
        │ TDA      │    │ TDA      │    │ TDA      │
        │ Features │    │ Features │    │ Features │
        └────┬─────┘    └────┬─────┘    └────┬─────┘
             │               │               │
             ▼               ▼               ▼
        PH diagram_A   PH diagram_B   PH diagram_C
             │               │               │
             └───────────────┴───────────────┘
                             │
                   Topological Similarity
                   (Wasserstein distance)
                             │
                   Cluster + Weight ──▶ Server Aggregation
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/MorillaLab/TopoFederatedL.git
cd TopoFederatedL
pip install -r requirements.txt
```

### Basic usage

```python
from topofederatedl import TopoFLClient, TopoFLServer

# Each client computes local TDA features and trains locally
client = TopoFLClient(data=local_dataset, n_clients=5)
client.fit_local()
topo_descriptor = client.get_topological_descriptor()  # PH features, not gradients

# Server aggregates using Wasserstein-weighted FedAvg
server = TopoFLServer(n_clients=5)
server.receive(topo_descriptor, client_id=0)
global_model = server.aggregate()  # topology-aware aggregation
```

### Run the notebook

```bash
jupyter notebook  # then open any notebook in the repo
```

---

## 📦 Dependencies

```
giotto-tda>=0.5.0       # Persistent homology, TDA pipeline
gudhi>=3.5.0            # Gudhi TDA library
POT>=0.8.0              # Python Optimal Transport (Wasserstein distance)
torch>=1.12.0           # Local model training
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

---

## 📁 Repository Structure

```
TopoFederatedL/
├── Topo_FL.png         # Framework overview figure
├── requirements.txt    # Python dependencies
├── README.md
└── LICENSE             # GPL-3.0
```

> **Note:** This is an active research repository. Source code modules are being added progressively. Star / watch to follow updates.

---

## 🔗 Related Work

TopoFederatedL draws on and extends other MorillaLab tools:

- **[GeoTop](https://github.com/MorillaLab/GeoTop)** — geometric-topological feature extraction backbone
- **[TaelCore](https://github.com/MorillaLab/Taelcore)** — TDA-enhanced dimensionality reduction (useful for client-side representation learning)
- **[RL-TBoost](https://github.com/MorillaLab/RL-TBoost)** — RL + TDA for clinical prediction (shares the TDA-as-state paradigm)

---

## 🔮 Research Roadmap

- [ ] Wasserstein-weighted FedAvg implementation
- [ ] Client clustering by topological similarity
- [ ] Differential privacy integration via topological abstraction
- [ ] Continual FL benchmark (CIFAR-100 sequential)
- [ ] Healthcare FL demo (multi-site clinical data)
- [ ] Comparison against FedProx, SCAFFOLD, pFedMe
- [ ] Preprint / paper submission

---

## 🎈 Citation

If you use TopoFederatedL in your research, please cite:

```bibtex
@software{morilla_topofederatedl_2025,
  author    = {Morilla, Ian and {MorillaLab}},
  title     = {TopoFederatedL: Topology-Enhanced Federated Learning
               with Guaranteed Data Privacy},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/MorillaLab/TopoFederatedL}
}
```

---

## 🤝 Contributing

We welcome contributions — FL aggregation strategies, new TDA feature types, privacy analysis, new application domains. Please open an issue before submitting a pull request. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## 📜 License

GNU General Public License v3.0 — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/MorillaLab">MorillaLab</a>
  <br/>
  <sub>Federated Learning · Topological Data Analysis · Privacy by Design</sub>
</div>
