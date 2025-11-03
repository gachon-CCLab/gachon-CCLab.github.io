---
layout: default
title: FedMAP Overview
nav_order: 1
parent: FedOps Aggregation method
---

## 🎯 FedMAP Overview

Federated Averaging (FedAvg) is the de facto baseline in FL, but it **fails under non-IID, imbalanced, and multimodal data distributions**. Even lately introduced FedProx was utilizing Fedavg aggregation on the server side, while using mu parameter to mitigate client drift.

**FedMAP** introduces a *metadata-driven, modality-aware, evaluation-guided* aggregation strategy that dynamically adapts client contributions.

This release provides a **drop-in replacement for FedAvg** that remains compatible with **Flower + FedOps**, while giving researchers and practitioners **clear extension points** for reuse and modification.

---

## 🚀 Key Features

- **Rich Client Metadata**: Clients report validation loss, gradient norm, imbalance ratio, modality usage, performance meta data such as post accuracy, improvement over the last round and diversity in addition to weights.
- **Teacher–Student Aggregation**:
    - *Teacher* uses Optuna to optimize α-weights across 7 client signals.
    - *Student* MLP aggregator distills teacher attention across FL rounds.
- **Evaluation-Guided**: Server-side F1-macro evaluation guides aggregation.
- **Fairness & Stability**: FedProx regularization, grad norm penalties, and balance/diversity scores ensure robustness under heterogeneity.
- **Multimodal Readiness**: Designed for BERT (text) + ResNet (image) fusion models.(this can be replaced with any usecases and its models(model.py) and data(data_preparation.py))

---

## 📊 Why FedMAP over FedAvg

| Aspect | FedAvg | FedMAP |
| --- | --- | --- |
| **Aggregation** | Simple averaging | Metadata-driven, modality-aware, adaptive |
| **Signals** | Only model weights, dataset size | + Val loss, post-accuracy, improvement over last round, grad norm, imbalance, diversity, modalities info |
| **Fairness** | Biased to large clients | Balances minority & diverse clients |
| **Evaluation** | Local only | Global F1-macro evaluation |
| **Robustness** | Sensitive to skew | FedProx + grad control |

---

## 📂 Availability

- Fully **Hydra-configurable** (all params in `conf/config.yaml`).
- Compatible with **FedOps + Flower** (drop-in for FedAvg).
- Aggregator weights saved to `aggregator_mlp.pth` for continuity over each FL round
