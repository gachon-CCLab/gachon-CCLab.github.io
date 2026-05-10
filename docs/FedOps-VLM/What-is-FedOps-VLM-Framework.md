---
layout: default
title: What is FedOps VLM Framework?
nav_order: 1
parent: FedOps VLM
---
# ✅ What is FedOps VLM Framework??
<br />

**FedOps VLM Framework** is an open, cloud-native federated learning (FL) platform purpose-built for **Vision-Language Models (VLMs)**. It bridges the gap between large-scale multimodal AI and privacy-preserving distributed training — enabling researchers and practitioners to fine-tune powerful VLMs across decentralized edge devices without ever centralizing raw data.

Built on top of the **FedOps** ecosystem (PyPI `fedops`), the framework combines:

- **Flower** — for robust FL communication and client–server orchestration
- **PEFT / LoRA / QLoRA** — for parameter-efficient fine-tuning of large VLMs on resource-constrained devices
- **FedMAP (Parameter-Aware Federated Aggregation)** — a novel component-aware weighted aggregation theorem that achieves tighter convergence bounds than flat averaging when LoRA and projection parameters have different learning dynamics.

The result is a full-stack FL solution that works from a **web portal download** all the way to a **Kubernetes-managed cloud FL server** — reducing the barrier to federated VLM research to just a few configuration steps.

---

## Core Concepts

### Federated Learning for VLMs

Traditional VLM fine-tuning requires pooling sensitive data into a central server. FedOps VLM Framework flips this: **the data never leaves the device**. Only lightweight parameter updates are sent to the server for aggregation.

### Plugin-Based Architecture

**Plugin-based** — models, datasets, and aggregation strategies are swappable. Pick your combination from the web portal, download, and train. No boilerplate wiring required.

### Supported Models

| Model | Size | Quantization |
| --- | --- | --- |
| OneVision | 0.5B | QLoRA (4-bit) |
| PhiVA | 3.8B | QLoRA (4-bit) |

> Custom models via `models.py`

### Supported Datasets

| Dataset | Domain | Size |
| --- | --- | --- |
| VQA-RAD | Medical radiology | 1793 train, 451 test |
| VQAv2 | General vision | Large-scale |
| PathVQA | Pathology | ~19.7K train |
| SLAKE | Multi-organ medical | ~4.9K train, 1K test |
| ChartQA | Chart reasoning | ~18K train |
| DocVQA | Document understanding | ~10K train |

> Custom datasets via `data_preparation.py`

### Aggregation Strategies

| Strategy | Description |
| --- | --- |
| FedAvg | Standard weighted average |
| FedMAP | Component-aware QP — separate λ for LoRA vs. projection layers |
