---
layout: default
title: Clustering Tuning Overview
nav_order: 1
parent: FedOps Clustering Tuning
---
# FedOps 1.2 guide

# FedOps Clustering Tuning

{: .highlight }
🖥️ why we have to use this method?

Conventional Federated Learning (FL) demonstrates high efficiency under Independent and Identically Distributed (IID) data settings; however, it faces the following limitations in Non-IID environments:
1. **Single global model aggregation:** Since each local model converges in different directions due to heterogeneous data distributions, directly aggregating them into one global model leads to degraded performance and instability.
2. **Fixed hyperparameters:** Using the same static hyperparameter configuration for all clients fails to capture each client’s local optimal setting.
To address these issues, this guide introduces a method that analyzes the **model update patterns of clients using a density-based approach**, automatically **clusters clients with similar learning behaviors**, and performs **hyperparameter optimization within each cluster** to ensure stable and robust global convergence.


{: .highlight }
🔥 Here is steop of Clustering Tuning 
---

# Fed-CO: Clustered Hyperparameter Optimization Framework for Federated Learning

`Fed-CO` is a framework designed to perform efficient hyperparameter optimization (HPO) in a Federated Learning environment.

The framework operates along three main axes:
1.  **Dynamic Client Clustering**
2.  **Cluster-wise Hyperparameter Optimization (HPO)**
3.  **Global Model Aggregation**

The core idea is to group clients exhibiting similar learning behaviors using **DBSCAN** and assign a separate **Optuna** optimization study to each cluster. This allows for exploring hyperparameters optimized for specific client groups while aggregating the training results from all clients into a single global model.



## 🚀 Overall Workflow

The process consists of 4 main steps.

### 1. Client Clustering

Clustering is performed by the 'Clustering Engine' on the server.

* **Initial Rounds:** This step is skipped during the initial rounds as there is no prior training feedback available. All clients perform local training using hyperparameters proposed from a shared **'Global Study'**. The server accumulates the feedback features required for clustering during this phase.
* **Clustering Step:** Once sufficient feedback is collected, the server begins performing DBSCAN clustering at the start of each round, using information reported by clients in the previous round.

#### Clustering Features (Input Features)

The input features used for clustering are as follows:

### Clustering Feature Vector

```math
\mathbf{x}_i = [\log_{10}(lr_i), \log_{2}(bs_i), \mathcal{l}_i]
```

* $lr_i$: Learning Rate of client $i$
* $bs_i$: Batch Size of client $i$
* $\mathcal{l}_i$: Local Loss of client $i$
* **Preprocessing:** These three variables are log-transformed and then standardized using Z-score normalization before being used as input for DBSCAN.
* **Excluded Hyperparameters:** Parameters such as `weight decay` were excluded from the clustering features. This decision was made because they primarily influence model generalization rather than convergence direction, making them unsuitable as discriminative factors. Instead, they are treated as independently searchable variables within each cluster's Optuna Study.

#### DBSCAN Settings and Noise Handling

* **Parameters:** Default values are set to `eps = 0.2` and `min_samples = 2`. This prevents excessive cluster merging in small-scale client environments and ensures stable core formation with a minimum of two clients per cluster.
* **Noise Handling:** Clients that do not belong to any cluster (i.e., labeled as –1) are identified as 'noise'.
    * These noise clients are immediately **promoted to independent clusters**.
    * A **dedicated Optuna Study is newly created** for each of them to perform personalized hyperparameter optimization.

---

### 2. Hyperparameter Proposal and Local Training

Once cluster assignment is complete, clients proceed with local training.

1.  Clients receive the **global model weights** broadcast from the server.
2.  Simultaneously, clients receive a set of **cluster-optimized hyperparameters** proposed by their corresponding **'Cluster Module'** (via an Optuna `Study.ask()` call).
3.  Clients perform local training using the received global weights and the proposed hyperparameters.

---

### 3. Result Reporting and Global Model Aggregation

After completing local training, each client transmits **two types of information** to *different modules* on the server.

#### a. Model Update
* **Destination:** **Global Aggregation Engine**
* **Content:** The change in weights (model update) and the number of samples obtained from local training.
* **Purpose:** The server collects updates from *all* clients—**regardless of their cluster membership**—and uses the FedAvg algorithm to update a **single global model**.

#### b. Training Feedback
* **Destination:** The client's corresponding **Cluster Module**
* **Content:** The resulting loss value and the hyperparameters used during local training.
* **Purpose:** This feedback is used to update the cluster-specific Optuna Study for subsequent optimization and clustering.

---

### 4. Cluster-wise Optimization Update

Each 'Cluster Module' that receives training feedback from its clients updates the dedicated Optuna Study for that cluster.

1.  **Metrics Collection:** Collects the reported loss values and hyperparameter feedback from the clients within the cluster.
2.  **Optuna Update:** Accumulates and integrates the collected performance feedback into the ongoing Study using the `Study.tell()` function.
3.  **Progressive Refinement:** As this process repeats over successive rounds, each cluster's Optuna Study progressively refines its search distribution toward a hyperparameter space that is increasingly well-suited to the data characteristics of the clients within that cluster.

---

## 🔁 Detailed Operation

### 1. Hyperparameter Optimization (HPO) Cycle

Each 'Cluster Study' follows a clear `ask-tell` loop:

* **Hyperparameter Proposal (Ask):** For each cluster, the corresponding Optuna Study calls `Study.ask()` to propose an optimized set of hyperparameters for its member clients.
* **Local Training (Train):** The client performs local training using the proposed hyperparameters and the global model, returning the resulting local model and the corresponding loss value.
* **Performance Feedback (Tell):** The server reports the returned loss back to the shared Cluster Study using the `Study.tell()` function. As this feedback accumulates, the Study progressively refines its optimization process.

### 2. Dynamic Clustering and Noise Handling

The clustering process operates dynamically in conjunction with the federated learning loop.

* **Periodic Execution:** Clustering is not performed in every round but is executed periodically with a cooldown interval. This design allows sufficient time in the early rounds to reliably accumulate client feature data.
* **Feature Preprocessing:** Before running DBSCAN, the collected feature set undergoes log transformation and standardization.
* **Noise Promotion:** Any client assigned a label of –1 after DBSCAN execution is immediately promoted to an independent cluster and connected to a separate Cluster Study for isolated optimization.
* **New Study Creation:** If a newly formed cluster (including promoted noise clients) does not already have an associated Optuna Study, a new one is automatically created for it.
