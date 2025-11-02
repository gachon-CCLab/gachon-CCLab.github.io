---
layout: default
title: What's new in FedOps 1.2
nav_order: 4
---
## What’s new in 1.2?

🧠 **Federated LLM Fine-Tuning**

 **(Link:** https://gachon-cclab.github.io/docs/FedOps-LLM/LLM-Finetune/**)**

- Adapt LLMs to distributed data **without sharing raw data**—fully privacy-preserving. Powered by **FlowerTune**, FedOps 1.2 automates the **end-to-end pipeline**: task config → distributed **LoRA training** → model aggregation → global checkpointing.
- **Solves**: High GPU/comms overhead, data silos, manual orchestration.
- **Delivers**: Parameter-efficient tuning, minimal setup, domain-specific LLM adaptation at scale.

**✨ Enhanced Automatic Configuration and FL Server Code Generation** 

- FedOps 1.2 auto-generates validated configs + runnable FL server/client code stubs instantly from your task spec. No more manual tuning—strategy, metrics, hooks, and datasets are pre-wired. Deployment just got drastically faster.

🔬 **Advanced Federated Learning Capabilities – Now Fully Turnkey via Simple Config Flags** 

- **Explainable AI (XAI) Built-In**
    
    **(Link:** https://gachon-cclab.github.io/docs/FedOps-XAI/How-to-use-XAI/**)**: 
    
    Enable federated interpretability with Grad-CAM-based XAI to visualize model decisions on local physiological or image data. Clients generate Grad-CAM heatmaps (e.g., MNIST), report aggregated metrics (entropy, similarity), plus per-round explainability hooks for feature importance, drift checks, and exportable reports. Close the **interpretability gap** in privacy-preserving FL.
    
- **Intelligent Client Clustering + Hyperparameter Optimization (HPO)**
    
    **(Link:** https://gachon-cclab.github.io/docs/FedOps-Hyperparameter-Optimize/How-to-use-Clustering/**)**: 
    
    Automatically group clients by data/behavioral signatures, then run **cluster-specific HPO** to unlock optimal performance even under severe **Non-IID conditions**.
    
- **FedMAP Aggregation for Multimodal FL (MMFL)**
    
    **(Link:** https://gachon-cclab.github.io/docs/FedOps-Aggregation-method/How-use-FedMAP/: 
    
    A new multimodal FL aggregation method that dynamically learns **adaptive client weights** from interpretable meta-features—engineered for **real-world multimodal, non-IID clients**.
    

**All features ship with full guided tutorials, end-to-end examples, and production-ready use cases.**

⌚ **Fitbit Wearable Pipeline**

     **(Link: please add Emedy link here,i cannot see emedy in fedops docs)**

- We've integrated a **real-world federated IoT health pipeline** using **Fitbit wearable data**. Enables privacy-preserving sleep-quality prediction and personalized health monitoring without centralizing user data.
- Introduces an open-source lightweight SleepLSTM model with 3-layer LSTM + projection bottleneck for temporal feature learning on multivariate Fitbit signals.

**🖥️ Enhanced FL Server Logs & Monitoring**

    (Link:** https://gachon-cclab.github.io/docs/FedOps-Tutorials**)

- **Problem Solved**: Hard to track system health, performance drift, and errors in production FL runs.
- **What’s New**: Deep observability with real-time insights into metrics, logs, and lifecycle state.
    - **FL Status Tracking**: Monitors **creation → execution → termination** stages, providing clear status indicators to guide next actions.
    - **Live Server Operation Logs**: Stream **server logs directly to the web dashboard**—view errors, training progress, and learning status **as it happens**.

👉 **No more blind runs. Full visibility, instant debugging.**

### Installation Procedure & Requirements

```jsx
**pip install fedops**
```

***This release is a valuable advancement for researchers, engineers, and teams building privacy-first AI on distributed data—delivering production-grade orchestration, LLM adaptation, multimodal robustness, and full observability with minimal-touch deployment.***

**Thanks to our Contributors**

We extend our sincere gratitude to the dedicated team at **Gachon Cognitive Computing Lab** who made this release possible:

- Yong-Gyom Kim
- InSeo Song
- JingYao Shi
- Ri-Ra Kang
- Akeel Ahamed
- MinHyuk
- JinYong
- Advised by Prof. KangYoon Lee

***Explore the new features and documentation on our website*** 

***( link:*** http://210.102.181.208:40007/ ***).***

***Join the Discussion: Connect with the community on our Slack channel***

***(link:*** https://join.slack.com/t/fedopshq/shared_invite/zt-1xvo9pkm8-drLEdtOT1_vNbcXoxGmQ5A***).***