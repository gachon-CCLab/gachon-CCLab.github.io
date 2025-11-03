---
layout: default
title: How to use Fitbit Health Pipeline
nav_order: 1
parent: FedOps Fitbit Health Pipeline
---
### overview

By applying temporal windowing to **Fitbit Health device data** (including steps, heart rate, calories, and sleep duration) and integrating it with the FedOps federated learning framework, the system enables collaborative training of an **LSTM-based** **sleep quality prediction model** across multiple devices while preserving data privacy, thus achieving automated model operations.

Data & Model:

- It automatically reads CSV files (steps, calories, heart rate, sleep), merges them by time and user, calculates a **StressLevel** feature, and generates **fixed-length time windows** per user.
- Use the "Fitbit Sleep and Activity Dataset" (https://www.kaggle.com/datasets/arashnic/fitbit) containing:
    
    
    heart_rate: Beats per minute (BPM)
    steps: Daily step count
    stress_level: Self-reported (1–10)
    
    sleep_quality: Binary label (0 = poor, 1 = good)
