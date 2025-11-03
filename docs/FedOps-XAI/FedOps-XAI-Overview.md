---
layout: default
title: FedOps XAI Overview
nav_order: 1
parent: FedOps XAI
---

## 🎯 FedOps XAI Overview
    
   Each FedOps client not only trains locally but also generates interpretable Grad-CAM visualizations to verify that the local model is learning meaningful spatial features before contributing updates to the global model.

1. Grad-CAM Generation (XAI Module)
   If `xai.enabled: true` and `xai.run_location: client` are set in the configuration, the client will:
   - Perform **forward** and **backward** passes on local validation samples.
   - Extract gradients and feature maps from the **target explanation layer** (e.g., `conv2`).
   - Compute **Grad-CAM weights** and overlay the resulting heatmap onto the original image.
   - Save the visualized results to the directory: `outputs/gradcam/`.
    
    
2. Local Explainability Evaluation
   These heatmaps help to evaluate:
   - Whether the model is focusing on the **actual stroke regions** of digits (indicating correct learning).
   - Whether there are **biased or noisy attention areas**, which could signal overfitting or abnormal local data.
    
---
- What's new here?
    
    New XAI Feature Description
    
    - **`xai_utils.py`** integrates the **pytorch-grad-cam** library to implement a configurable Grad-CAM visualization feature.
    - Add an `xai` section in **`config.yaml`** to enable or configure Grad-CAM:
    
    ```yaml
    xai:
      enabled: true
      layer: conv3
      save_dir: ./outputs/xai
    
    ```
    
    - During the client evaluation stage (`client_main.py → evaluate()`), the system automatically generates and saves Grad-CAM heatmaps.
    - The generated Grad-CAM images can be viewed locally under `outputs/xai/` for visualization and tracking.
    
    Added: XAI Grad-CAM Module
    
    ```python
    if cfg.xai.enabled:
        logger.info("Running Grad-CAM for interpretability")
    
        sample_batch = next(iter(test_loader))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        input_tensor = sample_batch[0][:1].to(device)  # take 1 test sample
        label = sample_batch[1][0]
    
        from xai_utils import apply_gradcam_configurable
    
        heatmap_img, cam_map = apply_gradcam_configurable(
            model=model,
            input_tensor=input_tensor,
            label=label,
            cfg=cfg
        )
    
    ```
