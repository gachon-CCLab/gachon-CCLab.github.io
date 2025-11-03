---
layout: default
title: FedOps XAI
nav_order: 9
has_children: true
permalink: docs/FedOps-XAI
---

# FedOps XAI


In summary:
    
   Each FedOps client not only trains locally but also generates interpretable Grad-CAM visualizations to verify that the local model is learning meaningful spatial features before contributing updates to the global model.
    
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
