---
layout: default
title: FL Server Create
parent: FedOps Tutorials
nav_order: 2
---
<aside>
💡

This guide walks you through creating a task and running the FL Server using the MNIST example.

Differences from FedOps 1.1.2

1. In 1.1.2, the FL server ran as a Kubernetes Job, so it terminated when an FL run finished, making post-run management and log inspection difficult. Code was managed via Git, which made debugging and fixes harder. In 1.2, the server runs as a Pod with dedicated resources for the FL server, making code management, debugging, and log inspection much easier.
2. In 1.1.2, there were no management features for the FL server, so resource scaling and file browsing were not possible. In 1.2, you can scale resources, browse files, and execute commands, making server management easier.
3. When creating a task, options are collected from the user and applied automatically, preventing typos and missing required fields in the initial config.
</aside>

1. To create an FL server, first create a task in the FedOps web UI.
    
    ![image.png](../img/FedOps%20Tutorials/FL%20Server%20Create/image%20(1).png)
    
    1. Click **Task** on the top navigation bar to open the Task management page.
    2. Click the **+** button in the top-right corner to open the create-task page.
    
2. On the create-task page, enter a **title** and click **Create** to make an FL task with default settings.

    
    ![image.png](../img/FedOps%20Tutorials/FL%20Server%20Create/image%20(2).png)
    
3. Click **Create Scalable Server** to create the FL server.

    
    ![image.png](../img/FedOps%20Tutorials/FL%20Server%20Create/image%20(3).png)
    
    1. By default, the Scalable Server is provisioned with **1 CPU** and **2 Gi** of memory. You can change these in **Resource Scaling** and apply them with **Scale Resources**.
    2. Once server creation starts successfully, the **FL Status** will change. (Click **Refresh Status** to update the status.)
    
        
        ![image.png](../img/FedOps%20Tutorials/FL%20Server%20Create/image%20(4).png)
        
    3. You can check the environment setup logs via **Server Logs**. When setup completes, the **FL Status** shows **“FL Server created.”** (Click **Refresh Logs** to update the logs.)
    
        
        ![image.png](../img/FedOps%20Tutorials/FL%20Server%20Create/image%20(5).png)
        
    4. The created server’s file structure is as follows. `models.py` and `data_preparation.py` include default MNIST example code. `conf/config.yaml` is populated with the options you set when creating the task.
        
        app/code/
        ├── client_main.py
        ├── client_manager_main.py
        ├── data_preparation.py
        ├── models.py
        ├── server_main.py
        ├── requirements.txt
        └── conf
            └── config.yaml
        
    5. Click **Start FL Server** to launch the FL Server.
    
    ![image.png](../img/FedOps%20Tutorials/FL%20Server%20Create/image%20(6).png)
    
    1. The FL server’s runtime logs are saved to **/app/data/logs/serverlog.txt**. Use **File Content** to view the server’s logs.
    2. **Stop FL Server** terminates the running FL Server process.