---
layout: default
title: How to use Fitbit Health Pipeline
nav_order: 2
parent: FedOps Fitbit Health Pipeline
---

# 📝 FedOps  Fitbit Health Pipeline

<br /> 

## Baseline

```
- Baseline
    - client_main.py
    - client_mananger_main.py
    - server_main.py
    - models.py
    - data_preparation.py
    - requirements.txt (for server)
    - conf
        - config.yaml
```

## Step

First，Start by cloning the FedOps（Place it cloning in the local location）

```
git clone https://github.com/gachon-CCLab/FedOps.git \
&& mv FedOps/silo/examples/torch/EMEDY . \
&& rm -rf FedOps
```

1. Create Task: The task name is required (e.g., **emdtest**). Since this instance belongs to a general machine learning or deep learning task, select **AI**. Keep all subsequent options as default, and finally, choose **FedAvg** as the federated aggregation strategy.
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(1).png)
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(2).png)
    
    **Title:** Task name, for example `fitbit`.
    
    **Model Type:** Select **AI** (for general machine learning or deep learning tasks).
    
    **XAI (Only for Image data):** Select **Disabled**.
    
    **Basic Training Parameters** and **FedAVG Parameters** can be left as default.
    
    **Dataset Parameters (Dataset and Model Settings)**
    
    | Parameter | Example | Description |  |
    | --- | --- | --- | --- |
    | **Dataset Name** | `fitbit` | Specifies the dataset used for training. |  |
    | **Model Name** | `models.SleepLSTM` | Specifies the model file (e.g., a custom SleepLSTM model). |  |
    
    After confirming all parameters are correct, click the **CREATE** button at the bottom to generate the task instance. The new task will then appear in the task list.
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(3).png)
    
2. Enter the server managent of the created task.
3. In Server Management, configure Resource Scaling (the default values are CPU: 1 and Memory: 2 Gi, so modify them if necessary).
    
    
    Then, click **Create Scalable Server** to create the server pod. Once created, this dashboard will show pod and PVC status as in the image above.
    
    （ {"replicas":1,"ready_replicas":1,"available_replicas":1} is normal status）
        
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(4).png)
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(5).png)

1. 
    
    To properly load the **Fitbit Sleep and Activity Dataset**, you need to install the **`kagglehub`** library.
    
    As shown in the image, you can do this by running the following command in the **Execute Command** section of your server interface:
    
    ```
    pip install kagglehub
    ```
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(6).png)
    
2. When editing or replacing files inside the Pod:
    
    At the top of the **File Browser**, enter the path `/app/code/` and click **Browse** to confirm the file directory.
    
    In the **File Content** section on the right, type the full file path and click **Load** for each of the following files:
    
    - `/app/code/models.py`
    - `/app/code/data_preparation.py`
    - `/app/code/server_main.py`
    - `/app/code/conf/config.yaml`
    
    Then, paste the new content you’ve prepared locally into the editor on the right and click **Save File** to apply the changes.
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(7).png)
    
3. Click **Set Start Command** to prepare the command for running the FL server.
    
    (Although you can also start the server by clicking **Start FL Server**, it will only run the server without saving logs. Therefore, it is recommended to use **Set Start Command** to review and confirm the command before execution.)
    
    Once the command is ready, click **Execute** to run it.
    
    Then, click **Check Process** to verify that the FL server process is running.
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(8).png)
    
4. Run the clients.
    - Run `client_main.py` and `client_manager_main.py`
    - Then, in the terminal to confirm whether it runs correctly.
    
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(9).png)
    
5. The monitoring page can confirm the global results
    ![image.png](../../img/How-to-use-Fitbit-Health-Pipeline/image(10).png)
