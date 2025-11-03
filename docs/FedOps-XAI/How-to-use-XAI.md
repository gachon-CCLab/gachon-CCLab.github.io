---
layout: default
title: How to use FedOps XAI
nav_order: 2
parent: FedOps XAI
---
This FL XAI explanation describes a **Grad-CAM (XAI) integration for the FedOps MNIST example**, which supports single-channel MNIST input and automatically generates heatmaps during local (client-side) federated evaluation.

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
    - **xai_utils.py**

```

## Step

First，Start by cloning the FedOps（Place it cloning in the local location）

```
git clone https://github.com/gachon-CCLab/FedOps.git \
&& mv FedOps/silo/examples/torch/MNIST-XAI . \
&& rm -rf FedOps

```

---

---
Create Task: The task name is required (e.g., **xaitest**). Since this instance belongs to a general machine learning or deep learning task, select **AI** and ****set **XAI options** in enabled. Keep all subsequent options as default, and finally, choose **FedAvg** as the federated aggregation strategy.

![image.png](../../img/How-to-use-XAI/image(1).png)

**XAI (Only for Image data):** Select **Enabled**.

Keep all other parameters as default, then click the **CREATE** button at the bottom to generate the task instance. The new task will appear in the task list.

![image.png](../../img/How-to-use-XAI/image(2).png)
Enter the server management of the created task.

![image.png](../../img/How-to-use-XAI/image(3).png)
In Server Management, configure Resource Scaling (the default values are CPU: 1 and Memory: 2 Gi, so modify them if necessary).
    
    
    Then, click **Create Scalable Server** to create the server pod. Once created, this dashboard will show pod and PVC status as in the image above.
    
    （ {"replicas":1,"ready_replicas":1,"available_replicas":1} is normal status）
    
    ![image.png](../../img/How-to-use-XAI/image(4).png)
Click **Set Start Command** to prepare the command for running the FL server.
    
    (Although you can also start the server by clicking **Start FL Server**, it will only run the server without saving logs. Therefore, it is recommended to use **Set Start Command** to review and confirm the command before execution.)
    
    Once the command is ready, click **Execute** to run it.
    
    Then, click **Check Process** to verify that the FL server process is running.
    

![image.png](../../img/How-to-use-XAI/image(5).png)
Run the clients.
    - Run `client_main.py` and `client_manager_main.py`
    - Then, in the terminal to confirm whether it runs correctly.
    
    ![image.png](../../img/How-to-use-XAI/image(6).png)
The monitoring page can confirm the global results

![image.png](../../img/How-to-use-XAI/image(7).png)
  
After each client completes local training, if the **XAI** feature is enabled, visualization results will be automatically saved in this directory.
    
    The file naming convention is usually **`gradcam_class_<class_lable>.jpg`** 
    
    You can open this directory (**/outputs/**) locally to view and analyze the model’s interpretability visualization results.
    
    ![image.png](../../img/How-to-use-XAI/image(8).png)

    
    # Error solutions
    
    - if you encounter an issue related to client/app.py do these changes.
        
        
        with fedops 1.1.30.4 version i had to modify client/app.py and client/client_fl.py as below to make it work woth mnist-xai.
        
         just paste the script below in vscode client terminal and it will automatically find the file and modifies client/app.py .
        
        ```jsx
        # Locate app.py dynamically, backup, overwrite with given content, then verify
        APP_PATH="$(python - <<'PY'
        import importlib, inspect, pathlib
        m = importlib.import_module("fedops.client.app")
        p = pathlib.Path(inspect.getsourcefile(m)).resolve()
        print(p)
        PY
        )"
        
        echo "[INFO] app.py -> $APP_PATH"
        
        # Backup first
        cp "$APP_PATH" "${APP_PATH}.bak.$(date +%s)" && echo "[INFO] Backup OK"
        
        # Overwrite with your content
        cat > "$APP_PATH" <<'PYCODE'
        #client/app.py
        import logging, json
        import socket
        import time
        from fastapi import FastAPI, BackgroundTasks
        import asyncio
        import uvicorn
        from datetime import datetime
        
        from . import client_utils
        from . import client_fl
        from . import client_wandb
        from . import client_api
        from ..utils.fedxai.gradcam import MNISTGradCAM
        
        class FLClientTask():
            def __init__(self, cfg, fl_task=None, xai=False):
                self.app = FastAPI()
                self.status = client_utils.FLClientStatus()
                self.cfg = cfg
                self.client_port = 8003
                self.task_id = cfg.task_id
                self.dataset_name = cfg.dataset.name
                self.output_size = cfg.model.output_size
                self.validation_split = cfg.dataset.validation_split
                self.wandb_use = cfg.wandb.use
                self.model_type = cfg.model_type
                self.model = fl_task["model"]
                self.model_name = fl_task["model_name"]
                self.xai = xai
                
                self.status.client_name = socket.gethostname()
                self.status.task_id = self.task_id
                self.status.client_mac = client_utils.get_mac_address()
                
                logging.info(f'init model_type: {self.model_type}')
                
                if self.wandb_use:
                    self.wandb_key = cfg.wandb.key
                    self.wandb_account = cfg.wandb.account
                    self.wandb_project = cfg.wandb.project
                    self.wandb_name = f"{self.status.client_name}-v{self.status.gl_model}({datetime.now()})"
                    
        
                if self.model_type=="Tensorflow":
                    self.x_train = fl_task["x_train"]
                    self.x_test = fl_task["x_test"]
                    self.y_train = fl_task["y_train"]
                    self.y_test = fl_task["y_test"]
        
                elif self.model_type == "Pytorch":
                    self.train_loader = fl_task["train_loader"]
                    self.val_loader = fl_task["val_loader"]
                    self.test_loader = fl_task["test_loader"]
                    self.train_torch = fl_task["train_torch"]
                    self.test_torch = fl_task["test_torch"]
        
                elif self.model_type == "Huggingface":
                    self.trainset = fl_task["trainset"]
                    self.tokenizer = fl_task["tokenizer"]
                    self.finetune_llm = fl_task["finetune_llm"]
                    self.data_collator = fl_task["data_collator"]
                    self.formatting_prompts_func = fl_task["formatting_prompts_func"]
                            
        
            async def fl_client_start(self):
                logging.info('FL learning ready')
        
                logging.info(f'fl_task_id: {self.task_id}')
                logging.info(f'dataset: {self.dataset_name}')
                logging.info(f'output_size: {self.output_size}')
                logging.info(f'validation_split: {self.validation_split}')
                logging.info(f'model_type: {self.model_type}')
        
                """
                Before running this code,
                set wandb api and account in the config.yaml
                """
                if self.wandb_use:
                    logging.info(f'wandb_key: {self.wandb_key}')
                    logging.info(f'wandb_account: {self.wandb_account}')
                    # Set the name in the wandb project
                    # Login and init wandb project
                    wandb_run = client_wandb.start_wandb(self.wandb_key, self.wandb_project, self.wandb_name)
                else:
                    wandb_run=None
                    self.wandb_name=''
                
                # Initialize wandb_config, client object
                wandb_config = {}
                # client = None
                
                try:
                    loop = asyncio.get_event_loop()
                    if self.model_type == "Tensorflow":
                        client = client_fl.FLClient(model=self.model, x_train=self.x_train, y_train=self.y_train, x_test=self.x_test,
                                                    y_test=self.y_test,
                                                    validation_split=self.validation_split, fl_task_id=self.task_id, client_mac=self.status.client_mac, 
                                                    client_name=self.status.client_name,
                                                    fl_round=1, gl_model=self.status.gl_model, wandb_use=self.wandb_use, wandb_name=self.wandb_name,
                                                    wandb_run=wandb_run, model_name=self.model_name, model_type=self.model_type)
        
                    elif self.model_type == "Pytorch":
                        client = client_fl.FLClient(model=self.model, validation_split=self.validation_split, 
                                                    fl_task_id=self.task_id, client_mac=self.status.client_mac, client_name=self.status.client_name,
                                                    fl_round=1, gl_model=self.status.gl_model, wandb_use=self.wandb_use,wandb_name=self.wandb_name,
                                                    wandb_run=wandb_run, model_name=self.model_name, model_type=self.model_type, 
                                                    train_loader=self.train_loader, val_loader=self.val_loader, test_loader=self.test_loader, 
                                                    cfg=self.cfg, train_torch=self.train_torch, test_torch=self.test_torch)
        
                    elif self.model_type == "Huggingface":
                        client = client_fl.FLClient(
                            model=self.model,
                            validation_split=self.validation_split,
                            fl_task_id=self.task_id,
                            client_mac=self.status.client_mac,
                            client_name=self.status.client_name,
                            fl_round=1,
                            gl_model=self.status.gl_model,
                            wandb_use=self.wandb_use,
                            wandb_name=self.wandb_name,
                            wandb_run=wandb_run,
                            model_name=self.model_name,
                            model_type=self.model_type,
                            trainset=self.trainset,
                            tokenizer=self.tokenizer,
                            finetune_llm=self.finetune_llm,
                            formatting_prompts_func=self.formatting_prompts_func,
                            data_collator=self.data_collator,
                        )
                    
                    # Check data fl client data status in the wandb
                    # label_values = [[i, self.y_label_counter[i]] for i in range(self.output_size)]
                    # logging.info(f'label_values: {label_values}')
        
                    # client_start object
                    client_start = client_fl.flower_client_start(self.status.server_IP, client)
        
                    # FL client start time
                    fl_start_time = time.time()
        
                    # Run asynchronously FL client
                    await loop.run_in_executor(None, client_start)
        
                    logging.info('fl learning finished')
        
                    # FL client end time
                    fl_end_time = time.time() - fl_start_time
        
                    # Grad-CAM 설명 생성
                    if self.xai:
                        try:
                            logging.info("Generating Grad-CAM explanations...")
                            gradcam = MNISTGradCAM(model=self.model)  # Replace "layer_name" with the desired layer
                            input_data = self.x_test[:1]  # Use a sample input for visualization
                            cam_output = gradcam.generate(input_data)
                            
                            # 저장 또는 시각화
                            gradcam.save(cam_output, "gradcam_output.png")  # 저장 경로 지정
                            logging.info("Grad-CAM explanation saved as gradcam_output.png")
                        except Exception as e:
                            logging.error(f"Error generating Grad-CAM explanations: {e}")
        
                    # Wandb 로그 추가 (옵션)
                    # if self.wandb_use:
                    #     wandb_run.log({"gradcam_output": wandb.Image("gradcam_output.png")})
                    
                    
                    if self.wandb_use:
                        wandb_config = {"dataset": self.dataset_name, "model_architecture": self.model_name}
                        wandb_run.config.update(wandb_config, allow_val_change=True)
                        
                        # client_wandb.data_status_wandb(wandb_run, label_values)
                        
                        # Wandb log(Client round end time)
                        wandb_run.log({"operation_time": fl_end_time, "gl_model_v": self.status.gl_model},step=self.status.gl_model)
                        # close wandb
                        wandb_run.finish()
                        
                        # Get client system result from wandb and send it to client_performance pod
                        client_wandb.client_system_wandb(self.task_id, self.status.client_mac, self.status.client_name, 
                                                         self.status.gl_model, self.wandb_name, self.wandb_account, self.wandb_project)
        
                    client_all_time_result = {"fl_task_id": self.task_id, "client_mac": self.status.client_mac, "client_name": self.status.client_name,
                                              "operation_time": fl_end_time,"gl_model_v": self.status.gl_model}
                    json_result = json.dumps(client_all_time_result)
                    logging.info(f'client_operation_time - {json_result}')
        
                    # Send client_time_result to client_performance pod
                    client_api.ClientServerAPI(self.task_id).put_client_time_result(json_result)
        
                    # Delete client object
                    del client
        
                    # Complete Client training
                    self.status.client_start = await client_utils.notify_fin()
                    logging.info('FL Client Learning Finish')
        
                except Exception as e:
                    logging.exception('[E][PC0002] learning')
                    self.status.client_fail = True
                    self.status.client_start = await client_utils.notify_fail()
                    raise e
        
            def start(self):
                # Configure routes, endpoints, and other FastAPI-related logic here
                # Example:
                @self.app.get('/online')
                async def get_info():
                    
                    return self.status
        
                # asynchronously start client
                @self.app.post("/start")
                async def client_start_trigger(background_tasks: BackgroundTasks):
        
                    # client_manager address
                    client_res = client_api.ClientMangerAPI().get_info()
        
                    # # # latest global model version
                    last_gl_model_v = client_res.json()['GL_Model_V']
        
                    # # next global model version
                    self.status.gl_model = last_gl_model_v
                    # self.status.next_gl_model = 1
        
                    logging.info('bulid model')
        
                    logging.info('FL start')
                    self.status.client_start = True
        
                    # get the FL server IP
                    self.status.server_IP = client_api.ClientServerAPI(self.task_id).get_port()
                    # self.status.server_IP = "0.0.0.0:8080"
        
                    # start FL Client
                    background_tasks.add_task(self.fl_client_start)
        
                    return self.status
        
                try:
                    # create client api => to connect client manager
                    uvicorn.run(self.app, host='0.0.0.0', port=self.client_port)
        
                except Exception as e:
                    # Handle any exceptions that occur during the execution
                    logging.error(f'An error occurred during execution: {e}')
        
                finally:
                    # FL client out
                    client_api.ClientMangerAPI().get_client_out()
                    logging.info(f'{self.status.client_name};{self.status.client_mac}-client close')
                    if self.xai == True:
                        # close xai
                        GradCAM.close_xai()
        PYCODE
        ```
        
        just paste the script below in vscode client terminal and it will automatically find the file and modifies client/client_fl.py .
        
        ```jsx
        # Locate client_fl.py dynamically, backup, overwrite with given content, then verify
        CLIENT_FL_PATH="$(python - <<'PY'
        import importlib, inspect, pathlib
        m = importlib.import_module("fedops.client.client_fl")
        p = pathlib.Path(inspect.getsourcefile(m)).resolve()
        print(p)
        PY
        )"
        
        echo "[INFO] client_fl.py -> $CLIENT_FL_PATH"
        
        # Backup first
        cp "$CLIENT_FL_PATH" "${CLIENT_FL_PATH}.bak.$(date +%s)" && echo "[INFO] Backup OK"
        
        # Overwrite with your content
        cat > "$CLIENT_FL_PATH" <<'PYCODE'
        #/home/ccl/anaconda3/envs/fedops_fedmm_env/lib/python3.9/site-packages/fedops/client/client_fl.py
        
        from collections import OrderedDict
        import json, logging
        import flwr as fl
        import time
        import os
        from functools import partial
        from . import client_api
        from . import client_utils
        
        # set log format
        handlers_list = [logging.StreamHandler()]
        
        if "MONITORING" in os.environ:
            if os.environ["MONITORING"] == '1':
                handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
            else:
                pass
        
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                            handlers=handlers_list)
        
        logger = logging.getLogger(__name__)
        import warnings
        import torch
        
        # Avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
        warnings.filterwarnings("ignore", category=UserWarning)
        
        class FLClient(fl.client.NumPyClient):
        
            def __init__(self, model, validation_split, fl_task_id, client_mac, client_name, fl_round,gl_model, wandb_use, wandb_name,
                         wandb_run=None, model_name=None, model_type=None, x_train=None, y_train=None, x_test=None, y_test=None, 
                         train_loader=None, val_loader=None, test_loader=None, cfg=None, train_torch=None, test_torch=None,
                         finetune_llm=None, trainset=None, tokenizer=None, data_collator=None, formatting_prompts_func=None, num_rounds=None):
                
                self.cfg = cfg
                self.model_type = model_type
                self.model = model
                self.validation_split = validation_split
                self.fl_task_id = fl_task_id
                self.client_mac = client_mac
                self.client_name = client_name
                self.fl_round = fl_round
                self.gl_model = gl_model
                self.model_name = model_name
                self.wandb_use = wandb_use
                self.wandb_run = wandb_run
                self.wandb_name = wandb_name            
                
                if self.model_type == "Tensorflow": 
                    self.x_train, self.y_train = x_train, y_train
                    self.x_test, self.y_test = x_test, y_test
                
                elif self.model_type == "Pytorch":
                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    self.test_loader = test_loader
                    self.train_torch = train_torch
                    self.test_torch = test_torch
        
                elif self.model_type == "Huggingface":
                    self.trainset = trainset
                    self.tokenizer = tokenizer
                    self.finetune_llm = finetune_llm
                    self.data_collator = data_collator
                    self.formatting_prompts_func = formatting_prompts_func
        
            def set_parameters(self, parameters):
                if self.model_type in ["Tensorflow"]:
                    raise Exception("Not implemented")
                
                elif self.model_type in ["Pytorch"]:
                    # Use ALL keys in deterministic order — must match server
                    keys = list(self.model.state_dict().keys())
                    assert len(keys) == len(parameters), (
                        f"client mismatch: {len(keys)} model keys vs {len(parameters)} arrays from server"
                    )
                    state_dict = OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, parameters))
                    # If your classifier head differs locally, switch strict=True -> strict=False
                    self.model.load_state_dict(state_dict, strict=True)
        
                elif self.model_type in ["Huggingface"]:
                    client_utils.set_parameters_for_llm(self.model, parameters)
        
            def get_parameters(self):
                """Get parameters of the local model."""
                if self.model_type == "Tensorflow":
                    raise Exception("Not implemented (server-side parameter initialization)")
                
                elif self.model_type == "Pytorch":
                    # Return ALL params in sorted(key) order — must match server
                    sd = self.model.state_dict()
                    return [v.detach().cpu().numpy() for _, v in sd.items()]
                
                elif self.model_type == "Huggingface":
                    return client_utils.get_parameters_for_llm(self.model)
        
            def get_properties(self, config):
                """Get properties of client."""
                raise Exception("Not implemented")
        
            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""
        
                print(f"config: {config}")
                # Get hyperparameters for this round
                batch_size: int = config["batch_size"]
                epochs: int = config["local_epochs"]
                num_rounds: int = config["num_rounds"]
        
                if self.wandb_use:
                    # add wandb config
                    self.wandb_run.config.update({"batch_size": batch_size, "epochs": epochs, "num_rounds": num_rounds}, allow_val_change=True)
        
                # start round time
                round_start_time = time.time()
        
                # model path for saving local model
                model_path = f'./local_model/{self.fl_task_id}/{self.model_name}_local_model_V{self.gl_model}'
        
                # Initialize results
                results = {}
                
                # Training Tensorflow
                if self.model_type == "Tensorflow":
                    # Update local model parameters
                    self.model.set_weights(parameters)
                    
                    # Train the model using hyperparameters from config
                    history = self.model.fit(
                        self.x_train,
                        self.y_train,
                        batch_size,
                        epochs,
                        validation_split=self.validation_split,
                    )
        
                    train_loss = history.history["loss"][len(history.history["loss"])-1]
                    train_accuracy = history.history["accuracy"][len(history.history["accuracy"])-1]
                    results = {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "val_loss": history.history["val_loss"][len(history.history["val_loss"])-1],
                        "val_accuracy": history.history["val_accuracy"][len(history.history["val_accuracy"])-1],
                    }
        
                    # Return updated model parameters
                    parameters_prime = self.model.get_weights()
                    num_examples_train = len(self.x_train)
        
                    # save local model
                    self.model.save(model_path+'.h5')
        
                # Training Torch
                elif self.model_type == "Pytorch":
                    # Update local model parameters
                    self.set_parameters(parameters)
                    
                    trained_model = self.train_torch(self.model, self.train_loader, epochs, self.cfg)
                    
                    train_loss, train_accuracy, train_metrics = self.test_torch(trained_model, self.train_loader, self.cfg)
                    val_loss, val_accuracy, val_metrics = self.test_torch(trained_model, self.val_loader, self.cfg)
                    
                    if train_metrics!=None:
                        train_results = {"loss": train_loss,"accuracy": train_accuracy,**train_metrics}
                        val_results = {"loss": val_loss,"accuracy": val_accuracy, **val_metrics}
                    else:
                        train_results = {"loss": train_loss,"accuracy": train_accuracy}
                        val_results = {"loss": val_loss,"accuracy": val_accuracy}
                        
                    # Prefixing keys with 'train_' and 'val_'
                    train_results_prefixed = {"train_" + key: value for key, value in train_results.items()}
                    val_results_prefixed = {"val_" + key: value for key, value in val_results.items()}
        
                    # Return updated model parameters
                    parameters_prime = self.get_parameters()
                    num_examples_train = len(self.train_loader)
                    
                    # Save model weights
                    torch.save(self.model.state_dict(), model_path+'.pth')
        
                elif self.model_type == "Huggingface":
                    train_results_prefixed = {}
                    val_results_prefixed = {}
        
                    # Update local model parameters: LoRA Adapter params
                    self.set_parameters(parameters)
                    trained_model = self.finetune_llm(self.model, self.trainset, self.tokenizer, self.formatting_prompts_func, self.data_collator)
                    parameters_prime = self.get_parameters()
                    num_examples_train = len(self.trainset)
        
                    train_loss = results["train_loss"] if "train_loss" in results else None
                    results = {"train_loss": train_loss}
        
                    model_save_path = model_path
                    self.model.save_pretrained(model_save_path)
                    # 선택적으로 tokenizer도 함께 저장
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(model_save_path)
        
                # send metrics back
                # If not set in TF branch, define empty dicts to avoid NameError
                if 'train_results_prefixed' not in locals():
                    train_results_prefixed = {}
                if 'val_results_prefixed' not in locals():
                    val_results_prefixed = {}
                results_payload = {**train_results_prefixed, **val_results_prefixed}
        
                return parameters_prime, num_examples_train, results_payload
        
        def flower_client_start(server_address, client):
            fl.client.start_numpy_client(server_address=server_address, client=client)
        PYCODE
        ```
      
        
    
- 
    
