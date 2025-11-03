---
layout: default
title: How to use FedMap
nav_order: 2
parent: FedOps Aggregation method
---
- How to run this method in your environment?
    
    Following is a usecase which utilized this method (FedMAP) with Facebook AI’s hateful memes multimodal dataset.
    
## Baseline Project Structure

```
- FedOps/multimodal/Usecases/Facebook_AI_benchmark
    - dataset/
    - all_in_one_dataset/
    - conf/
        - config.yaml (# includes metadata and configurations)
    - client_main.py (# main client file you need to run in separate terminal)
    - client_manager_main.py (# client manager file, run this too in a terminal)
    - data_preparation.py
    - flclient_patch_task.py
    - models.py
    - requirements.txt (#server side requirements file)
    - server_main.py (# main server side file, dont need to run this)

```

---
    
### 🔁 1. Clone the Repository at Client


```bash
# Clone the repo without downloading all files
git clone --no-checkout https://github.com/gachon-CCLab/FedOps.git
cd FedOps

# Enable sparse checkout
git sparse-checkout init --cone

# Specify the folder you want
git sparse-checkout set multimodal/Usecases/Facebook_AI_benchmark

# Checkout only that folder
git checkout main

cd multimodal/Usecases/Facebook_AI_benchmark

pip install fedops

```

---

### 📝 2. Download the Dataset for Clients

- Manually download dataset from below links, and  unzip & place it within the client side folder structure .
- “**dataset**” folder: https://drive.google.com/file/d/1EGmoFeCt97_Sgrngs6-ryxg6mRS9EXCJ/view?usp=sharing

![image.png](../../img/How-to-use-FedMAP/image(1).png)

{: .highlight }
💡contains partitioned dataset info for each client (**client_0,client_1,client_2,client_3,and client_4**) . From here  you can decide which client data you have to use and then update **client_id**: in **conf/config.yaml**
This will be helpful in retrieving exact client specific data from “all_in_one_dataset” folder .  



- “**all_in_one_dataset**” folder: https://drive.google.com/drive/folders/1AS2QNc2bG18ctV1uHu59I6LXJaxqoKAz?usp=sharing

![image.png](../../img/How-to-use-FedMAP/image(2).png)

{: .highlight }
💡all images live here. and please do update the path for the datasets as instructed below:

- Now update the dataset folders path in data_preparation.py file (set the **relative path from data_preparation.py)**. Find below lines.

{: .highlight }
```python
# ------------ Paths ------------
# Client-side layout (update path here)
BASE_DATASET_DIR = os.path.abspath("/dataset")          # contains client_1/, client_2/, ...
IMG_DIR = os.path.abspath("/all_in_one_dataset/img")    # all images live here
```



- Now Paste below script  in  your client side local IDE (eg: VS Code) to automatically edit **client/ app.py** file( **remember you just have to copy paste whole script at once in terminal**).

```python
APP_FILE="$(python -c 'import fedops.client.app as m; print(m.__file__)')"; \
cp "$APP_FILE" "${APP_FILE}.bak-$(date +%Y%m%d-%H%M%S)"; \
cat > "$APP_FILE" <<'PY'
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
            
            # client_start object
            client_start = client_fl.flower_client_start(self.status.server_IP, client)

            # Guard: avoid calling None/non-callable during/after shutdown
            fn = client_start
            if not callable(fn):
                logging.info("client_start is not callable (likely after shutdown). Skipping FL execution.")
                return

            # FL client start time
            fl_start_time = time.time()

            # Run asynchronously FL client with TypeError safety during teardown
            try:
                await loop.run_in_executor(None, fn)
            except TypeError:
                logging.exception("client_start became non-callable during shutdown. Ignoring and exiting cleanly.")
                return

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
            logging.info('[E][PC0002] learning: %s', e)
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
PY

```

- Now Paste below script  in  your client side local IDE (eg: VS Code) to automatically edit **client/client_fl.py** file( **remember you just have to copy paste whole script at once in terminal**).

```python
# 1) Locate the file and back it up
APP_FILE="$(python -c 'import fedops.client.client_fl as m; print(m.__file__)')"
echo "Target: $APP_FILE"
cp "$APP_FILE" "${APP_FILE}.bak-$(date +%Y%m%d-%H%M%S)"

# 2) Overwrite with your content
cat > "$APP_FILE" <<'PYCODE'
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
            keys = sorted(self.model.state_dict().keys())
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
            return [sd[k].cpu().numpy() for k in sorted(sd.keys())]
        
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

# 3) (Optional) Verify write
nl -ba "$APP_FILE"

# 4) (Optional) Byte-compile to avoid first-run latency
python - <<'PY'
import py_compile, importlib, sys
import fedops.client.client_fl as m
py_compile.compile(m.__file__, cfile=None, dfile=None, doraise=True)
print("Compiled:", m.__file__)
PY

```

**now we are ready with client side files, lets move to server side setup !!!!!!**!

---

### 📝 3. Create a Task on FedOps Platform

- Visit: https://ccl.gachon.ac.kr/fedops/task
- Sign up or log in.
- Create a new task by filling in like below:

![image.png](../../img/How-to-use-FedMAP/image(3).png)

- **Task ID**
- Disable XAI, & Custering HPO
- Other required task metadata as below (choose ModalityAwareAggregation as strategy to activate fedmap. And please do fill its own parameter values as your own need.
- FedMAP parameters and its meaning.
    - **aggregator_lr: 0.001**
        
        Learning rate for the aggregator MLP which runs behind FedMAP that distills teacher → student weights. If weights oscillate, lower it (e.g., 0.00005); if learning is sluggish, raise a bit.
        
    - **entropy_coeff: 0.01**
        
        Regularizes the student attention to stay spread out (higher = flatter weights, lower = peakier). Increase when a few clients dominate too early; decrease to allow sharper selections.
        
    - **n_trials_per_round: 4**
        
        Optuna trials per round to search teacher “alpha” mixture. More trials can improve global eval but cost more server time (each trial triggers an evaluation). Use small values (2–6) for quick runs; 8–16 for thorough tuning.
        
    - **perf_mix_lambda: 0.7**
        
        Mixes **pk** (post-val accuracy) and **ck** (improvement) into a single performance feature: `0.7*pk + 0.3*ck` here.
        
        Raise toward 1.0 to favor absolute strong performers; lower toward 0.0 to reward fast improvers.
        
    - **z_clip: 3.0**
        
        After z-scoring features, clip to ±3 to tame outliers and noisy metrics. If your metrics are clean, you can relax (e.g., 4–5); if you see instability, tighten (2–3).
        
    
- Fill below details as it is. (Val split =0 because we have separate val set, so we don’t need to split val from training data)

![image.png](../../img/How-to-use-FedMAP/image(4).png)

{: .highlight }
💡Now after task creation go to your cloned repo’s ***conf/config.yaml*** and update task id to the same task id you used when creating the task



---

### **3. Server Side Code Management**

- Visit: https://ccl.gachon.ac.kr/fedops/task
- Select your task
- Go to Server Management
- Press Below shown green button to create a Server pod in Fedops K8 Environment
    
    ![image.png](../../img/How-to-use-FedMAP/image(5).png)
    
- Scale the Resources(memory): please do enter Scale Resources button after entering 10Gi as memory.

![image.png](../../img/How-to-use-FedMAP/image(6).png)

- Now check the server status through below picture. keep refreshing the status and wait until you see similar to this.

![image.png](../../img/How-to-use-FedMAP/image(7).png)

- Wait for around 6-7 mins until you see  yellow colored log in Server log section below in same server management tab (shown in picture). Keep refreshing the logs to see real time log.

![pod alive.PNG](../../img/How-to-use-FedMAP/image(8).png)

- **Then, start editing server side files**

{: .highlight }
💡You can check server side files structure by typing file browser path as below and clicking browse button.



![image.png](../../img/How-to-use-FedMAP/image(9).png)

{: .highlight }
💡Paste the file names with path in yellow colored text space → Copy & paste the file content from cloned repo structure (multimodal/Usecases/Facebook_AI_benchmark folder) into code space → Press save file button. Likewise this, do for each and every file path mentioned below.



![config yaml.PNG](../../img/How-to-use-FedMAP/image(10).png)

filenames with path as follows:

{: .highlight }
1. ***/app/code/server_main.py***
2. ***/app/code/models.py***
3. ***/app/code/data_preparation.py*** 
4. ***/usr/local/lib/python3.10/site-packages/fedops/server/app.py***


- How to edit **/usr/local/lib/python3.10/site-packages/fedops/server/app.py**? see below.
Since this file isn’t exist in downloaded repo, paste below code content into this file code.

```python
# server/app.py
#FILE_PATH=/usr/local/lib/python3.10/site-packages/fedops/server/app.py

import logging
from typing import Dict, Optional, Tuple
import flwr as fl
import datetime
import os
import json
import time
import numpy as np
import shutil
from . import server_api
from . import server_utils
from collections import OrderedDict
from hydra.utils import instantiate

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, NDArrays
from ..utils.fedco.best_keeper import BestKeeper

# TF warning log filtering
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class FLServer():
    def __init__(self, cfg, model, model_name, model_type, gl_val_loader=None, x_val=None, y_val=None, test_torch=None):
        self.task_id = os.environ.get('TASK_ID')  # Set FL Task ID

        self.server = server_utils.FLServerStatus()  # Set FLServerStatus class
        self.model_type = model_type
        self.cfg = cfg
        self.strategy = cfg.server.strategy
        
        self.batch_size = int(cfg.batch_size)
        self.local_epochs = int(cfg.num_epochs)
        self.num_rounds = int(cfg.num_rounds)

        self.init_model = model
        self.init_model_name = model_name
        self.next_model = None
        self.next_model_name = None

        # Will be set in fl_server_start for Pytorch
        self._server_keys = None
        
        if self.model_type == "Tensorflow":
            self.x_val = x_val
            self.y_val = y_val  
        elif self.model_type == "Pytorch":
            self.gl_val_loader = gl_val_loader
            self.test_torch = test_torch
        elif self.model_type == "Huggingface":
            pass

        # ====== 클러스터 전략 여부/메트릭 키 결정 (비클러스터는 영향 없음) ======
        try:
            strat_target = str(self.strategy._target_)
        except Exception:
            strat_target = ""
        self.is_cluster = "server.strategy_cluster_optuna.ClusterOptunaFedAvg" in strat_target

        metric_key = "accuracy"
        if self.is_cluster:
            # yaml: server.strategy.objective -> maximize_f1 | maximize_acc | minimize_loss
            try:
                objective = str(getattr(self.strategy, "objective", "")).lower()
            except Exception:
                objective = ""
            if "maximize_f1" in objective:
                metric_key = "val_f1_score"
            elif "minimize_loss" in objective:
                metric_key = "val_loss"
            else:
                metric_key = "accuracy"

        # 클러스터일 때만 BestKeeper 활성화
        self.best_keeper = BestKeeper(save_dir="./gl_best", metric_key=metric_key) if self.is_cluster else None
        # ===============================================================

    def init_gl_model_registration(self, model, gl_model_name) -> None:
        logging.info(f'last_gl_model_v: {self.server.last_gl_model_v}')

        if not model:
            logging.info('init global model making')
            init_model, model_name = self.init_model, self.init_model_name
            print(f'init_gl_model_name: {model_name}')
            self.fl_server_start(init_model, model_name)
            return model_name
        else:
            logging.info('load last global model')
            print(f'last_gl_model_name: {gl_model_name}')
            self.fl_server_start(model, gl_model_name)
            return gl_model_name

    def fl_server_start(self, model, model_name):
        # Load and compile model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation

        model_parameters = None  # Init model_parameters variable
        
        if self.model_type == "Tensorflow":
            model_parameters = model.get_weights()

        elif self.model_type == "Pytorch":
            # Use ONE deterministic order of keys, shared everywhere
            self._server_keys = sorted(model.state_dict().keys())
            model_parameters = [model.state_dict()[k].detach().cpu().numpy() for k in self._server_keys]

            # Debug a few shapes (safe to keep or remove)
            try:
                for k, a in list(zip(self._server_keys, model_parameters))[:10]:
                    print("[SERVER:init] ", k, np.shape(a))
            except Exception:
                pass

        elif self.model_type == "Huggingface":
            json_path = "./parameter_shapes.json"
            model_parameters = server_utils.load_initial_parameters_from_shape(json_path)

        strategy = instantiate(
            self.strategy,
            initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
            evaluate_fn=self.get_eval_fn(model, model_name),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
        
        # Start Flower server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )

        # ===== 학습 종료 후: (클러스터 전략일 때만) 최고 전역모델로 최종 파일 덮어쓰기 =====
        if self.is_cluster and self.best_keeper is not None:
            try:
                best_params = self.best_keeper.load_params()
                if best_params is not None:
                    gl_model_path = f'./{model_name}_gl_model_V{self.server.gl_model_v}'

                    if self.model_type == "Pytorch":
                        import torch
                        best_nds = parameters_to_ndarrays(best_params)
                        keys = self._server_keys or sorted(model.state_dict().keys())
                        assert len(keys) == len(best_nds), f"[BEST] mismatch: {len(keys)} keys vs {len(best_nds)} arrays"
                        state_dict = OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, best_nds))
                        model.load_state_dict(state_dict, strict=True)
                        torch.save(model.state_dict(), gl_model_path + '.pth')
                        logger.info("[BEST] Saved best global model to %s.pth", gl_model_path)

                        # (선택) 중앙 검증 로그
                        try:
                            loss_b, acc_b, met_b = self.test_torch(model, self.gl_val_loader, self.cfg)
                            logger.info(f"[FINAL-BEST] loss={loss_b:.4f}, acc={acc_b:.6f}, extra={met_b}")
                        except Exception:
                            pass

                    elif self.model_type == "Tensorflow":
                        best_nds = parameters_to_ndarrays(best_params)
                        model.set_weights(best_nds)
                        model.save(gl_model_path + '.h5')
                        logger.info("[BEST] Saved best global model to %s.h5", gl_model_path)

                        # (선택) 중앙 검증 로그
                        try:
                            loss_b, acc_b = model.evaluate(self.x_val, self.y_val, verbose=0)
                            logger.info(f"[FINAL-BEST] loss={loss_b:.4f}, acc={acc_b:.6f}")
                        except Exception:
                            pass

                    elif self.model_type == "Huggingface":
                        logger.info("[BEST] (HF) finalization skipped")
            except Exception as e:
                logger.error(f"[BEST] finalization error: {e}")

    def get_eval_fn(self, model, model_name):
        """Return an evaluation function for server-side evaluation."""

        def evaluate(
            server_round: int,
            parameters_ndarrays: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # model path for saving global model snapshot by round
            gl_model_path = f'./{model_name}_gl_model_V{self.server.gl_model_v}'
            metrics = None
            
            if self.model_type == "Tensorflow":
                model.set_weights(parameters_ndarrays)
                loss, accuracy = model.evaluate(self.x_val, self.y_val, verbose=0)
                model.save(gl_model_path + '.h5')
            
            elif self.model_type == "Pytorch":
                import torch
                # Use the exact same deterministic order as initialization
                keys = self._server_keys or sorted(model.state_dict().keys())
                assert len(keys) == len(parameters_ndarrays), \
                    f"server mismatch: {len(keys)} keys vs {len(parameters_ndarrays)} arrays"

                # Optional: quick debug of first few shapes
                try:
                    for k, a in list(zip(keys, parameters_ndarrays))[:10]:
                        print("[SERVER:evaluate] ", k, np.shape(a))
                except Exception:
                    pass

                state_dict = OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, parameters_ndarrays))
                model.load_state_dict(state_dict, strict=True)
            
                loss, accuracy, metrics = self.test_torch(model, self.gl_val_loader, self.cfg)
                torch.save(model.state_dict(), gl_model_path + '.pth')

            elif self.model_type == "Huggingface":
                logging.warning("Skipping evaluation for Huggingface model")
                loss, accuracy = 0.0, 0.0
                os.makedirs(gl_model_path, exist_ok=True)
                np.savez(os.path.join(gl_model_path, "adapter_parameters.npz"), *parameters_ndarrays)

            # === 라운드별 로그/리포팅 (원래 로직 유지) ===
            if self.server.round >= 1:
                self.server.end_by_round = time.time() - self.server.start_by_round
                if metrics is not None:
                    server_eval_result = {
                        "fl_task_id": self.task_id,
                        "round": self.server.round,
                        "gl_loss": loss,
                        "gl_accuracy": accuracy,
                        "run_time_by_round": self.server.end_by_round,
                        **metrics,
                        "gl_model_v": self.server.gl_model_v,
                    }
                else:
                    server_eval_result = {
                        "fl_task_id": self.task_id,
                        "round": self.server.round,
                        "gl_loss": loss,
                        "gl_accuracy": accuracy,
                        "run_time_by_round": self.server.end_by_round,
                        "gl_model_v": self.server.gl_model_v,
                    }
                json_server_eval = json.dumps(server_eval_result)
                logging.info(f'server_eval_result - {json_server_eval}')
                server_api.ServerAPI(self.task_id).put_gl_model_evaluation(json_server_eval)
            
            # === (클러스터 전략일 때만) BestKeeper 갱신 ===
            if self.is_cluster and self.best_keeper is not None:
                merged_metrics = {"accuracy": accuracy}
                if metrics is not None:
                    merged_metrics.update(metrics)
                try:
                    self.best_keeper.update(
                        server_round=server_round,
                        parameters=ndarrays_to_parameters(parameters_ndarrays),
                        metrics=merged_metrics,
                    )
                except Exception as e:
                    logger.warning(f"[BEST] update skipped: {e}")

            if metrics is not None:
                return loss, {"accuracy": accuracy, **metrics}
            else:
                return loss, {"accuracy": accuracy}

        return evaluate
    

    def fit_config(self, rnd: int):
        """Return training configuration dict for each round."""
        fl_config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "num_rounds": self.num_rounds,
        }

        # For PyTorch, include the exact server key order so clients can zip arrays correctly
        if self.model_type == "Pytorch":
            if self._server_keys is None and hasattr(self, "init_model") and self.init_model is not None:
                self._server_keys = sorted(self.init_model.state_dict().keys())
            try:
                fl_config["param_keys_json"] = json.dumps(self._server_keys)
            except Exception:
                pass

        # increase round
        self.server.round += 1

        # fit aggregation start time
        self.server.start_by_round = time.time()
        logging.info('server start by round')

        return fl_config

    def evaluate_config(self, rnd: int):
        """Return evaluation configuration dict for each round."""
        return {"batch_size": self.batch_size}

    def start(self):
        today_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

        # Loaded last global model or no global model in s3
        self.next_model, self.next_model_name, self.server.last_gl_model_v = server_utils.model_download_s3(
            self.task_id, self.model_type, self.init_model
        )
        
        # New Global Model Version
        self.server.gl_model_v = self.server.last_gl_model_v + 1

        # API that sends server status to server manager
        inform_Payload = {
            "S3_bucket": "fl-gl-model",
            "Last_GL_Model": "gl_model_%s_V.h5" % self.server.last_gl_model_v,
            "FLServer_start": today_time,
            "FLSeReady": True,
            "GL_Model_V": self.server.gl_model_v,
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)

        try:
            fl_start_time = time.time()

            # Run fl server
            gl_model_name = self.init_gl_model_registration(self.next_model, self.next_model_name)

            fl_end_time = time.time() - fl_start_time  # FL end time

            server_all_time_result = {"fl_task_id": self.task_id, "server_operation_time": fl_end_time,
                                        "gl_model_v": self.server.gl_model_v}
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            
            # Send server time result to performance pod
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
            
            # upload global model (최종 파일은 비클러스터는 원래 파일, 클러스터는 BEST로 덮어쓴 파일)
            if self.model_type == "Tensorflow":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.h5"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type == "Pytorch":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type == "Huggingface":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}"
                npz_file_path = f"{global_model_file_name}.npz"
                model_dir = f"{global_model_file_name}"
                real_npz_path = os.path.join(model_dir, "adapter_parameters.npz")
                shutil.copy(real_npz_path, npz_file_path)
                server_utils.upload_model_to_bucket(self.task_id, npz_file_path)

            logging.info(f'upload {global_model_file_name} model in s3')

        # server_status error
        except Exception as e:
            logging.error("error: %s", e)
            data_inform = {'FLSeReady': False}
            server_api.ServerAPI(self.task_id).put_server_status(json.dumps(data_inform))

        finally:
            logging.info('server close')

            # Modifying the model version in server manager
            server_api.ServerAPI(self.task_id).put_fl_round_fin()
            logging.info('global model version upgrade')
```

---

- Install below dependency as below by pressing execute:

![image.png](../../img/How-to-use-FedMAP/image(11).png)

### 🧠 4. Start the FL Server

**Start the actual server:**

{: .highlight }
💡Click Start FL Server button  to prepare the command to run the FL server. 
you can then see log below says FL server created.



![image.png](../../img/How-to-use-FedMAP/image(12).png)

{: .highlight }
💡You must type **/app/data/logs/serverlog.txt** in File content field and press load button to see real time server side logs, and monitor server side FL global model training process.



{: .highlight }
💡You can stop the server by “**stop FL server**” button  ,if you want to stop the server in middle.



---

### 🧠 5. Run Client & Manager Scripts

**Start the actual client:**

```bash
python client_main.py  
```

**Start the client manager (handles communication with FedOps server):**

```bash
python client_manager_main.py

```
{: .highlight }
> 💡 You can run both scripts simultaneously in separate terminals.
> 

---

---

### 📊 6. Monitor Your Task

Use the following tabs to track progress:

- **Monitoring** – Client training status
- **Global Model** – View updates to the central global model
- **Server Management** – Admin controls and logs related to the central server connection

---

### After Successfully finishing FL rounds, don’t forget to execute below command again in your local client side . (Reason: since we modified client/app.py and client/client_fl.py for this setup. By doing pip install again it will bring those two files into original code )

```python
pip install fedops
```
