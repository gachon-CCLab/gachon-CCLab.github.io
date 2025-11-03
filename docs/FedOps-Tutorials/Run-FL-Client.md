---
layout: default
title: Run FL Client
parent: Get started with FedOps
nav_order: 2
---

{: .highlight }
> 💡
> 
> 
> **How to run an FL Client using the MNIST example**
> 
1. **Install the `fedops` library on the client environment.**

    ```bash
    pip install fedops
    ```

1. **Prepare the project directory structure as follows.**
    
    The files required to run the client should be organized like this.
    

    ```
    app/code/
    ├── client_main.py
    ├── client_manager_main.py
    ├── data_preparation.py
    ├── models.py
    └── conf
        └── config.yaml
    ```

`models.py` and `data_preparation.py` must contain **the same code** as the files on the FL server.

Below is the `config.yaml` required to run the client.

```yaml
'conf/config.yaml'
# Common
random_seed: 42

learning_rate: 0.001 # Input model's learning rate

model_type: 'Pytorch' # This value should be maintained
model:
  _target_: models.MNISTClassifier # Input your custom model
  output_size: 10 # Input your model's output size (only classification)

dataset:
    name: 'MNIST' # Input your data name
    validation_split: 0.2 # Ratio of dividing train data by validation

# client
task_id: 'task_id' # Input your Task Name that you register in FedOps Website

wandb:
  use: false # Whether to use wandb
  key: 'your wandb api key' # Input your wandb api key
  account: 'your wandb account' # Input your wandb account
  project: '${dataset.name}_${task_id}'

# server
num_epochs: 1 # number of local epochs
batch_size: 128
num_rounds: 2 # Number of rounds to perform
clients_per_round: 1 # Number of clients participating in the round

server:
  strategy:
    _target_: flwr.server.strategy.FedAvg # aggregation algorithm (defalut: fedavg)
    fraction_fit: 0.00001 # because we want the number of clients to sample on each round to be solely defined by min_fit_clients
    fraction_evaluate: 0.000001 # because we want the number of clients to sample on each round to be solely defined by min_fit_clients
    min_fit_clients: ${clients_per_round} # Minimum number of clients to participate in training
    min_available_clients: ${clients_per_round} # Minimum number of clients to participate in a round
    min_evaluate_clients: ${clients_per_round} # Minimum number of clients to participate in evaluation

```

```yaml
'/conf/config.yaml'
# Common
random_seed: 42

learning_rate: 0.001 # Input model's learning rate

model_type: 'Pytorch' # This value should be maintained
model:
  _target_: models.MNISTClassifier # Input your custom model
  output_size: 10 # Input your model's output size (only classification)

dataset:
    name: 'MNIST' # Input your data name
    validation_split: 0.2 # Ratio of dividing train data by validation

# client
task_id: 'task_id' # Input your Task Name that you register in FedOps Website

wandb:
  use: false # Whether to use wandb
  key: 'your wandb api key' # Input your wandb api key
  account: 'your wandb account' # Input your wandb account
  project: '${dataset.name}_${task_id}'

```

```python
'client_main.py'
import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import data_preparation
import models

from fedops.client import client_utils
from fedops.client.app import FLClientTask
import logging
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # set log format
    handlers_list = [logging.StreamHandler()]

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

    logger = logging.getLogger(__name__)

    # Set random_seed
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    print(OmegaConf.to_yaml(cfg))

    """
    Client data load function
    After setting model method in data_preparation.py, call the model method.
    """
    train_loader, val_loader, test_loader= data_preparation.load_partition(dataset=cfg.dataset.name,
                                                                        validation_split=cfg.dataset.validation_split,
                                                                        batch_size=cfg.batch_size)

    logger.info('data loaded')

    """
    Client local model build function
    Set init local model
    After setting model method in models.py, call the model method.
    """
    # torch model
    model = instantiate(cfg.model)
    model_type = cfg.model_type     # Check tensorflow or torch model
    model_name = type(model).__name__
    train_torch = models.train_torch() # set torch train
    test_torch = models.test_torch() # set torch test

    # Local model directory for saving local models
    task_id = cfg.task_id  # FL task ID
    local_list = client_utils.local_model_directory(task_id)

    # If you have local model, download latest local model
    if local_list:
        logger.info('Latest Local Model download')
        # If you use torch model, you should input model variable in model parameter
        model = client_utils.download_local_model(model_type=model_type, task_id=task_id, listdir=local_list, model=model)

    # Don't change "registration"
    registration = {
        "train_loader" : train_loader,
        "val_loader" : val_loader,
        "test_loader" : test_loader,
        "model" : model,
        "model_name" : model_name,
        "train_torch" : train_torch,
        "test_torch" : test_torch
    }

    fl_client = FLClientTask(cfg, registration)
    fl_client.start()

if __name__ == "__main__":
    main()

```

```python
'client_manager_main.py'

from pydantic.main import BaseModel
import logging
import uvicorn
from fastapi import FastAPI
import asyncio
import json
from datetime import datetime
import requests
import os
import sys
import yaml
import uuid
import socket
from typing import Optional

handlers_list = [logging.StreamHandler()]
if "MONITORING" in os.environ:
    if os.environ["MONITORING"] == '1':
        handlers_list.append(logging.FileHandler('./fedops/client_manager.log'))
    else:
        pass
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)
logger = logging.getLogger(__name__)
app = FastAPI()

# 날짜를 폴더로 설정
global today_str
today = datetime.today()
today_str = today.strftime('%Y-%m-%d')

global inform_SE

def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[i:i + 2] for i in range(0, 12, 2)])

def get_hostname():
    return socket.gethostname()

class FLTask(BaseModel):
    FL_task_ID: Optional[str] = None
    Device_mac: Optional[str] = None
    Device_hostname: Optional[str] = None
    Device_online: Optional[bool] = None
    Device_training: Optional[bool] = None

class manager_status(BaseModel):
    global today_str, inform_SE

    FL_client: str = '0.0.0.0:8003'
    if len(sys.argv) == 1:
        FL_client = 'localhost:8003'
    else:
        FL_client = 'fl-client:8003'
    server_ST: str = 'ccl.gachon.ac.kr:40019'
    server: str = 'ccl.gachon.ac.kr'
    S3_bucket: str = 'fl-gl-model'
    s3_ready: bool = False
    GL_Model_V: int = 0  # model version
    FL_ready: bool = False

    client_online: bool = False  # flower client online
    client_training: bool = False  # flower client learning

    task_id: str = ''
    task_status: FLTask = None

    client_mac: str = get_mac_address()
    client_name: str = get_hostname()

    inform_SE = f'http://{server_ST}/FLSe/'

manager = manager_status()

@app.on_event("startup")
def startup():
    ##### S0 #####

    # get_server_info()

    ##### S1 #####
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.create_task(check_flclient_online())
    loop.create_task(health_check())
    # loop.create_task(register_client())
    loop.create_task(start_training())

# fl server occured error
def fl_server_closed():
    global manager
    try:
        requests.put(inform_SE + 'FLSeClosed/' + manager.task_id, params={'FLSeReady': 'false'})
        logging.info('server status FLSeReady => False')
    except Exception as e:
        logging.error(f'fl_server_closed error: {e}')

@app.get("/trainFin")
def fin_train():
    global manager
    logging.info('fin')
    manager.client_training = False
    manager.FL_ready = False
    fl_server_closed()
    return manager

@app.get("/trainFail")
def fail_train():
    global manager
    logging.info('Fail')
    manager.client_training = False
    manager.FL_ready = False
    fl_server_closed()
    return manager

@app.get('/info')
def get_manager_info():
    return manager

@app.get('/flclient_out')
def flclient_out():
    manager.client_online = False
    manager.client_training = False
    return manager

def async_dec(awaitable_func):
    async def keeping_state():
        while True:
            try:
                # logging.debug(str(awaitable_func.__name__) + '함수 시작')
                # print(awaitable_func.__name__, '함수 시작')
                await awaitable_func()
                # logging.debug(str(awaitable_func.__name__) + '_함수 종료')
            except Exception as e:
                # logging.info('[E]' , awaitable_func.__name__, e)
                logging.error('[E]' + str(awaitable_func.__name__)+ ': ' + str(e))
            await asyncio.sleep(0.5)

    return keeping_state

# send client name to server_status
# @async_dec
# async def register_client():
#     global manager, inform_SE

#     res = requests.put(inform_SE + 'RegisterFLTask',
#                        data=json.dumps({
#                            'FL_task_ID': manager.task_id,
#                            'Device_mac': manager.client_mac,
#                            'Device_hostname': manager.client_name,
#                            'Device_online': manager.client_online,
#                            'Device_training': manager.client_training,
#                        }))

#     if res.status_code == 200:
#         pass
#     else:
#         logging.error('FLSe/RegisterFLTask: server_ST offline')
#         pass

#     await asyncio.sleep(14)
#     return manager

# check Server Status
@async_dec
async def health_check():
    global manager

    health_check_result = {
        "client_training": manager.client_training,
        "client_online": manager.client_online,
        "FL_ready": manager.FL_ready
    }
    json_result = json.dumps(health_check_result)
    logging.info(f'health_check - {json_result}')

    # If Server is Off, Client Local Learning = False
    if not manager.FL_ready:
        manager.client_training = False

    if (not manager.client_training) and manager.client_online:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None, requests.get, (
                    'http://' + manager.server_ST + '/FLSe/info/' + manager.task_id + '/' + get_mac_address()
            )
        )
        if (res.status_code == 200) and (res.json()['Server_Status']['FLSeReady']):
            manager.FL_ready = res.json()['Server_Status']['FLSeReady']
            manager.GL_Model_V = res.json()['Server_Status']['GL_Model_V']

            # Update manager.FL_task_status based on the server's response
            task_status_data = res.json()['Server_Status']['Task_status']
            logging.info(f'task_status_data - {task_status_data}')
            if task_status_data is not None:
                manager.task_status = FLTask(**task_status_data)
            else:
                manager.task_status = None

        elif (res.status_code != 200):
            # manager.FL_client_online = False
            logging.error('FLSe/info: ' + str(res.status_code) + ' FL_server_ST offline')
            # exit(0)
        else:
            pass
    else:
        pass
    await asyncio.sleep(10)
    return manager

# check client status
@async_dec
async def check_flclient_online():
    global manager
    logging.info('Check client online info')
    if not manager.client_training:
        try:
            loop = asyncio.get_event_loop()
            res_on = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_client + '/online'))
            if (res_on.status_code == 200) and (res_on.json()['client_online']):
                manager.client_online = res_on.json()['client_online']
                manager.client_training = res_on.json()['client_start']
                manager.task_id = res_on.json()['task_id']
                logging.info('client_online')

            else:
                logging.info('client offline')
                pass
        except requests.exceptions.ConnectionError:
            logging.info('client offline')
            pass

        res_task = requests.put(inform_SE + 'RegisterFLTask',
                       data=json.dumps({
                           'FL_task_ID': manager.task_id,
                           'Device_mac': manager.client_mac,
                           'Device_hostname': manager.client_name,
                           'Device_online': manager.client_online,
                           'Device_training': manager.client_training,
                       }))

        if res_task.status_code == 200:
            pass
        else:
            logging.error('FLSe/RegisterFLTask: server_ST offline')
            pass

    else:
        pass

    await asyncio.sleep(6)
    return manager

# Helper function to make the POST request
def post_request(url, json_data):
    return requests.post(url, json=json_data)

# make trigger for client fl start
@async_dec
async def start_training():
    global manager
    # logging.info(f'start_training - FL Client Learning: {manager.FL_learning}')
    # logging.info(f'start_training - FL Client Online: {manager.FL_client_online}')
    # logging.info(f'start_training - FL Server Status: {manager.FL_ready}')

    # Check if the FL_task_status is not None
    if manager.task_status:
        if manager.client_online and (not manager.client_training) and manager.FL_ready:
            logging.info('start training')
            loop = asyncio.get_event_loop()
            # Use the helper function with run_in_executor
            res = await loop.run_in_executor(None, post_request,
                                             'http://' + manager.FL_client + '/start', {"server_ip": manager.server, "client_mac": manager.client_mac})

            manager.client_training = True
            logging.info(f'client_start code: {res.status_code}')
            if (res.status_code == 200) and (res.json()['FL_client_start']):
                logging.info('flclient learning')

            elif res.status_code != 200:
                manager.client_online = False
                logging.info('flclient offline')
            else:
                pass
        else:
            # await asyncio.sleep(11)
            pass
    else:
        logging.info("FL_task_status is None")

    await asyncio.sleep(8)
    return manager

if __name__ == "__main__":
    # asyncio.run(training())
    uvicorn.run("client_manager_main:app", host='0.0.0.0', port=8004, reload=True, loop="asyncio")

```

```python
'models.py'
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

# Define MNIST Model
class MNISTClassifier(nn.Module):
    # To properly utilize the config file, the output_size variable must be used in __init__().
    def __init__(self, output_size):
        super(MNISTClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)  # Image size is 28x28, reduced to 14x14 and then to 7x7
        self.fc2 = nn.Linear(1000, output_size)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Set the torch train & test
# torch train
def train_torch():
    def custom_train_torch(model, train_loader, epochs, cfg):
        """
        Train the network on the training set.
        Model must be the return value.
        """
        print("Starting training...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.train()
        for epoch in range(epochs):
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    pbar.update()  # Update the progress bar for each batch

        model.to("cpu")

        return model

    return custom_train_torch

# torch test
def test_torch():

    def custom_test_torch(model, test_loader, cfg):
        """
        Validate the network on the entire test set.
        Loss, accuracy values, and dictionary-type metrics variables are fixed as return values.
        """
        print("Starting evalutation...")

        criterion = nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        correct = 0
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        model.to(device)
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    # Calculate loss
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    pbar.update()  # Update the progress bar for each batch

        accuracy = correct / len(test_loader.dataset)
        average_loss = total_loss / len(test_loader)  # Calculate average loss

        # if you use metrics, you set metrics
        # type is dict
        # for example, Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        # Add F1 score to metrics
        metrics = {"f1_score": f1}
        # metrics=None

        model.to("cpu")  # move model back to CPU
        return average_loss, accuracy, metrics

    return custom_test_torch

```

```python
'data_preparation.py'
import json
import logging
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)

"""
Create your data loader for training/testing local & global model.
Keep the value of the return variable for normal operation.
"""
# Pytorch version

# MNIST
def load_partition(dataset, validation_split, batch_size):
    """
    The variables train_loader, val_loader, and test_loader must be returned fixedly.
    """
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    # MNIST Data Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
    ])

    # Download MNIST Dataset
    full_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)

    # Splitting the full dataset into train, validation, and test sets
    test_split = 0.2
    train_size = int((1 - validation_split - test_split) * len(full_dataset))
    validation_size = int(validation_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

    # DataLoader for training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
    ])

    # Load the test set of MNIST Dataset
    val_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)

    # DataLoader for validation
    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return gl_val_loader

```

---

**Operation flow**

1. Run `client_main.py` and `client_manager_main.py`.
    
    ```bash
    python3 client_main.py
    python3 client_manager_main.py
    ```
    
2. When `client_main.py` starts for the first time, it loads the data required for training, loads the most recent local model (if any), and then waits to connect to the FL server.
    
    ![image.png](../../img/FedOps-Tutorials/FL-Client/image(1).png)
    
3. If the FL server is running, the client receives the server port from the server and attempts to connect.
    
    ![image.png](../../img/FedOps-Tutorials/FL-Client/image(2).png)
    
4. The client receives the initial global model parameters from the server, then performs training and evaluation. The number of local epochs is provided by the server. After the server finishes all rounds, …
    
    ![image.png](../../img/FedOps-Tutorials/FL-Client/image(3).png)