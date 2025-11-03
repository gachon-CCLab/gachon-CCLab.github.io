---
layout: default
title: How to use FedOps LLM FineTune
parent: FedOps LLM
nav_order: 2
---
# 📝 FedOps LLM Guide

<br />

{: .highlight }
This guide provides step-by-step instructions on how to implement FedOps LLM Fine-Tune, a federated learning lifecycle management operations framework.

This use case will work just fine without modifying anything.

## Baseline

```bash
- Baseline
    - generate_paramshape.py
    - client_main.py
    - client_mananger_main.py
    - server_main.py
    - models.py
    - data_preparation.py
    - requirements.py
    - conf
        - config.yaml
```

## **Step-by-Step Guide**

---

### 1. Create a Task in FedOps Web.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(1).png)

### 2. Modify the task content as desired.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(2).png)

**[You can run it without making any changes.]**

**[(Optional) means that you can edit any part you want. It also means that you don't have to edit anything.]**

2.1 Enter a Task Title.

2.2 Select LLM as the Model Type.

2.3 Edit the Training Parameters & FedAvg Parameters. **(Optional)**

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(3).png)

2.4 Enter the name of the dataset registered in Hugging Face and set the split ratio. **(Optional)**

2.5 Enter the LLM registered in Hugging Face. **(Optional)**

2.6 Edit the LLM Fine-tuning Parameters (LoRA Args). **(Optional)**

2.7 You are ready to create a task. Click the Create button to complete task creation.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(4).png)

2.8 Copy the contents to your temporary notepad and close the window.

### 3. Creating a Federated Learning Server.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(5).png)

3.1 Create a dedicated Federated Learning server for your task using the Create Scalable Server button. (This button automatically creates a pod in FedOps's k8s environment.)

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(6).png)

3.2 You can check the current status of the server via Refresh Logs. Click to check the progress.

**It will take about 10 minutes. Once the server is created and all preparations are complete, the message**

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(7).png)

`Requirements installed successfully`, `[FL server Pod keeping alive]` will appear.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(8).png)

In this guide, we'll be working with a single client.

Therefore, we'll edit the code using the File Browser.

3.3 Browse the `/app/code/` path. You'll see the Federated Learning-related files stored on the server.

3.4 After loading the `/app/code/conf/config.yaml` file, edit the client_per_round value.

3.4.1 Change it from 5 to 1. (You can also change it to the number of clients you actually plan to participate.) **(Optional)**

3.5 Click the Save File button to save your changes.

**After completing steps 4 through 4.6.2, proceed to step 3.6.**

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(9).png)

**You can run the server with either 3.6 or 3.6-2 method.**

3.6 Click the Set Start Command button to see the command to run the Federated Learning server for global model aggregation.

3.7 Start the Federated Learning server by running the command.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(10).png)

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(11).png)

3.6-2 Click the Start FL Server button to start the server.

**`<How to check the log of a running federated learning server>`**

Enter `/app/data/logs/serverlog.txt` in the File Browser, File Content path of FedOps Web and click the Load button. You can view the logs up to that point.

### 4. Federated Learning Client Participation

4.1  Clone the Baseline for FedOps LLM Finetune to your desired directory (path).

Use this command:

```bash
git clone https://github.com/gachon-CCLab/FedOps.git \
&& mv FedOps/llm/usecase/finetune . \
&& rm -rf FedOps
```

4.2 Create a Python environment for Federated Learning client participation (local learning).

**If you already have it, skip to 4.4.**

```bash
conda create -n fedopsclient python=3.11.5
```

```bash
conda activate fedopsclient
```

4.3 Install the required pip libraries in the conda Python environment.

```bash
cd finetune
```

```bash
pip install -r requirements.txt
```

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(12).png)

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(13).png)

4.4 Copy the server's config.yaml and replace the contents of the client's config.yaml.

4.5 Extract the parameter shape of LLM for aggregation.

(General AI models have the model itself and the server aggregates it, but in the case of LLM, the parameter size is large, so it does not have the model itself, but only the structure.)

Run the following command:

```bash
python generate_paramshape.py
```

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(14).png)

4.6 Copy the generated parameter json file to the server.

4.6.1 Open the json file and copy all the contents.

4.6.2  On the Task Management (Server Management) page of FedOps Web, enter `/app/code/parameter_shapes.json` in the File Content path in the FileBrowser and click Load.

Then, you'll see "`cat: /app/code/parameter_shapes.json: No such file or directory.`" Delete this, paste the copied JSON content, and click the Save File button.

You can confirm that the `/app/code` path has been added by clicking the Browse button.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(15).png)

4.7 Run the client to participate in federated learning.

4.7.1 Open two terminals.

4.7.1 Run `python client_main.py` on one side and `python client_manager.py` on the other.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(16).png)

When a client participates in federated learning, information such as the `FL_task_ID` and the `MAC address` of the participating client are output as `task_status_data` in the `client_manager_main` log.

Federated Learning begins when the actual number of participating clients meets the number set in `config.yaml.`

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(17).png)

When federated learning is running, you can see that local learning has started in the `client_main` log.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(18).png)

After local training is complete, the learned parameters are sent to the model.

The round ends when all clients have completed training and sent their parameters to the server.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(19).png)

The server aggregates the parameters received from clients to create a global model, and repeats this for the number of rounds specified in `config.yaml`.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(20).png)

Progress for each round can be checked in real time on the Monitoring Page of FedOps Web.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(21).png)

Federated Learning terminates after all rounds are completed. The client_main log also displays the completion of training.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(22).png)

The server outputs the results of all rounds and also indicates that federated learning has ended.

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(23).png)

Once training is complete, you can download the generated global model from the Global Model window.

**If you would like to try inference using the generated global model, try the example code below.**

```bash
import numpy as np
from omegaconf import DictConfig, OmegaConf

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from collections import OrderedDict
from flwr.common.typing import NDArrays
import torch

# .npz load
loaded = np.load("Your own path")

print(loaded.files)  # → ['arr_0', 'arr_1', 'arr_2', ...]

parameters = [loaded[k] for k in loaded.files]

def set_parameters_for_llm(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

def load_model(model_name: str, quantization: int, gradient_checkpointing: bool, peft_config):
        if quantization == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

        if gradient_checkpointing:
            model.config.use_cache = False

        return get_peft_model(model, peft_config)

quantization = 4
gradient_checkpoining = True
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.075,
    task_type="CAUSAL_LM",
)

model = load_model(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization=quantization,
    gradient_checkpointing=gradient_checkpoining,
    peft_config=peft_config,
)

set_parameters_for_llm(model, parameters)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)

input_text = "hello"

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"🧠 Generated:\n{generated_text}")
```

![image.png](../../img/FedOps-LLM/LLM-Finetune/image(24).png)

**Thank you.**

**You have successfully completed the FedOps LLM FineTuning.**
