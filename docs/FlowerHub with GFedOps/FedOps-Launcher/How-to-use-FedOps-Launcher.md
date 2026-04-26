---
layout: default
title: How to use FedOps Launcher
parent: FedOps Launcher
grand_parent: FlowerHub with GFedOps
nav_order: 2
---
# 📝 FedOps Launcher Guide

<br />

{: .highlight }
This guide provides step-by-step instructions for using FedOps Launcher, a FedOps-based client execution environment.

## **Step-by-Step Guide**

---

### 1. Install Docker 

[Docker Download](https://www.docker.com/get-started/)

![image.png](../../../img/FedOps-Launcher/image(1).png)

**Windows users should run Docker in a WSL environment,**

**because Docker is generally more stable in Unix-based environments such as macOS and Linux.**

### 2. Run the command in a CLI environment

2.1 Default command

![image.png](../../../img/FedOps-Launcher/image(2).png)

```bash
docker rm -f fedops-launcher 2>/dev/null; docker run -d --name fedops-launcher -p 5600:5600 -e WORKSPACE_DIR=/workspace -v "$HOME/fedops-workspace:/workspace" joseongjin311/fedops-launcher:v1.0.0 && echo "Now Running On: http://127.0.0.1:5600"
```

In my case, I use VSCode in a Windows WSL environment.

Just copy the command and run it in your CLI environment.

2.2 Command for using GPU

**If your environment does not have an NVIDIA GPU, an error will occur.**

```bash
docker rm -f fedops-launcher 2>/dev/null; docker run -d --name fedops-launcher --gpus all -p 5600:5600 -e WORKSPACE_DIR=/workspace -v "$HOME/fedops-workspace:/workspace" joseongjin311/fedops-launcher:v1.0.0 && echo "Now Running On: http://127.0.0.1:5600"
```

### 3. Open your browser and connect to FedOps Launcher

3.1 Open your browser and enter [http://127.0.0.1:5600](http://127.0.0.1:5600) in URL bar

![image.png](../../../img/FedOps-Launcher/image(3).png)

### 4. Log in

![image.png](../../../img/FedOps-Launcher/image(5).png)

4.1 Use the same ID and password that you use on FedOps website

4.2 You can also continue in Guest mode

**In Guest mode, you cannot load the Task info what you created on the FedOps website.**

### 5. Main functions of FedOps Launcher

![image.png](../../../img/FedOps-Launcher/image(6).png)

5.1 Start the FL client with these buttons

5.2 Manage file system

5.3 View logs and execute commands in terminal

5.4 Edit code

5.5 Extra buttons

Logout, **Save file**, Run Code.

**You must "Save File" after editing your code.**

### 6. How to start the FL client

6.1 FlowerHub

Go to Gfedops FlowerHub

![image.png](../../../img/FedOps-Launcher/image(7).png)

6.2 Select app you want to create

![image.png](../../../img/FedOps-Launcher/image(8).png)

6.3 Copy the command

![image.png](../../../img/FedOps-Launcher/image(9).png)

6.4 CreateFL

Paste the command into the input box

This will create the FL project in your directory.

![image.png](../../../img/FedOps-Launcher/image(10).png)

6.5 Install

This installs the Python dependencies in your environment.

![image.png](../../../img/FedOps-Launcher/image(11).png)

6.6 Task

If you are logged in, you can choose a Task what you created on FedOps website.

![image.png](../../../img/FedOps-Launcher/image(12).png)

**"ssj" is my Task name, so you should select your own Task**

Next, select the FL project to which you want to apply the Task.

![image.png](../../../img/FedOps-Launcher/image(13).png)

**The selected Task must already be running on the FedOps website.**

**If the Task is not running on the FL server, FedOps Launcher cannot connect to the server.**

6.7 Run

All settings are complete. Just click the **Run** button.

![image.png](../../../img/FedOps-Launcher/image(14).png)

**Thank you.**

**You have successfully started the FedOps Launcher.**
