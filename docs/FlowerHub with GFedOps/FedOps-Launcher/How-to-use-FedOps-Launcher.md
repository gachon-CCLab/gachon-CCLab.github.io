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

FedOps Launcher runs as a Docker container. Docker is required to provide an isolated execution environment for the launcher, manage the workspace volume, and run the FL client consistently across different operating systems.

[Docker Download](https://www.docker.com/get-started/)

![image.png](../../../img/FedOps-Launcher/image(1).png)

**For Windows users, we recommend running Docker in a WSL environment because Docker is generally more stable in Unix-based environments such as macOS and Linux.**

### 2. Run the command in a terminal

2.1 Default command

![image.png](../../../img/FedOps-Launcher/image(2).png)

```bash
docker rm -f fedops-launcher 2>/dev/null; docker run -d --name fedops-launcher -p 5600:5600 -e WORKSPACE_DIR=/workspace -v "$HOME/fedops-workspace:/workspace" joseongjin311/fedops-launcher:v1.0.0 && echo "Now Running On: http://127.0.0.1:5600"
```

The examples in this guide are performed in VS Code within the Linux environment provided by Windows WSL.

Copy the command and run it in your terminal.

2.2 Command for GPU usage

**If your environment does not have an NVIDIA GPU, an error will occur.**

```bash
docker rm -f fedops-launcher 2>/dev/null; docker run -d --name fedops-launcher --gpus all -p 5600:5600 -e WORKSPACE_DIR=/workspace -v "$HOME/fedops-workspace:/workspace" joseongjin311/fedops-launcher:v1.0.0 && echo "Now Running On: http://127.0.0.1:5600"
```

### 3. Open your browser and connect to FedOps Launcher

3.1 Open your browser and enter [http://127.0.0.1:5600](http://127.0.0.1:5600) in the address bar.

![image.png](../../../img/FedOps-Launcher/image(3).png)

### 4. Log in

![image.png](../../../img/FedOps-Launcher/image(5).png)

4.1 Use the same ID and password that you use on the FedOps website.

4.2 You can also continue in Guest mode.

**In Guest mode, you cannot load Task information created on the FedOps website.**

### 5. Main functions of FedOps Launcher

![image.png](../../../img/FedOps-Launcher/image(6).png)

5.1 Start the FL client using these buttons.

5.2 Manage the file system.

5.3 View logs and execute commands in the terminal.

5.4 Edit code.

5.5 Use the extra buttons.

Logout, **Save File**, and Run Code.

**Click "Save File" after editing your code.**

### 6. How to start the FL client

6.1 FlowerHub

Go to GFedOps FlowerHub.

![image.png](../../../img/FedOps-Launcher/image(7).png)

6.2 Select the app you want to create.

![image.png](../../../img/FedOps-Launcher/image(8).png)

6.3 Copy the command.

![image.png](../../../img/FedOps-Launcher/image(9).png)

6.4 CreateFL

Paste the command into the input box.

This creates the FL project in your directory.

![image.png](../../../img/FedOps-Launcher/image(10).png)

6.5 Install

This installs the Python dependencies in your environment.

![image.png](../../../img/FedOps-Launcher/image(11).png)

6.6 Server-side task and FL server setup

Open the [FedOps Task Page](https://ccl.gachon.ac.kr/fedops/task).

![image.png](../../../img/FedOps-Launcher/image(12).png)

Sign in and create a Task on the FedOps website.

![image.png](../../../img/FedOps-Launcher/image(13).png)

Modify the task content as needed. First, enter a Task Title.

You can also edit the Training Parameters. In this example, set **Clients Per Round** to `1` because only one client will be used.

![image.png](../../../img/FedOps-Launcher/image(14).png)

After reviewing the task settings, click the **Create** button to complete task creation.

![image.png](../../../img/FedOps-Launcher/image(15).png)

Create a dedicated Federated Learning server for your task by clicking the **Create Scalable Server** button.

![image.png](../../../img/FedOps-Launcher/image(16).png)

Click **Refresh Status** to check the current server status.

![image.png](../../../img/FedOps-Launcher/image(17).png)

{: .highlight }
💡 When the status shows **"FL Server created"**, you can proceed to the next step.

Next, click the **Start FL Server** button to start the server.

![image.png](../../../img/FedOps-Launcher/image(18).png)

{: .highlight }
💡 To check the logs of the running server, enter `/app/data/logs/serverlog.txt` in the File Browser and click the **Load** button.

![image.png](../../../img/FedOps-Launcher/image(19).png)

When the server is fully prepared, proceed to the next step.

6.7 Task

If you are logged in, you can select a Task created on the FedOps website.

![image.png](../../../img/FedOps-Launcher/image(20).png)

**In this example, "ssj" is the Task name. Select the Task created for your own environment.**

Next, select the FL project to which you want to apply the Task.

![image.png](../../../img/FedOps-Launcher/image(21).png)

**If the Task is not running on the FL server, FedOps Launcher cannot connect to the server.**

6.8 Run

After completing all settings, click the **Run** button.

![image.png](../../../img/FedOps-Launcher/image(22).png)

### 7. Stop FedOps Launcher

When you finish using FedOps Launcher, stop and remove the running container with the following command:

```bash
docker rm -f fedops-launcher
```

**Thank you.**

**You have successfully learned how to use the FedOps Launcher.**
