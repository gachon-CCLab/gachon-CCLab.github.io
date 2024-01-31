---
layout: default
title: What is FedOps
nav_order: 2
---
# **FedOps: Federated Learning Lifecycle Operations Management**

## What is FedOps?
Since the concept of Federated Learning was introduced, numerous research efforts have been dedicated to exploring this paradigm. However, many researchers primarily conduct experiments on single machines, rather than on distinct devices, mainly due to challenges in simulating experiments on actual devices. This arises from the significant effort required to partition the ongoing development models for experimentation or the difficulty in managing the entire lifecycle of Federated Learning MLOps tools, making it challenging to obtain detailed experiment results.

To address these challenges, we have developed FedOps, representing an all-encompassing tool that allows for the easy portability of existing models to the federated learning environment and facilitates the management of the entire lifecycle.

During the development of FedOps, we identified three key problems to solve:

Q1. How can we seamlessly transition existing AI/ML projects to the federated learning environment?

Q2. How can we easily involve numerous clients in the federated learning environment, akin to MLOps?

Q3. How should we manage the lifecycle of clients and servers?

We propose the following solutions to address these problems.

![FedOps_Architecture](../img/fedops_architecture.png)

A1. Utilize a flow-based code structure to seamlessly migrate users' models and data to the federated learning environment without direct manipulation.

A2. Implement CI/CD using Git to continuously maintain clients and servers, facilitating ongoing federated learning.

A3. Employ client and server managers to monitor and manage each other's states continually.

FedOps is the platform that encompasses these three solution approaches.

## Why FedOps?

FedOps is easy to develop and deploy.

FedOps, being an extended platform from traditional MLOps, allows for immediate application without requiring modifications to existing models. Moreover, deploying both clients and servers through Git enables easy expansion to actual devices. Unlike traditional Federated Learning MLOps lacking robust functionality to monitor the lifecycle of clients and servers, FedOps provides a comprehensive view of the lifecycle of all entities participating in federated learning through client and server managers. This significant difference and advantage allow monitoring the lifecycle of actual devices, addressing issues that arise during training, and facilitating easy problem resolution.

![FedOps_Overview](../img/FedOps_Overview.PNG)