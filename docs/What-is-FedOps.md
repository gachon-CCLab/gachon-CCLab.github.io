# What is FedOps?

## **FedOps: Federated Learning Lifecycle Operations Management**

## Motivation


 Need a platform that can manage FL operations by extending the concept of MLOps


![FedOps_Overview](./img/FedOps_Overview.PNG)

  Q1. How can AI/ML Project be easily applied to FL?



  Q2. How to deploy and run numerous clients in FL       environment like MLOps?



  Q3. How can we manage the client/server lifecycle for FL operations?

## FedOps Architecture

![](./img/architecture.PNG)

FedOps has five key features:

1. FLScalize: It simplifies the application of data and models in a FL environment by leveraging Flower's Client and Server.

2.  the manager oversees and manages the real-time FL progress of both clients and server
3. Contribution Evaluation and Client Selection processes incentivize individual clients through a BCFL based on their performance.

4. the CI/CD/CFL system integrates with a Code Repo, 
enabling code deployment to multiple clients and servers for continuous or periodic federated learning

5. the FL dashboard is available for monitoring and observing the lifecycle of FL clients and server
