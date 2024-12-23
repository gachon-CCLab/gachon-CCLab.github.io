---
layout: default
title: FedOps Silo
nav_order: 4
---

# FedOps Silo

## Silo Components
![FedOps Silo Image](../img/silo_detail.png)

### Silo Client

- Support silo client env based on docker or shell file
- Check FL client status and occur trigger in manager
- Load data and build/save model in FL client
- Support silo client Dashboard for monitoring

### Silo Server
- Support silo server env based on k8s 
- Create and run FL server for starting FL rounds
- Monitor local/global model performance
- Manage/Save global model according to version


## FedOps Silo Scenario
![FedOps Silo Scenario](../img/silo_scenario.png)<br>
- [FedOps Silo Guide](https://gachon-cclab.github.io/docs/user-guide/silo-guide/)