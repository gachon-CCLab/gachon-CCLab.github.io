---
layout: default
title: FedOps Package
nav_order: 4
# h2부터 h5까지 제목을 표시합니다
toc_min_heading_level: 2
toc_max_heading_level: 5
---


# FedOps Package

- [Client](#client)
  - [app](#App)
  - [client_api](#Client API)
  - [client_fl](#client_fl)
  - [client_utils](#client_utils)
  - [client_wandb](#client_wandb)
  
- [Server](#server)
  - [app](#app)
  - [server_api](#server_api)
  - [server_utils](#server_utils)
  
## Client
### App
```python
class fedops.client.app.FLClientTask
```
`Register the FL Task to FedOps Client`


```python
    async def fl_client_start()
```
`Client participates in FL task round`

- The reason for processing asynchronously is that the client is configured so that FL can be performed without affecting it under the assumption that the client continues to perform tasks such as data collection and analysis.

 
```python
    def start()
```
`API that communicates with the Client Manager, delivers the status of the Client, receives an FL start trigger, and participates in the FL task round.`


```python
        @self.app.get('/online')
        async def get_info()
```
`Check the client's status`
- **RETURN**
  - self.status: Client's information(FL_task_id, FL_client_num, FL_client_mac, FL_client_online, FL_client_start, FL_client_fail, FL_server_IP, FL_next_gl_model)


```python
        @self.app.post("/start")
        async def client_start_trigger(background_tasks: BackgroundTasks, manager_data: client_utils.ManagerData)
```
`Asynchronous background client participates in FL task round`
- **PARAMETERS**
  - BackgroundTasks: This is useful for operations that need to happen after a request, but that the client doesn't really have to be waiting for the operation to complete before receiving the response.
  - client_utils.ManagerData: IP of server to connect with FL server, client mac address for unique value of client


### Client API
#### Client Manager API
```python
class fedops.client.client_api.ClientManagerAPI
```
`API for communication with client manager`

```python
    def get_info()
```
`Get the client manager information`

```python
    def get_client_out()
```
`Client out due to error or normal FL termination`

```python
    def get_train_fin()
```
`End of client local training`

```python
    def get_train_fail()
```
`Fail Client local training with error`

<br>

#### Server Manager API

```python
class fedops.client.client_api.ClientServerAPI
```
`Get FL server pod ip and port that assigned to the FL task`

```python
    def get_port()
```
`G`

```python
    def get_client_out()
```
`C`

```python
    def get_train_fin()
```
`E`

```python
    def get_train_fail()
```
`F`



## Server
```
class fedops.server.app.FLServer
```
