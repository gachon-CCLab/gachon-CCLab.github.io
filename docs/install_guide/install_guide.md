# FedOps Install Guide (Manual ver.)

Created: 2022년 11월 8일 오전 7:50
Last Edited Time: 2022년 11월 9일 오후 3:39
Status: In Progress
Tags: Guide
Writer/Maintainer: 문지환, 양세모

<aside>
  본 설치는 Ubuntu 20.04.5 LTS (Focal Fossa) 환경에서 진행되었습니다.

</aside>

<aside>
  (네트워크 설정이 되어있는) VM 등 가상환경에서도 설치가 가능합니다.

</aside>

# FATE(Federated AI Technology Enabler)

<aside>
  Link 1: [https://github.com/FederatedAI/FATE](https://github.com/FederatedAI/FATE)
Link 2: [https://fate.readthedocs.io/en/latest/](https://fate.readthedocs.io/en/latest/)

</aside>

- FATE: 기업과 기관이 데이터 보안 및 개인 정보를 보호하면서 데이터에 대해 협업할 수 있도록 하는 세계 최초의 산업 등급 연합 학습 오픈 소스 프레임워크
- Deployment
    - [독립형 배포](https://github.com/FederatedAI/FATE/wiki/Download)
    - 클러스터 배포
        - 확장성, 안정성 및 관리 용이성을 달성하기 위해 FATE를 여러 노드에 배포
        - [**KubeFATE](https://github.com/FederatedAI/KubeFATE): Container 및 Kubernetes 활용한 FATE 플랫폼 운영 도구**
        - [CLI](https://github.com/FederatedAI/FATE/blob/master/deploy/cluster-deploy)
        - [Ansible](https://github.com/FederatedAI/AnsibleFATE)
- Related Repository
    - [FATE-Flow](https://github.com/FederatedAI/FATE-Flow): 연합 학습 파이프라인을 위한 다자간 보안 작업 스케줄링 플랫폼
    - [FATE-Board](https://github.com/FederatedAI/FATE-Board): 연합 모델을 쉽고 효과적으로 탐색하고 이해할 수 있는 시각화 도구
    - [FATE-Serving](https://github.com/FederatedAI/FATE-Serving): 연합 학습 모델을 위한 고성능의 프로덕션 준비 서빙 시스템
    - [FATE-Cloud](https://github.com/FederatedAI/FATE-Cloud): 산업 등급 연합 학습 클라우드 서비스를 구축하고 관리하기 위한 인프라
    - [EggRoll](https://github.com/WeBankFinTech/eggroll): (연합된) 기계 학습을 위한 간단한 고성능 컴퓨팅 프레임워크
    - [AnsibleFATE](https://github.com/FederatedAI/AnsibleFATE): Ansible을 통해 구성 및 배포 작업을 최적화하고 자동화하는 도구
    - [FATE-Builder](https://github.com/FederatedAI/FATE-Builder): FATE 및 KubeFATE용 패키지 및 도커 이미지를 빌드하는 도구
    
    ![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled.png)
    

# Kubernetes 기반의 FedOps

### 쿠버네티스 설치

```bash
sudo snap install microk8s --classic --channel=1.22/stable
```

snap을 통해 설치하고자 하는 컴퓨터(노드)에서 microk8s를 설치합니다. 버전은 1.22를 사용합니다.

### Microk8s (addon) dns, storage 추가

```bash
microk8s enable dns storage
```

dns와 storage 애드온을 활성화 시킵니다. 만약 multi node cluster를 구성할 예정이라면 호스트가 될 노드 하나에서만 실행하면 됩니다.

### (선택) Node 추가, HA Cluster 구성

<aside>
  multi node cluster를 구성하는 경우 아래 내용을 진행합니다.

</aside>

```bash
# Master node
microk8s add-node
```

```bash
microk8s join <address and token here>
```

```bash
ccl@ccl-d-server:~$ kubectl get no
NAME              STATUS   ROLES    AGE   VERSION
ccl-e-server      Ready    <none>   42h   v1.22.15-3+6a8fc82baa4d77
ccl-d-server      Ready    <none>   42h   v1.22.15-3+6a8fc82baa4d77
ccl-c-server-vm   Ready    <none>   37h   v1.22.15-3+6a8fc82baa4d77
```

```bash
ccl@ccl-d-server:~$ microk8s status
microk8s is running
high-availability: yes
  datastore master nodes: 192.9.201.228:19001 192.9.201.74:19001 192.9.202.178:19001
  datastore standby nodes: none
```

### Helm 활성화

- helm: Kubernetes 패키지 설치/관리를 도와주는 것(패키지매니저)으로, yaml 파일의 모음

```bash
microk8s enable helm3
```

### Lens에 클러스터 추가하기

```bash
ccl@ccl-d-server:~$ microk8s config
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUREekNDQWZlZ0F3SUJBZ0lVQ081SnFscmwzN0tlUUlwWDQxTjRrZlk5aExJd0RRWUpLb1pJaHZjTkFRRUwKQlFBd0Z6RVZNQk1HQTFVRUF3d01NVEF1TVRVeUxqRTRNeTR4TUI0WERUSXlNVEF3TnpBeE16WTBOVm9YRFRNeQpNVEF3TkRBeE16WTBOVm93RnpFVk1CTUdBMVVFQXd3TU1UQXVNVFV5TGpFNE15NHhNSUlCSWpBTkJna3Foa2lHCjl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUE5VUdTN0hsNGI4SU5VY3hkZXczQko1eFhkV2tiVXo2cTN1cTEKWVJoeFRGRXQ2QWZEVHdvUjZEdHRwdUtZODBEWFBJY1hOUzRNekZLa0FQNXQwS3RCTGhVMFEwVW9xN3hYMWZFSwpIa0paOUY0M01wc216WklFUXljYVR1UWRxb3FSaEVkU25SZE9sK243ZmNiN2ZYRGh4K1pFMG9reWRDVWg5aWx1CmpvYmtrVU9sdXBTdGtZODFJR0Y1dm1rZitTVm9iRVZVdFVtMk1VbGdINmFQek9ZY2hYVHBVTXdWWk04Rk03YUoKdldtQVlJMTFkYWRWNE5ERG5JZ3hzN3o0S0hMZGowZi9waDJnd2RkYVExLzZvY2xNYzl4WnMzZHJiVDBpb0NOLwpQbE1vMVBTMGtLR2pFU01sVk94RHZBdjE3UlFDdVhCN1hUeUVwR2kxTzJZNVN3Zlgxd0lEQVFBQm8xTXdVVEFkCkJnTlZIUTRFRmdRVW1rR2c4RU1IdXVBTkFzTDA0Nit1NlZZeTNyOHdId1lEVlIwakJCZ3dGb0FVbWtHZzhFTUgKdXVBTkFzTDA0Nit1NlZZeTNyOHdEd1lEVlIwVEFRSC9CQVV3QXdFQi96QU5CZ2txaGtpRzl3MEJBUXNGQUFPQwpBUUVBTy9OVEFZTFYzRlZLVlgvYUFIY3AvMk85T1J2REFPUDFxYzB5Wll0UE5YbkdlREFkZHp1MU9zZ2lwQ1ZLClFhY2tXbnc3MmZLOEgrU0k0L2Z1dFdhQ1dsK3ErSDE4a3JNd0Q3c05iekF4UVM5Um5BekJSQjZ6NVBGamh6TWwKWk9yeUh4dTNEVW5JSXplK09UYnVsWEpEOHdBVjRDb0xMcitVbUtCcG8ydkI2YnNrcmJFRkJ2Z1NzSzBvYW5UYwphNWQvT3ZPaXhDd1BXWDlNd0hYeXJGeE1DNm9yUVlzLzNtQzQrTVQvM0NtcTVXSFRyVEJoRm12eTFXK3EvdzVSCit1QmptY2pDVVhMWnZGWXBDZEZzV1JUUEJueXhRZnFMbDRueHlTMzRrd2poU1VtS0g2RXIxY1labGM3VG9kSisKcDdsSGUydy94ODlWR01mN2VaVWJPeWZHZmc9PQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg==
    server: **https://192.9.201.228:16443**
  name: microk8s-cluster
contexts:
- context:
    cluster: microk8s-cluster
    user: admin
  name: microk8s
current-context: microk8s
kind: Config
preferences: {}
users:
- name: admin
  user:
    token: YkQ2NGN3QndsR2FwZ1FqN0txaTUrMzV3OFZXbnhXSDFUamdvblNhM1N5ND0K
```

 **microk8s config**  명령어로 나온 값안의 주소값(192.9.201.228)을 **127.0.0.1로 변경**해주고 전체를 복사합니다.

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%201.png)

우하단에 있는  **Add from kubeconfig**  클릭합니다.

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%202.png)

나오는 텍스트 창에 붙여넣습니다.

### Prometheus 활성화

```bash
microk8s enable metrics-server promehteus
```

메트릭을 수집하고 시각화하는 prometheus 애드온을 설치해줍니다.

### Service Mesh / Load Balancer 설치

```bash
microk8s enable ingress istiod
```

```bash
ccl@ccl-d-server:~$ microk8s enable metallb
Enabling MetalLB
Enter each IP address range delimited by comma (e.g. '10.64.140.43-10.64.140.49,192.168.0.105-192.168.0.111'): 192.168.1.240/28,10.0.0.0/28
```

```bash
sudo microk8s enable linkerd
```

로드 밸런싱 애드온을 설치해줍니다.

### (선택) NFS Provisioner 설정

<aside>
  NFS로 스토리지를 프로비저닝 하는 경우 아래 절차를 진행합니다.

</aside>

<aside>
  본 설치 과정에서 NFS Server는 Synology NAS로 구성하였으나 Ubuntu server 등 추가적인 옵션도 고려 가능합니다.

</aside>

### nfs-common 패키지 설치

```bash
apt-get install nfs-common
```

노드에서 NFS 패키지를 설치해줍니다.

### NAS NFS 서버 규칙 변경

![NFS Server가 켜져 있는지 확인](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%203.png)

NFS Server가 켜져 있는지 확인

![NFS Server에 접근할 수 있는 IP 대역 (CIDR) 확인](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%204.png)

NFS Server에 접근할 수 있는 IP 대역 (CIDR) 확인

### NFS-Subdir-External-Provisioner 설치

NAS의 NFS 서버를 이용해 클러스터에 마운트하고 Storage를 Provisioning 해주는 패키지 설치.

```bash
ccl@ccl-d-server:~/Desktop$ helm install nfs-subdir-external-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
>     --set nfs.server=192.9.202.101 \
>     --set nfs.path=/volume3/디스크03/k8s/home
NAME: nfs-subdir-external-provisioner
LAST DEPLOYED: Mon Oct 10 16:25:41 2022
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

nfs.server → NAS의 IP주소 

nfs.path → 연결할 디렉토리의 path

### StorageClass 생성

([https://kubernetes.io/docs/concepts/storage/storage-classes/](https://kubernetes.io/docs/concepts/storage/storage-classes/))

([https://waspro.tistory.com/771](https://waspro.tistory.com/771))

StorageClass를 이용하면 Provisioning하는 Storage의 구분을 할 수 있고 삭제, 백업 정책 등을 구분할 수 있습니다. storage system의 ‘Profile’이라고 생각할 수 있는 개념입니다.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
    name: airflow-nfs
provisioner: cluster.local/nfs-subdir-external-provisioner
parameters:
    pathPattern: "${.PVC.namespace}-${.PVC.name}" 
    onDelete: delete
```

### Default StorageClass 변경

([https://kubernetes.io/docs/tasks/administer-cluster/change-default-storage-class/](https://kubernetes.io/docs/tasks/administer-cluster/change-default-storage-class/))

```bash
kubectl patch storageclass microk8s-hostpath -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'
kubectl patch storageclass airflow-nfs -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

```bash
ccl@ccl-d-server:~/Desktop$ kubectl get sc
NAME                    PROVISIONER                                     RECLAIMPOLICY   VOLUMEBINDINGMODE   ALLOWVOLUMEEXPANSION   AGE
nfs-client              cluster.local/nfs-subdir-external-provisioner   Delete          Immediate           true                   2d22h
microk8s-hostpath       microk8s.io/hostpath                            Delete          Immediate           false                  6d3h
airflow-nfs (default)   cluster.local/nfs-subdir-external-provisioner   Delete          Immediate           false                  2d21h
```

### ArgoCD 설치

- 역할
    - **Flower Framework 기반의 여러 Client Pod 생성**
    - **ELK + FileBeat로 로그 수집/시각화 Pod 생성**
- namespace 생성

```bash
kubectl create namespace argocd
```

- 배포

```bash
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

- 배포 확인

```bash
kubectl get all -n argocd
```

- Lens로 접속

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%205.png)

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%206.png)

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%207.png)

### ArgoCD에서 APP(Client Pod) 생성

다음 문서의 [ArgoCD 파트](https://www.notion.so/FedMLops-Guide-9b731598c01848839ddebfd1747e1212?pvs=21)를 참고하세요.

### Airflow 설치

- 역할
    - Flower Framework 기반의 Server를 주기적으로 실행
- helm repository에 추가

```bash
helm repo add apache-airflow https://airflow.apache.org
```

- helm으로 airflow chart를 설치
    - 버전: 1.5.0
    - 파라미터로 gitSync 등 옵션을 활성화
        - --set dags.gitSync.repo 부분에 Flower Server의 git 주소 설정

```bash
helm install airflow apache-airflow/airflow --namespace airflow --version 1.5.0 --set dags.gitSync.enabled=true --set dags.gitSync.branch=master --set dags.gitSync.repo=https://github.com/gachon-CCLab/airflow-dag.git --set dags.gitSync.subPath=dags --set postgresql.volumePermissions.enabled=true --create-namespace
```

## ELK + FileBeat로 로그 수집/시각화 환경 구성

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%208.png)

위와 같이 4개의 앱을 이용해 로그 수집/파싱/저장/시각화 시나리오를 구성합니다.

### 설치

ArgoCD에서 필요한 앱(FileBeat/Logstash/Elasticsearch/Kibana)의 설치를 진행합니다.

**FileBeat:**

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%209.png)

**Logstash:**

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2010.png)

### Logstash Pipeline config 활성화

([https://github.com/elastic/helm-charts/tree/7.17/logstash](https://github.com/elastic/helm-charts/tree/7.17/logstash))

Logstash에서 로그를 파싱하고 포맷팅하기 위해서 Configmap에서 pipeline을 설정해주어야 합니다. 

Default 값으로 설치 시에는 configmap이 생성되지 않아 다음과 같은 방법으로 설정합니다.

```yaml
# values.yaml
replicas: 1

# Allows you to add any config files in /usr/share/logstash/config/
# such as logstash.yml and log4j2.properties
#
# Note that when overriding logstash.yml, `http.host: 0.0.0.0` should always be included
# to make default probes work.
logstashConfig: {}
#  logstash.yml: |
#    key:
#      nestedkey: value
#  log4j2.properties: |
#    key = value

# Allows you to add any pipeline files in /usr/share/logstash/pipeline/
### ***warn*** there is a hardcoded logstash.conf in the image, override it first
**logstashPipeline:
  logstash.conf: |
    input {
      exec {
        command => "uptime"
        interval => 30
      }
    }
    output { stdout { } }**

...
```

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2011.png)

ArgoCD를 통해 설치 시 values.yaml 파일을 override해 logstash pipeline configmap이 만들어지도록 합니다.

**ElasticSearch:**

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2012.png)

**Kibana:**

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2013.png)

### Connect FileBeat to Logstash

<aside>
  Lens를 통해 ConfigMap 등 resource의 수정을 할 수 있습니다.

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2014.png)

</aside>

([https://stackoverflow.com/questions/42255331/beat-and-logstash-connection-reset-by-peer/42340533#42340533?newreg=eb788fa5d7974c1a8bc8d03f9e38e6a2](https://stackoverflow.com/questions/42255331/beat-and-logstash-connection-reset-by-peer/42340533#42340533?newreg=eb788fa5d7974c1a8bc8d03f9e38e6a2))

```yaml
# ConfigMap: filebeat-filebeat-daemonset-config
...
# output.elasticsearch:
  # host: '${NODE_NAME}'
  # hosts: '${ELASTICSEARCH_HOSTS:elasticsearch-master:9200}'
output.logstash:
  host: '${NODE_NAME}'
  hosts: [logstash-logstash-headless.elk.svc.cluster.local]
```

### Connect Logstash to Elasticsearch

([https://www.elastic.co/guide/en/logstash/current/plugins-inputs-beats.html](https://www.elastic.co/guide/en/logstash/current/plugins-inputs-beats.html))

```yaml
# ConfigMap: elk/logstash-logstash-pipeline
data:
  logstash.conf: |
    input {
      beats {
        port => 5044
      }
    }

    output {
      elasticsearch {
        # hosts => ["https://localhost:9200"]
        **hosts => ["10.1.186.71:9200", "10.1.196.91:9200", "10.1.51.159:9200"]**
        index => "%{[@metadata][beat]}-%{[@metadata][version]}" 
      }
    }
```

위와 같이 logstash configmap을 수정합니다.

### Logstash Pipeline config

([https://www.elastic.co/guide/en/logstash/current/plugins-filters-dissect.html](https://www.elastic.co/guide/en/logstash/current/plugins-filters-dissect.html))

([https://www.elastic.co/guide/en/logstash/current/plugins-filters-json.html](https://www.elastic.co/guide/en/logstash/current/plugins-filters-json.html))

([https://www.elastic.co/guide/en/logstash/current/plugins-filters-mutate.html#plugins-filters-mutate-gsub](https://www.elastic.co/guide/en/logstash/current/plugins-filters-mutate.html#plugins-filters-mutate-gsub))

Logstash를 이용해 원하는 Log를 파싱하고 원하는 형태로 다시 만들어내야 합니다.

- 로그의 형태와 파싱 방법
    
    현재 FL과정에서 만들어내는 로그의 형태는 다음과 같습니다.
    
    [FL Kibana Log](https://www.notion.so/FL-Kibana-Log-1592cc8e7873413f9eddd86dca4e46ad?pvs=21)
    
    ```bash
    2022-10-11 06:18:52,629 [   INFO] train_time - **{"client_num": 0, "round": 5, "next_gl_model": 3, "execution_time": "23.0092"}**
    ```
    
    위 로그의 패턴을 파싱하기 위해 다음과 같은 dissect filter 적용이 가능합니다.
    
    ```bash
    dissect {
      mapping => { "message" => "**%{ts} %{+ts} [    %{loglevel}] %{title} - %{json_contents}**" }
    }
    ```
    
    json_contents라는 파라미터로 얻은 json object에 다시 json filter를 적용해 파싱합니다.
    
    ```bash
    json {
      source => "json_contents"
      target => "doc"
    }
    ```
    

```yaml
# ConfigMap: elk/logstash-logstash-pipeline
apiVersion: v1
data:
  logstash.conf: |
    input {
      beats {
        port => 5044
      }
    }

    **filter {
      dissect {
        mapping => { "message" => "%{ts} %{+ts} [    %{loglevel}] %{title} - %{json_contents}" }
      }**

      mutate {
        gsub => [ "message", "\},\{", "\r\n"]
      }

      **json {
        source => "json_contents"
        target => "doc"
      }

    }**

    output {
      elasticsearch {
        hosts => ["10.1.186.71:9200", "10.1.196.91:9200", "10.1.51.159:9200"]
        index => "%{[@metadata][beat]}-%{[@metadata][version]}" 
      }
    }
kind: ConfigMap
...
```

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2015.png)

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2016.png)

filter를 적용한 결과 위와 같이 파싱되었습니다. 각 파라미터 값이 시각화에 바로 사용할 수 있도록 나온 것을 확인 가능합니다.

### Kibana Index Pattern

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2017.png)

Kibana에서 **Management - Kibana - Index Patterns**에 들어가 **create index pattern** 버튼을 클릭합니다.

이름을 **filebeat-***로 입력해 filebeat에서 나오는 모든 로그를 패턴으로 받습니다.

### (선택)Kibana Visualization 만들기

![title: “server_time”, X축: Round, Y축: Operation Time](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-10-13_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_6.05.45.png)

title: “server_time”, X축: Round, Y축: Operation Time

(1) Search Query: 검색 쿼리로 시각화에 사용할 데이터로 범위를 좁혀 사용할 수 있음.

(2) Horizontal axis: 가로축에 해당. 화면 좌측에 Fields로부터 사용할 데이터를 가져와 사용할 수 있음. 

(3) Vertical axis: 세로축에 해당.

Filter나 Formula를 이용해 값을 Custom 하면서 시각화도 가능합니다.

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2018.png)

![Kibana Visualization Dashboard 화면](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2019.png)

Kibana Visualization Dashboard 화면

### Kibana Dashboard 불러오기

Kibana에서 **Management - Kibana - Saved Objects**에 들어가 **Import** 버튼을 클릭합니다.

![Untitled](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/Untitled%2020.png)

[export.ndjson](FedOps%20Install%20Guide%20(Manual%20ver%20)%201b29e4486883455588dda4623907b35b/export.ndjson)

위 파일을 통해 대시보드를 바로 불러올 수 있습니다.