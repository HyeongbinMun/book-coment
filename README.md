# Book coment 감정 분류 모델

## Introduce
 - 본 문서는 book comet 감정 분류 모델 학습 및 평가에 대한 벤치마크이다.
 - base git code : https://github.com/SKTBrain/KoBERT

## Installation
환경 설정은 docker를 통해 구현하였으며 환경 설정을 하기 위해서는 아래와 같은 패키지 설치가 필요하다.

docker 환경 : cuda : 11.7.1 / cudnn : 8.0 / ubuntu : 20.04
* [Docker](https://docs.docker.com/engine/install/ubuntu/)
* [Docker-compose](https://docs.docker.com/compose/install/)
* [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

1. 해당 패키지 설치 후 docker.compose.yml 파일에서 볼륨 및 포트 변경

```python
# : 기준으로 앞쪽이 자신의 로컬 directory, 뒤쪽이 생성되는 mount directory
    volumes:
      - "/media/mmlab/hdd:/hdd"  # 원하는 디렉토리로 수정
      
# 앞쪽의 25100 ~ 25103 부분의 변경 
    ports:
      - "25100:22"               # ssh port
      - "25101:6006"             # tensorboard port
      - "25102:8000"             # web port
      - "25103:8888"             # jupyter port
      
```

2. docker container 생성 및 접속

```python
docker-compose up -d                # 생성
docker attach [CONTAINER_NAME]      # 접속
```

 - 환경 변경 후 container 재적용 실행 command(단 이전의 작업이 날아갈 수 있음)
```python
docker-compose up -d --build
```

 - ssh port 접근 이외에 docker container 다중 접속
```python
docker exec -it [CONTAINER_NAME] bash
```

3. 해당 패키지 설치
```python
pip install -r requirements.txt
```

4. train
 - train하기 위한 txt 생성(예시는 data directory 참고)
```python
python convert.py \
--data_path 'data_path/train.xlsx' \
--data_type 'train' # type : train(default), test
```
- train code
```python
python train.py \
--train_path 'data_path/train.txt' \
--val_path   'data_path/val.txt'   \
--save_path './result/'
```

5. test
```python
python test.py \
--model_path 'result/last.pt' \
--test_path 'data_path/test.txt' \
--save_path './result/'
```
