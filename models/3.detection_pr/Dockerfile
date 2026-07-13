FROM python:3.10.12-slim

WORKDIR /pr

# YOLO(OpenCV) 구동에 필요한 리눅스 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

## 한글 폰트 다운로드
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-nanum \
        fontconfig \
    && fc-cache -f -v \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install kafka-python
RUN python -c "from kafka import KafkaConsumer, KafkaProducer; print('kafka-python 설치 성공')"
## 모두 복사
COPY pr_inference.py .
COPY best_pr.pt .
COPY best_sd.pt .
COPY utils.py .


# 작성하신 app.py가 이 경로의 모델을 찾도록 환경변수 고정
ENV MODEL_PATH=/pr/best.pt

# 통신 및 추론 파일 실행
CMD ["python", "pr_inference.py"]
