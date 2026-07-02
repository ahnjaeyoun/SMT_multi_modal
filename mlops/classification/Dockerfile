FROM python:3.10-slim

WORKDIR /app

# 시스템 라이브러리 설치 (필요시)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 패키지 먼저 설치 (캐싱 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 소스 코드, 모델 가중치 파일, 추론 스크립트 복사
COPY . .

# 환경변수 기본값 (compose에서 덮어쓰기 가능)
ENV KAFKA_BROKER="100.70.106.105:9092"
ENV KAFKA_TOPIC="class"

CMD ["python", "realtime_inference_app.py"]
ENV PYTHONUNBUFFERED=1
