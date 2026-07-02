FROM python:3.10.12-slim

WORKDIR /shap

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
## 모두 복사
COPY analyze.py .
COPY kafka_manager_shap.py .
COPY ./result_cause /shap/result_cause


# 작성하신 app.py가 이 경로의 모델을 찾도록 환경변수 고정
ENV MODEL_PATH=/pr/best.pt

# 통신 및 추론 파일 실행
CMD ["python", "kafka_manager_shap.py"]
