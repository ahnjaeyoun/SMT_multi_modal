import os
import json
import base64
from kafka import KafkaConsumer, KafkaProducer
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
import numpy as np
import cv2
from pymongo import MongoClient

from utils import *

model=YOLO('./best.pt')

# ----------------- 설정 부분 -----------------
KAFKA_BROKER = '100.70.106.105:9092'  # 실제 Tailscale IP
KAFKA_TOPIC = 'edge_data_topic_Goo'       # Producer와 동일한 토픽 이름
KAFKA_TOPIC_PR = 'pr'       # pr 전송용 별도 토픽
GROUP_ID = 'edge-consumer-group-file' # 그룹 ID 변경 (처음부터 다시 받기 위함)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()

# 수신된 파일들을 저장할 최상위 디렉토리
OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__), 'received_data')

# MongoDB 설정
MONGO_URL = "mongodb://woo:young@100.70.106.105:27017"
mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db = mongo_client["ai_factory_db"]
collection = db["inference_results"]
# ---------------------------------------------

def decode_base64_to_file(b64_string):
    """Base64 문자열을 디코딩하여 물리적 파일로 저장합니다."""
    if not b64_string:
        return
    try:
        file_data = base64.b64decode(b64_string)
        return file_data
    except Exception as e:
        print(f"❌ 파일 복원 오류: {e}")

def start_consumer():
    """Kafka 토픽을 구독하고 수신된 데이터를 로컬 파일로 복원(저장)합니다."""
    # 저장할 기본 디렉토리 생성
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print(f"🔄 Kafka 브로커({KAFKA_BROKER})의 '{KAFKA_TOPIC}' 토픽 수신 대기 중...")
    
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        group_id=GROUP_ID,
        fetch_max_bytes=10485760,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )
    print("✅ Consumer 준비 완료! 데이터를 수신하여 파일로 저장합니다. (종료: Ctrl+C)")
    
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        max_request_size=10 * 1024 * 1024,  # 10MB
        request_timeout_ms=10000,
        max_block_ms=10000
    )
    print("Kafka 브로커 연결 성공!")
    print("수신 대기 중...")
    
    for message in consumer:
        payload = message.value
        category = payload.get('category', 'unknown')
        base_name = payload.get('base_name', f'unknown_{message.offset}')
        
        print(f"📥 [메시지 수신] 카테고리: {category} | Base Name: {base_name}")
            
        # 2. JPG 복원
        image_b64 = payload.get('image_base64')
        if image_b64:
            img = decode_base64_to_file(image_b64)
            image_np = np.frombuffer(img, dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
        excel_b64 = payload.get('excel_base64')
        
        # 3. 모델 예측
        result = predict(model, image, base_name)
        status = 'NOR'
        
        # 불량 유형 및 정확도 추출
        defect_types = []
        accuracies = []
        
        for i in result.get('detections', []):
            # class_id-1 >= 16 이면 불량으로 판정하던 기존 로직
            if i.get('class_id', 0) - 1 >= 16:
                status = 'DEF'
                defect_types.append(i.get('class_name', str(i.get('class_id'))))
            accuracies.append(i.get('confidence', 0.0))

        # 4. bbox 표기 및 이미지 저장
        img_path = draw_bbox(image, result, output_path='./bbox_image/', filename=base_name+'.jpg')
        print(f"   └── ✅ 파일 복원 및 추론, bbox이미지 생성 완료!")
        print("-" * 50)

        # 5. DB 저장 데이터 구성
        name_parts = base_name.split('_')
        process_type = name_parts[0] if len(name_parts) > 0 else category
        
        db_data = {
            "file_name": base_name,
            "process_type": process_type,
            "status": status,
            "defect_types": defect_types if status == 'DEF' else [],
            "accuracy": max(accuracies) if accuracies else 0.0,
            "all_accuracies": accuracies,
            "timestamp": datetime.now()
        }
        
        # DB 저장 시도
        try:
            insert_result = collection.insert_one(db_data)
            print(f"✅ DB 저장 완료! (ID: {insert_result.inserted_id}) - 상태: {status}")
        except Exception as e:
            print(f"❌ DB 저장 실패: {e}")

        # 6. 데이터 전송
        absolute_path = os.path.abspath(img_path)
        payload = {
            "category": category,
            "file_name": base_name,
            'excel_file': excel_b64,
            "predictions": result,
            "image_path": absolute_path,
            "db_status": "saved"
        }
        
        # payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        print(f"토픽 '{KAFKA_TOPIC_PR}'으로 메시지 전송 시도...")
        future = producer.send(KAFKA_TOPIC_PR, value=payload)
        record_metadata = future.get(timeout=10)
        print(f"전송 성공! (파티션: {record_metadata.partition}, 오프셋: {record_metadata.offset})")
        
        print("10초 대기 중...")
        time.sleep(10)

if __name__ == "__main__":
    start_consumer()
