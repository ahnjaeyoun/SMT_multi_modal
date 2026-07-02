import os
import json
import base64
import time
from collections import Counter

from kafka import KafkaConsumer, KafkaProducer
from pymongo import MongoClient
from datetime import datetime
from ultralytics import YOLO

import numpy as np
import cv2
from utils import *
# model: YOLO | None = None
model_pr=YOLO('./best_pr.pt')
model_sd=YOLO('./best_sd.pt')

# ----------------- 설정 부분 -----------------
KAFKA_BROKER = '100.70.106.105:9092'  # 실제 Tailscale IP
KAFKA_TOPIC = 'edge_data_topic_Goo'       # Producer와 동일한 토픽 이름
KAFKA_TOPIC_PR = 'pr'       # pr 전송용 별도 토픽
KAFKA_TOPIC_SHAP='shap'
GROUP_ID = 'edge-consumer-group-file' # 그룹 ID 변경 (처음부터 다시 받기 위함)

# MongoDB 설정
MONGO_URL = "mongodb://woo:young@100.70.106.105:27017"
mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db = mongo_client["ai_factory_db"]
results_collection = db["obb_results"]
sensor_collection= db["sensor_data"]

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
    """Kafka 토픽을 구독및 데이터 수신."""

    print(f"🔄 Kafka 브로커({KAFKA_BROKER})의 '{KAFKA_TOPIC}' 토픽 수신 대기 중...")
    try:  
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            group_id=GROUP_ID,
            # 대용량 Base64 메시지도 처리할 수 있도록 fetch 크기 증가 (10MB)
            fetch_max_bytes=10485760,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            # 새로운 그룹ID로 시작하므로 'earliest'로 설정하면 기존에 쌓인 데이터를 처음부터 다 받습니다.
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        print("✅ Consumer 준비 완료! 데이터를 수신하여 파일로 저장합니다. (종료: Ctrl+C)")
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            # JSON 형태로 자동 직렬화 (Base64가 포함되므로 크기가 커질 수 있음)
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            max_request_size=10 * 1024 * 1024,  # 10MB
            request_timeout_ms=10000,
            max_block_ms=10000
        )
        producer_shap = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            # JSON 형태로 자동 직렬화 (Base64가 포함되므로 크기가 커질 수 있음)
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            max_request_size=10 * 1024 * 1024,  # 10MB
            request_timeout_ms=10000,
            max_block_ms=10000
        )
        print("Kafka 브로커 연결 성공!")
    except Exception as e:
        print(f"Kafka 연결 실패: {e}")
    print("수신 대기 중...")
    # try:
    for message in consumer:
        payload = message.value
        category = payload.get('category', 'unknown')
        base_name = payload.get('base_name', f'unknown_{message.offset}')
        
        print(f"📥 [메시지 수신] 카테고리: {category} | Base Name: {base_name}")
            
        # 2. JPG 복원
        image_b64 = payload.get('image_base64')
        if image_b64:
            img= decode_base64_to_file(image_b64)
            image_np = np.frombuffer(img, dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # excel를 사용하지 않으니 그대로 전송
        excel_b64 = payload.get('excel_base64')
        #3. 모델 예측
            # 공정 단계 확인
        if base_name.split('_')[0]=='PR':
            model=model_pr
        else: model=model_sd
        result=predict(model, image, base_name)
            ## 하나라도 불량이 있으면 불량으로 간주 -> 이름 그대로 사용!
        # 카테고리 지정: 가장 많은 불량유형을 category_id로 지정!
        class_total=[]
        conf_total=[]
        for i in result['detections']:
            conf_total.append(i['confidence'])
        if len(conf_total)>0:
            conf= sum(conf_total)/ len(conf_total)
        else: conf=99.99

        #4. bbox 표기 및 이미지 저장
        img_path=draw_bbox(image, result, output_path='./bbox_image/', filename=base_name+'.jpg')
        print(f"   └── ✅ 파일 복원 및 추론, bbox이미지 생성 완료!")
        print("-" * 50)

        # 3. 데이터 전송

        # 이미지 base64 문자열로 변환   -> 현재 디렉토리에 저장으로 변경!

        # byte_image= base64.b64encode(bbox_image).decode("utf-8")

        # 공정_불량여부_불량카테고리(일단제외)_공장_시간으로 변경 -> 이름 수정 -> 취소!
        name_parts= base_name.split('_')
        class_id= DEF_CLS[name_parts[2]]
        # process=name_parts[0]
        # factory=name_parts[3]
        # current_time= datetime.now().strftime("%Y%m%d-%H%M%S")
        # new_name= f'{process}_{status}_{factory}_{current_time}'

        absolute_path = os.path.abspath(img_path)
        ## 전송할 값들 딕셔너리에 저장
        payload={
            "category": category,
            "cause": class_id,
            "file_name": base_name,
            "confidence": conf,
            'excel_file': excel_b64,
        }
        print('shap:' ,class_id)
        payload_shap= {
            "category": category,
            "cause": class_id,
            "file_name": base_name,
            "confidence": conf,
            'excel_file': excel_b64,
            'category_id': class_id,
            'sensor_file': excel_b64
        }

        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        print(f"전송 payload: {len(payload_bytes) / 1024:.2f} MB")

        # 5. DB 저장 데이터 구성
        name_parts = base_name.split('_')
        process_type = name_parts[0] if len(name_parts) > 0 else category
        

        # 6. 데이터 전송
        # absolute_path = os.path.abspath(img_path)
        try:  
            # print(f"토픽 '{KAFKA_TOPIC_PR}'으로 메시지 전송 시도...")
            # future = producer.send(KAFKA_TOPIC_PR, value=payload)
            print(f"토픽 '{KAFKA_TOPIC_SHAP}'으로 메시지 전송 시도...")
            future_2 = producer_shap.send(KAFKA_TOPIC_SHAP, value=payload_shap)
            # 전송 대기 (동기식)
            record_metadata = future_2.get(timeout=10)
            print(f"전송 성공! (파티션: {record_metadata.partition}, 오프셋: {record_metadata.offset})")
        except Exception as e:
            print(f"전송 실패: {e}")
        payload_db={
            "category": category,
            "cause": class_id,
            "file_name": base_name,
            "confidence": conf,
            'excel_file': excel_b64,
            "img_path": img_path
        }
        # DB 저장 시도
        try:
            insert_result = results_collection.insert_one(payload_db)
            print(f"✅ DB 저장 완료! (ID: {insert_result.inserted_id})")
        except Exception as e:
            print(f"❌ DB 저장 실패: {e}")
        
        # 10초 대기 후 다음 세트 전송
        print("10초 대기 중...")
        # time.sleep(10)
    
                    
    # except KeyboardInterrupt:
    #     print("\n🛑 사용자에 의해 시스템이 안전하게 종료되었습니다.")
    # except Exception as e:
    #     print(f"❌ 실행 중 오류 발생: {e}")
    # finally:
    #     if 'consumer' in locals():
    #         consumer.close()

if __name__ == "__main__":
    start_consumer()