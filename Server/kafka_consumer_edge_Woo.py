import os
import json
import base64
from kafka import KafkaConsumer

# ----------------- 설정 부분 -----------------
KAFKA_BROKER = '100.70.106.105:9092'  # 실제 Tailscale IP
KAFKA_TOPIC = 'edge_data_topic_Woo'       # Producer와 동일한 토픽 이름
GROUP_ID = 'edge-consumer-group-file' # 그룹 ID 변경 (처음부터 다시 받기 위함)

# 수신된 파일들을 저장할 최상위 디렉토리
OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__), 'received_data')
# ---------------------------------------------

def decode_base64_to_file(b64_string, output_path):
    """Base64 문자열을 디코딩하여 물리적 파일로 저장합니다."""
    if not b64_string:
        return
    try:
        file_data = base64.b64decode(b64_string)
        with open(output_path, 'wb') as f:
            f.write(file_data)
    except Exception as e:
        print(f"❌ 파일 복원 오류 ({output_path}): {e}")

def save_dict_to_json(data_dict, output_path):
    """딕셔너리 데이터를 JSON 파일로 저장합니다."""
    if not data_dict:
        return
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"❌ JSON 저장 오류 ({output_path}): {e}")

def start_consumer():
    """Kafka 토픽을 구독하고 수신된 데이터를 로컬 파일로 복원(저장)합니다."""
    # 저장할 기본 디렉토리 생성
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

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

        for message in consumer:
            payload = message.value
            
            category = payload.get('category', 'unknown')
            base_name = payload.get('base_name', f'unknown_{message.offset}')
            
            print(f"📥 [메시지 수신] 카테고리: {category} | Base Name: {base_name}")
            
            # 카테고리(pr, sd)별 하위 폴더 생성 (예: received_data/pr)
            category_dir = os.path.join(OUTPUT_BASE_DIR, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # 복원할 파일들의 최종 경로 설정
            json_path = os.path.join(category_dir, f"{base_name}.json")
            jpg_path = os.path.join(category_dir, f"{base_name}.jpg")
            xlsx_path = os.path.join(category_dir, f"{base_name}.xlsx")
            
            # 1. JSON (Metadata) 복원
            metadata = payload.get('metadata')
            if metadata:
                save_dict_to_json(metadata, json_path)
                
            # 2. JPG 복원
            image_b64 = payload.get('image_base64')
            if image_b64:
                decode_base64_to_file(image_b64, jpg_path)
                
            # 3. XLSX 복원
            excel_b64 = payload.get('excel_base64')
            if excel_b64:
                decode_base64_to_file(excel_b64, xlsx_path)
                
            print(f"   └── ✅ 파일 3종 복원 및 저장 완료! ({category_dir} 폴더 확인)")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 시스템이 안전하게 종료되었습니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
    finally:
        if 'consumer' in locals():
            consumer.close()

if __name__ == "__main__":
    start_consumer()