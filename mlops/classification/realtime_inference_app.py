import os
import json
import base64
import io
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from kafka import KafkaConsumer
import albumentations as A
from albumentations.pytorch import ToTensorV2

from train import DualEncodingModel

# ----------------- Configuration -----------------
# When running in docker-compose, 'kafka' is the hostname for the broker on the internal network.
# If running locally, you might want to use '100.70.106.105:9092' or 'localhost:29092'
KAFKA_BROKER = os.getenv('KAFKA_BROKER', '100.70.106.105:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'class')
GROUP_ID = os.getenv('GROUP_ID', 'inference-group-2')

# ---------------------------------------------

def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def setup_model(device):
    print("⏳ 로컬 가중치 파일에서 모델 로딩 중...")
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"🚨 에러: 모델 가중치 파일({model_path})을 찾을 수 없습니다.")
        return None
    
    model = DualEncodingModel(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ AI 모델(DualEncodingModel) 로드 완료! ({device} 환경)")
    return model

def process_message(model, device, transform, payload):
    try:
        base_name = payload.get('base_name', 'unknown')
        
        # 1. 이미지 디코딩 및 전처리
        image_b64 = payload.get('image_base64')
        if not image_b64:
            print(f"⚠️ [{base_name}] 이미지가 누락되었습니다.")
            return
            
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        image_tensor = transform(image=image_np)['image']
        
        # 2. 엑셀(센서) 디코딩 및 전처리
        excel_b64 = payload.get('excel_base64')
        if not excel_b64:
            print(f"⚠️ [{base_name}] 센서(엑셀) 데이터가 누락되었습니다.")
            return
            
        excel_bytes = base64.b64decode(excel_b64)
        df = pd.read_excel(io.BytesIO(excel_bytes))
        
        # 문자열(시간 등) 데이터 제외하고 숫자형 데이터만 추출
        df_numeric = df.select_dtypes(include=[np.number])
        
        # 숫자 데이터만 5x5 로 캐스팅
        sensor_data = df_numeric.values[:5, :5].astype(np.float32)
        sensor_tensor = torch.FloatTensor(sensor_data)
        
        # 3. AI 모델 추론
        with torch.no_grad():
            img_batch = image_tensor.unsqueeze(0).to(device)
            sensor_batch = sensor_tensor.unsqueeze(0).to(device)
            
            outputs = model(img_batch, sensor_batch)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            
            pred_label = np.argmax(probs)
            # classes = ['Normal(정상)', 'Defect(불량)']
            result_str = 'Defect (불량) 🚨' if pred_label == 1 else 'Normal (정상) ✅'
            confidence = probs[pred_label] * 100
            
            print(f"\n[{base_name}] 추론 결과: {result_str} (확률: {confidence:.2f}%)")
            print(f"   세부 확률 -> 정상: {probs[0]*100:.2f}%, 불량: {probs[1]*100:.2f}%")
            
    except Exception as e:
        print(f"❌ 데이터 처리 중 오류 발생: {e}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_model(device)
    if model is None:
        return
        
    transform = get_transform()
    
    print(f"🔄 Kafka 브로커({KAFKA_BROKER}) 연결 대기 중...")
    
    # Kafka 브로커가 켜질 때까지 재시도
    consumer = None
    while consumer is None:
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                group_id=GROUP_ID,
                fetch_max_bytes=10485760, # 10MB
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest', # 실시간 추론이므로 latest
                enable_auto_commit=True
            )
        except Exception as e:
            print(f"⚠️ 연결 실패, 5초 후 재시도... ({e})")
            time.sleep(5)
            
    print("✅ 실시간 추론 Consumer 연결 완료! 데이터를 기다립니다. (종료: Ctrl+C)")
    
    try:
        for message in consumer:
            payload = message.value
            process_message(model, device, transform, payload)
            
    except KeyboardInterrupt:
        print("\n🛑 시스템 종료.")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
