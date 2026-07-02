# shap 토픽에서 수신된 base64 Excel을 복호화하고 XGBoost + SHAP 분석을 실시간으로 수행
#
# 실행:
#   cd models/4.cause
#   python kafka_manager_shap.py
#
# 사전 조건:
#   train.py 를 먼저 실행해 result_cause/models/ 와 result_cause/scalers/ 가 생성되어 있어야 함

import io
import base64
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import shap
import xgboost as xgb
from kafka import KafkaConsumer, KafkaProducer
from pymongo import MongoClient
# analyze.py 의 함수·상수 재사용
from analyze import (
    PROCESS_DEFECTS,
    SENSOR_KEYS,
    RESULT_DIR,
    normalize_sensor,
)

KAFKA_BROKER = "100.70.106.105:9092"
KAFKA_TOPIC  = "shap"                   ## 수신 토픽
KAFKA_TOPIC_RESULT = 'shap_result'     ## 송신 토픽
GROUP_ID     = "smt-backend-shapmodel"

MODELS_DIR  = RESULT_DIR / "models"
SCALERS_DIR = RESULT_DIR / "scalers"

# Excel 컬럼명 → 내부 센서 키 매핑
COLUMN_MAP = {
    "temperature(°c)": "temperature",
    "humidity(%)":     "humidity",
    "vibration(m/s²)": "vibration",
    "acceleration(g)": "acceleration",
    "noise(db)":       "noise",
}
# MongoDB 설정
MONGO_URL = "mongodb://woo:young@100.70.106.105:27017"
mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db = mongo_client["ai_factory_db"]
results_collection = db["obb_results"]
sensor_collection= db["sensor_data"]

def load_models():
    """서버 시작 시 모델·스케일러를 메모리에 한 번만 로드"""
    if not MODELS_DIR.exists() or not SCALERS_DIR.exists():
        print("[오류] result_cause/models 또는 scalers 폴더가 없습니다. train.py 를 먼저 실행하세요.")
        return None, None

    models  = {}
    scalers = {}

    for process in PROCESS_DEFECTS:
        scaler_path = SCALERS_DIR / f"{process}_scaler.pkl"
        if scaler_path.exists():
            scalers[process] = joblib.load(scaler_path)

        for defect_name, _ in PROCESS_DEFECTS[process]:
            safe_name  = defect_name.replace("/", "_")
            model_path = MODELS_DIR / f"{process}_{safe_name}.json"
            if model_path.exists():
                clf = xgb.XGBClassifier()
                clf.load_model(str(model_path))
                models[f"{process}_{defect_name}"] = clf

    print(f"[모델 로드] {len(models)}개 모델 / {len(scalers)}개 스케일러 로드 완료")
    return models, scalers


def parse_excel(sensor_b64):
    """base64 Excel → 센서 피처 10개 (mean/max × 5센서) 추출"""
    excel_bytes = base64.b64decode(sensor_b64)
    df = pd.read_excel(io.BytesIO(excel_bytes))

    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    features = []
    for key in SENSOR_KEYS:
        if key in df.columns:
            arr = df[key].dropna().values.astype(np.float64)
            features.append(float(arr.mean()))
            features.append(float(arr.max()))
        else:
            features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def find_defect_name(process, category_id):
    """category_id 로 해당 공정의 불량명을 찾아 반환, 없으면 None"""
    for defect_name, cid in PROCESS_DEFECTS.get(process, []):
        if cid == category_id:
            return defect_name
    return None


def run_shap(models, scalers, process, features, category_id):
    """category_id 에 해당하는 불량 하나에 대해서만 SHAP 분석 수행"""
    if process not in scalers:
        print(f"[경고] {process} 스케일러 없음")
        return {}

    defect_name = find_defect_name(process, category_id)
    if defect_name is None:
        print(f"[경고] category_id={category_id} 에 해당하는 불량 없음")
        return {}

    key = f"{process}_{defect_name}"
    clf = models.get(key)
    if clf is None:
        print(f"[경고] 모델 없음: {key}")
        return {}

    X         = scalers[process].transform(features.reshape(1, -1))
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)  # shape: (1, 10)

    importance = normalize_sensor(np.abs(shap_vals[0]))
    if not importance:
        return {}

    critical = next(iter(importance))
    return {
        "defect_name":       defect_name,
        "category_id":       category_id,
        "critical_sensor":   critical,
        "sensor_importance": importance,
    }


def main():
    models, scalers = load_models()
    if models is None:
        return

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        group_id=GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        fetch_max_bytes=10 * 1024 * 1024,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            # JSON 형태로 자동 직렬화 (Base64가 포함되므로 크기가 커질 수 있음)
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            max_request_size=10 * 1024 * 1024,  # 10MB
            request_timeout_ms=10000,
            max_block_ms=10000
        )

    print(f"[SHAP Consumer] '{KAFKA_TOPIC}' 토픽 구독 시작 (Ctrl+C 로 종료)")
    print("=" * 60)

    try:
        for msg in consumer:
            payload    = msg.value
            file_name  = payload.get("file_name", "")
            category= payload.get("category", "")
            sensor_b64 = payload.get("sensor_file")
            category_id = payload.get("category_id", "")
            confidence= payload.get("confidence", "")
            excel_file= payload.get("excel_file", "")
            cause= payload.get("cause", "")

            print(f"\n[수신] file_name={file_name} | category_id={category_id}| sensor_file 길이={len(sensor_b64) if sensor_b64 else 0}")

            if not sensor_b64:
                print(f"[스킵] sensor_file 없음: {file_name}")
                continue

            # file_name 앞 두 글자로 공정 구분 (PR_DEF_... / SD_DEF_...)
            process = file_name.split("_")[0].upper()
            if process not in PROCESS_DEFECTS:
                print(f"[스킵] 공정 식별 불가: {file_name}")
                continue

            try:
                features = parse_excel(sensor_b64)
            except Exception as e:
                print(f"[오류] Excel 파싱 실패 ({file_name}): {e}")
                continue

            shap_result = run_shap(models, scalers, process, features, int(category_id))

            result = {
                "category": category,
                "cause": cause,
                "confidence": confidence,
                "file_name":  file_name,
                "timestamp":  datetime.now().isoformat(),
                "shap_result": shap_result,
                'excel_file': excel_file
            }
            db_payload= {
                "file_name":  file_name,
                "shap_result": shap_result,
                'sensor_data': sensor_b64
            }
            print(f"\n[분석 완료] {file_name}")
            # print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=" * 60)
            try:  
                print(f"토픽 '{KAFKA_TOPIC_RESULT}'으로 메시지 전송 시도...")
                future = producer.send(KAFKA_TOPIC_RESULT, value=result)
                # 전송 대기 (동기식)
                record_metadata = future.get(timeout=10)
                print(f"전송 성공! (파티션: {record_metadata.partition}, 오프셋: {record_metadata.offset})")
            except Exception as e:
                print(f"전송 실패: {e}")
                # DB 저장 시도
            try:
                insert_result = sensor_collection.insert_one(db_payload)
                print(f"✅ DB 저장 완료! (ID: {insert_result.inserted_id})")
            except Exception as e:
                print(f"❌ DB 저장 실패: {e}")

    except KeyboardInterrupt:
        print("\n[SHAP Consumer] 종료")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
