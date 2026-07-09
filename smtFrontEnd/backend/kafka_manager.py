# Kafka에서 SMT 공정 데이터를 수신하고, 가벼운 데이터는 메모리 버퍼에 저장해 SSE로 전달하는 백그라운드 consumer 모듈
import io
import os
import json
import base64
import asyncio
import pprint
from collections import deque
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaConsumer

load_dotenv()

# Kafka 브로커(메시지 서버)의 IP 주소와 포트 — .env의 KAFKA_BROKER 또는 기본값 사용
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "100.70.106.105:9092")

# 엣지 디바이스가 이미지+센서 데이터를 발행하는 토픽(채널) 이름
KAFKA_TOPIC  = "shap_result"

# 여러 consumer가 같은 토픽을 나눠서 읽을 때 구분하는 그룹 ID
# 동일 그룹 내 consumer끼리는 메시지를 중복 수신하지 않음
GROUP_ID     = "smt-backend-realtime"

# 최근 수신된 가벼운 데이터를 최대 100건 메모리에 유지
# deque(maxlen=100) : 101번째 항목이 들어오면 가장 오래된 항목이 자동으로 밀려남
realtime_buffer: deque = deque(maxlen=100)

# SSE 구독자 목록 (연결된 클라이언트마다 asyncio.Queue 하나씩)
# 클라이언트가 /api/realtime/stream 에 접속하면 Queue가 한 개 추가됨
subscribers: list[asyncio.Queue] = []


def parse_excel_sensor(excel_b64: str) -> list:
    """excel_file(base64)을 디코딩해 센서 데이터 행 목록으로 변환"""
    try:
        excel_bytes = base64.b64decode(excel_b64)
        df = pd.read_excel(io.BytesIO(excel_bytes))
        # 컬럼명을 소문자로 통일 후 센서 관련 컬럼만 추출
        df.columns = [c.lower().strip() for c in df.columns]
        ts_cols = [c for c in df.columns if "timestamp" in c or "time" in c]
        sensor_cols = [c for c in df.columns if any(k in c for k in ["temp", "humid", "vibr", "accel", "noise", "온도", "습도", "진동", "가속", "소음"])]
        keep_cols = ts_cols + sensor_cols
        if keep_cols:
            return df[keep_cols].dropna().to_dict(orient="records")
        # 센서 컬럼 못 찾으면 전체 반환
        return df.dropna(how="all").to_dict(orient="records")
    except Exception as e:
        print(f"[Kafka] 엑셀 파싱 오류: {e}")
        return []


def parse_lightweight(payload: dict) -> dict:
    """
    payload에서 용량이 작은 데이터만 추출해 메모리 버퍼에 저장할 dict 반환.
    excel_file은 파싱해서 sensor_data로 변환 후 포함, 이미지 관련 대용량 필드는 제외.
    """
    HEAVY_KEYS = {"image_base64", "excel_base64", "excel_file"}

    result = {k: v for k, v in payload.items() if k not in HEAVY_KEYS}

    # excel_file → sensor_data 변환
    excel_b64 = payload.get("excel_file")
    result["sensor_data"] = parse_excel_sensor(excel_b64) if excel_b64 else []

    return result

async def _broadcast(data: dict):
    """수신된 데이터를 SSE 구독자 전체에게 전달"""
    # Queue가 꽉 차서 더 이상 넣을 수 없는 클라이언트(느린 네트워크 등)를 제거하기 위한 목록
    dead = []
    for q in subscribers:
        try:
            # 비동기 대기 없이 즉시 Queue에 데이터를 넣음
            # Queue가 가득 찬 경우 QueueFull 예외가 발생
            q.put_nowait(data)
        except asyncio.QueueFull:
            # 데이터를 받지 못한 구독자는 나중에 일괄 제거
            dead.append(q)
    for q in dead:
        # 응답이 밀린 클라이언트의 Queue를 구독 목록에서 제거해 메모리 누수 방지
        subscribers.remove(q)


async def kafka_consumer_task():
    """FastAPI lifespan에서 백그라운드 태스크로 실행되는 Kafka consumer"""
    # 현재 실행 중인 이벤트 루프를 가져옴 — run_in_executor 호출 시 필요
    loop = asyncio.get_event_loop()

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        group_id=GROUP_ID,
        # 이미지를 Base64로 인코딩하면 원본보다 약 33% 커지므로, 한 번에 받을 수 있는
        # 최대 바이트를 10 MB로 넉넉하게 설정
        fetch_max_bytes=10 * 1024 * 1024,
        # JSON 바이트 문자열을 자동으로 dict로 역직렬화
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        # 서버가 재시작해도 이미 처리한 과거 메시지를 다시 받지 않도록 최신 오프셋부터 읽음
        auto_offset_reset="latest",
        # 처리 완료 오프셋을 Kafka에 자동으로 커밋해 재시작 시 중복 수신을 최소화
        enable_auto_commit=True,
    )

    print(f"[Kafka] '{KAFKA_TOPIC}' 구독 시작")

    try:
        while True:
            # KafkaConsumer.poll()은 동기 블로킹 함수여서 그대로 호출하면 이벤트 루프가 멈춤
            # run_in_executor로 별도 스레드에서 실행해 비동기 루프를 블로킹하지 않도록 함
            records = await loop.run_in_executor(
                None, lambda: consumer.poll(timeout_ms=1000)
            )
            # records는 {TopicPartition: [메시지 목록]} 형태의 딕셔너리
            for _, messages in records.items():
                for msg in messages:
                    try:
                        # 수신된 원본 payload 키 목록과 값 출력
                        print("\n===== [Kafka] 수신 데이터 =====")
                        for k, v in msg.value.items():
                            if k in ("image_base64", "excel_base64"):
                                print(f"  {k}: [Base64 데이터 {len(v)}자]")
                            elif k == "metadata":
                                print(f"  metadata:")
                                pprint.pprint(v, indent=4)
                            else:
                                print(f"  {k}: {v}")
                        print("================================\n")

                        # 가벼운 데이터만 추출해 메모리 버퍼에 저장 (file_name 기준 중복 제외)
                        lightweight = parse_lightweight(msg.value)
                        file_name = lightweight.get("file_name") or lightweight.get("base_name")
                        if any(b.get("file_name") == file_name for b in realtime_buffer):
                            continue

                        realtime_buffer.appendleft(lightweight)

                        # 현재 연결된 모든 SSE 클라이언트에게 실시간 전달
                        await _broadcast(lightweight)

                        print(f"[Kafka] 수신: {lightweight.get('category','?')} | {file_name}")
                    except Exception as e:
                        # 개별 메시지 처리 실패 시 전체 루프를 멈추지 않고 오류만 출력 후 계속 진행
                        print(f"[Kafka] 처리 오류: {e}")
    except asyncio.CancelledError:
        # FastAPI 서버 종료 시 task.cancel()이 호출되면 이 예외가 발생
        print("[Kafka] consumer 종료")
    finally:
        # 예외 발생 여부와 관계없이 항상 Kafka 연결을 닫아 리소스를 반환
        consumer.close()
