import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. 처음에 바꿔서 쓰는 설정값
# ============================================================

# Kafka 서버 주소입니다.
# Producer가 보내는 서버 주소와 같아야 합니다.
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "100.70.106.105:9092")

# Kafka topic 이름입니다.
# Producer가 보내는 topic 이름과 같아야 합니다.
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "edge_data_topic")

# Consumer group id입니다.
# 같은 group id를 쓰면 Kafka가 "이미 읽은 메시지"를 기억합니다.
# 처음부터 다시 받고 싶으면 값을 바꿔서 실행하세요.
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "edge-consumer-group-ai-test")

# AI 모델 경로입니다.
# 다른 best.pt를 테스트하려면 MODEL_PATH 환경변수로 바꿀 수 있습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "5.models"
    / "detection_sd"
    / "results_train"
    / "20260623_101618"
    / "weights"
    / "best.pt"
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()

# 추론 기준값입니다.
# conf가 높을수록 더 확실한 것만 검출합니다.
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))


# ============================================================
# 2. 작은 도우미 함수들
# ============================================================

def base64_to_image(image_base64: str) -> Image.Image:
    """
    Kafka 메시지 안에 들어있는 base64 문자열을 PIL 이미지로 바꿉니다.

    Producer가 이미지를 파일 그대로 보낼 수 없기 때문에 보통 base64 문자열로 바꿔서 보냅니다.
    Consumer에서는 다시 이미지로 복원해야 AI 모델에 넣을 수 있습니다.
    """
    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def make_consumer() -> KafkaConsumer:
    """
    Kafka Consumer를 만듭니다.

    value_deserializer는 Kafka 메시지의 bytes 값을 Python dict로 바꿔주는 부분입니다.
    Producer가 JSON으로 보냈다는 가정입니다.
    """
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        group_id=GROUP_ID,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        fetch_max_bytes=10 * 1024 * 1024,
        max_partition_fetch_bytes=10 * 1024 * 1024,
    )


def predict_image(model: YOLO, image: Image.Image) -> list[dict[str, Any]]:
    """
    이미지 1장을 YOLO 모델로 추론하고, 보기 쉬운 dict 리스트로 바꿉니다.
    """
    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
        save=False,
    )

    detections: list[dict[str, Any]] = []
    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index])
        detections.append(
            {
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": round(float(boxes.conf[index]), 4),
                "bbox_xyxy": [round(float(value), 2) for value in boxes.xyxy[index].tolist()],
            }
        )

    return detections


def print_result(base_name: str, detections: list[dict[str, Any]]) -> None:
    """
    추론 결과를 터미널에서 보기 쉽게 출력합니다.
    """
    print(f"\n[수신 완료] image={base_name}")

    if not detections:
        print("  - 검출된 객체가 없습니다.")
        return

    print(f"  - 검출 개수: {len(detections)}")
    for number, detection in enumerate(detections, start=1):
        print(
            "  "
            f"{number}. class={detection['class_name']} "
            f"confidence={detection['confidence']} "
            f"bbox={detection['bbox_xyxy']}"
        )


# ============================================================
# 3. 메인 실행 함수
# ============================================================

def start_consumer() -> None:
    """
    Kafka 메시지를 실시간으로 받고, 메시지 안의 이미지를 바로 AI 모델로 추론합니다.

    예상 Kafka 메시지 예시는 아래와 같습니다.

    {
        "base_name": "sample_001",
        "category": "sd",
        "image_base64": "...base64 image string..."
    }
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    print("[준비] YOLO 모델을 로드합니다.")
    print(f"  - model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("[준비] Kafka Consumer를 시작합니다.")
    print(f"  - broker: {KAFKA_BROKER}")
    print(f"  - topic: {KAFKA_TOPIC}")
    print(f"  - group_id: {GROUP_ID}")
    print("  - 종료하려면 Ctrl+C를 누르세요.")

    consumer = make_consumer()

    try:
        for message in consumer:
            payload = message.value
            base_name = payload.get("base_name", f"message_offset_{message.offset}")
            image_base64 = payload.get("image_base64")

            if not image_base64:
                print(f"\n[건너뜀] image_base64가 없습니다. base_name={base_name}")
                continue

            try:
                image = base64_to_image(image_base64)
                detections = predict_image(model, image)
                print_result(base_name, detections)
            except Exception as error:
                print(f"\n[오류] 메시지 처리 실패: base_name={base_name}")
                print(f"  - reason: {error}")

    except KeyboardInterrupt:
        print("\n[종료] 사용자가 Consumer를 종료했습니다.")
    finally:
        consumer.close()


def run_local_image_test(image_path: Path) -> None:
    """
    Kafka 없이 로컬 이미지 1장으로 모델 추론을 테스트합니다.

    Kafka 연결 문제가 있을 때도 "모델이 정상 로드되는지", "추론 결과가 나오는지"를
    빠르게 확인할 수 있습니다.
    """
    image_path = image_path.expanduser().resolve()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    if not image_path.exists():
        raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {image_path}")

    print("[로컬 테스트] YOLO 모델을 로드합니다.")
    print(f"  - model: {MODEL_PATH}")
    print(f"  - image: {image_path}")

    model = YOLO(str(MODEL_PATH))
    image = Image.open(image_path).convert("RGB")
    detections = predict_image(model, image)
    print_result(image_path.name, detections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kafka로 이미지를 받아 YOLO best.pt 모델로 실시간 추론하는 테스트 코드"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Kafka 없이 로컬 이미지 1장으로 추론 테스트를 실행합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.image:
        run_local_image_test(args.image)
    else:
        start_consumer()
