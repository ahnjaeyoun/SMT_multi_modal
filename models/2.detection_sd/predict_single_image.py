import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# 정상 bbox는 초록색, 불량 bbox는 빨간색.
# OpenCV 색상은 BGR 순서.
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# 현재 프로젝트의 최상위 폴더 가져오기
def get_project_root():
    """현재 프로젝트의 최상위 폴더를 찾습니다."""
    if os.getenv("PROJECT_ROOT"):
        return Path(os.getenv("PROJECT_ROOT")).resolve()
    return Path(__file__).resolve().parents[2]


def get_latest_best_pt(project_root):
    """가장 최근 학습 결과 폴더에서 best.pt를 자동으로 찾습니다."""
    train_root = project_root / "dataset" / "5.models" / "detection_sd" / "results_train"
    best_files = sorted(train_root.glob("20*/weights/best.pt"))

    if not best_files:
        raise FileNotFoundError(f"best.pt를 찾을 수 없습니다: {train_root}")

    return best_files[-1]


def clean_input_path(text, project_root):
    """입력받은 경로를 실제 파일 경로로 바꿉니다."""
    path = Path(text.strip().strip('"').strip("'"))

    if path.is_absolute():
        return path.resolve()

    # 현재 터미널 위치 기준 경로가 있으면 먼저 사용합니다.
    if path.exists():
        return path.resolve()

    # 터미널 위치가 달라도 dataset/... 같은 프로젝트 기준 경로를 사용할 수 있게 합니다.
    return (project_root / path).resolve()


def get_color(class_id, is_json_label=False):
    """
    정상/불량 class 기준으로 bbox 색을 정합니다.

    YOLO txt와 모델 예측 class_id: 0~15 정상, 16~31 불량
    COCO json category_id: 1~16 정상, 17~32 불량
    """
    defect_start = 17 if is_json_label else 16
    return RED if class_id >= defect_start else GREEN


def bgr_to_rgb(color):
    """OpenCV용 BGR 색상을 PIL용 RGB 색상으로 바꿉니다."""
    blue, green, red = color
    return red, green, blue


def get_korean_font(size=20):
    """bbox 글자에 사용할 한글 폰트를 불러옵니다."""
    font_path = Path("C:/Windows/Fonts/malgun.ttf")

    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)

    return ImageFont.load_default()


def draw_bbox(image, bbox, text, color):
    """이미지 위에 bbox와 한글 글자를 그립니다."""
    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # cv2.putText는 한글이 깨질 수 있어서 글자는 PIL로 그립니다.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = get_korean_font()

    text_x = x1
    text_y = max(y1 - 26, 0)
    draw.text((text_x, text_y), text, font=font, fill=bgr_to_rgb(color))

    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def load_txt_labels(label_path, image_width, image_height):
    """
    YOLO txt 라벨을 bbox 목록으로 바꿉니다.
    txt 형식: class_id center_x center_y width height
    좌표는 0~1 사이의 비율 값입니다.
    """
    bboxes = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            center_x, center_y, box_w, box_h = map(float, parts[1:5])

            x1 = (center_x - box_w / 2) * image_width
            y1 = (center_y - box_h / 2) * image_height
            x2 = (center_x + box_w / 2) * image_width
            y2 = (center_y + box_h / 2) * image_height

            bboxes.append(
                {
                    "class_id": class_id,
                    "name": f"class_{class_id}",
                    "bbox": [x1, y1, x2, y2],
                    "color": get_color(class_id),
                }
            )

    return bboxes


def load_json_labels(label_path):
    """
    COCO json 라벨을 bbox 목록으로 바꿉니다.
    json bbox 형식: x, y, width, height
    """
    with open(label_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    categories = {}
    for category in data.get("categories", []):
        categories[int(category["id"])] = category.get("name", f"class_{category['id']}")

    bboxes = []
    for annotation in data.get("annotations", []):
        class_id = int(annotation["category_id"])
        x, y, w, h = map(float, annotation["bbox"])

        bboxes.append(
            {
                "class_id": class_id,
                "name": categories.get(class_id, f"class_{class_id}"),
                "bbox": [x, y, x + w, y + h],
                "color": get_color(class_id, is_json_label=True),
                "is_json_label": True,
            }
        )

    return bboxes


def load_label_bboxes(label_path, image_width, image_height):
    """라벨 파일 확장자에 따라 txt 또는 json 라벨을 읽습니다."""
    if label_path.suffix.lower() == ".txt":
        return load_txt_labels(label_path, image_width, image_height)
    if label_path.suffix.lower() == ".json":
        return load_json_labels(label_path)

    raise ValueError("라벨 파일은 .txt 또는 .json만 사용할 수 있습니다.")


def save_label_bbox_image(original_image, label_bboxes, output_path, class_names):
    """라벨 데이터 기준 bbox 이미지를 저장합니다."""
    image = original_image.copy()

    for item in label_bboxes:
        # YOLO txt는 class_id가 0부터, COCO json은 category_id가 1부터 시작합니다.
        model_class_id = item["class_id"] - 1 if item.get("is_json_label") else item["class_id"]
        if model_class_id in class_names:
            item["name"] = class_names[model_class_id]
        draw_bbox(image, item["bbox"], item["name"], item["color"])

    cv2.imwrite(str(output_path), image)


def save_prediction_bbox_image(original_image, image_path, model, output_path):
    """best.pt 예측 결과 기준 bbox 이미지를 저장합니다."""
    image = original_image.copy()
    results = model.predict(source=str(image_path), conf=0.25, iou=0.45, save=False, verbose=False)

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        for index in range(len(boxes)):
            class_id = int(boxes.cls[index])
            confidence = float(boxes.conf[index])
            class_name = model.names[class_id]
            bbox = boxes.xyxy[index].tolist()
            color = get_color(class_id)
            text = f"{class_name} {confidence:.2f}"

            draw_bbox(image, bbox, text, color)

    cv2.imwrite(str(output_path), image)


def main():
    project_root = get_project_root()
    model_path = get_latest_best_pt(project_root)
    model = YOLO(str(model_path))

    print("이미지 경로와 라벨 경로를 입력하면 bbox 이미지 2장을 만듭니다.")
    print("best.pt는 가장 최근 학습 결과에서 자동으로 불러옵니다.")
    print()

    image_path = clean_input_path(input("이미지 경로: "), project_root)
    label_path = clean_input_path(input("라벨 경로(.txt 또는 .json): "), project_root)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"라벨 파일이 없습니다: {label_path}")

    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    height, width = original_image.shape[:2]
    label_bboxes = load_label_bboxes(label_path, width, height)

    output_dir = project_root / "dataset" / "5.models" / "detection_sd" / "results_predict_bbox"
    output_dir.mkdir(parents=True, exist_ok=True)

    label_output_path = output_dir / f"{image_path.stem}_label_bbox{image_path.suffix}"
    pred_output_path = output_dir / f"{image_path.stem}_bestpt_bbox{image_path.suffix}"

    save_label_bbox_image(original_image, label_bboxes, label_output_path, model.names)
    save_prediction_bbox_image(original_image, image_path, model, pred_output_path)

    print()
    print("완료")
    print(f"사용한 모델: {model_path}")
    print(f"라벨 bbox 이미지: {label_output_path}")
    print(f"best.pt bbox 이미지: {pred_output_path}")


if __name__ == "__main__":
    main()
