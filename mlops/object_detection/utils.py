
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from typing import Any

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
CATEGORIES= [
    {
      "id": 1,
      "name": "정상_미납"
    },
    {
      "id": 2,
      "name": "정상_납부족"
    },
    {
      "id": 3,
      "name": "정상_납쇼트"
    },
    {
      "id": 4,
      "name": "정상_납볼"
    },
    {
      "id": 5,
      "name": "정상_납좌표밀림"
    },
    {
      "id": 6,
      "name": "정상_납형성불량"
    },
    {
      "id": 7,
      "name": "정상_냉납"
    },
    {
      "id": 8,
      "name": "정상_밀림"
    },
    {
      "id": 9,
      "name": "정상_쇼트"
    },
    {
      "id": 10,
      "name": "정상_오삽"
    },
    {
      "id": 11,
      "name": "정상_미삽"
    },
    {
      "id": 12,
      "name": "정상_역삽"
    },
    {
      "id": 13,
      "name": "정상_뒤집힘"
    },
    {
      "id": 14,
      "name": "정상_일어섬"
    },
    {
      "id": 15,
      "name": "정상_납금감/핀홀"
    },
    {
      "id": 16,
      "name": "정상_납고드름"
    },
    {
      "id": 17,
      "name": "불량_미납"
    },
    {
      "id": 18,
      "name": "불량_납부족"
    },
    {
      "id": 19,
      "name": "불량_납쇼트"
    },
    {
      "id": 20,
      "name": "불량_납볼"
    },
    {
      "id": 21,
      "name": "불량_납좌표밀림"
    },
    {
      "id": 22,
      "name": "불량_납형성불량"
    },
    {
      "id": 23,
      "name": "불량_냉납"
    },
    {
      "id": 24,
      "name": "불량_밀림"
    },
    {
      "id": 25,
      "name": "불량_쇼트"
    },
    {
      "id": 26,
      "name": "불량_오삽"
    },
    {
      "id": 27,
      "name": "불량_미삽"
    },
    {
      "id": 28,
      "name": "불량_역삽"
    },
    {
      "id": 29,
      "name": "불량_뒤집힘"
    },
    {
      "id": 30,
      "name": "불량_일어섬"
    },
    {
      "id": 31,
      "name": "불량_납금감/핀홀"
    },
    {
      "id": 32,
      "name": "불량_납고드름"
    }
  ]
DEF_CLS={
        "NM": 17,
        "NB": 18,
        "NS": 19,
        'OM': 20,
        'MF': 21,
        'CC': 22,
        'ML': 23,
        'SH': 24,
        'WS': 25,
        'MS': 26,
        'RS': 27,
        'FL': 28,
        'ST': 29,
        'PH': 30,
        'IC': 31,
        'PB': 32
}



def predict(model, file, basename) -> dict[str, Any]:
    # if model is None:
    #     raise HTTPException(status_code=503, detail="Model is not loaded")

    # if not file.content_type or not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="Only image files are supported")

    # suffix = Path(file.filename or "").suffix or ".jpg"

    # with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
    #     temp_path = Path(temp_file.name)
    #     temp_file.write(file.read())

    try:
        results = model.predict(
            source=file,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            save=False,
        )
    except Exception as e:
        print(e)

    detections = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index]+1)      ## 학습기준 카테고리 0베이스 -> 1번부터 시작하기 위해 설정
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": model.names[class_id-1],    ## model.names는 0베이스 -> 다시 -1
                    "confidence": float(boxes.conf[index]),
                    "bbox": [float(value) for value in boxes.xyxy[index].tolist()],
                }
            )
    return {"detections": detections}
FONT_PATH ="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"    ## dockerfile에 폰트 설치 명령어 추가 및 경로설정
# FONT_PATH = (r"C:\Windows\Fonts\malgun.ttf")
## 한국어 깨짐 문제를 해결하려면 font를 불러와야함 -> 우분투 환경에서는 어떡하지? -> dockerfile에서 나눔폰트 다운 받도록 설정!
def draw_bbox(image_file, result, output_path, filename ,color=(0, 255, 0), thickness=2):
    """
    모델 예측 결과를 통해 이미지에 bbox를 그리고 저장하는 함수
    
    Args:
        image_file (str): 원본 이미지 파일
        result (str): bbox 정보가 있는 딕셔너리 (모델 예측 결과 딕셔너리)
        color (tuple): BGR 색상 튜플 (default: 녹색)
        thickness (int): 선 두께
    Return: bbox친 이미지파일
    """

    # 이미지 로드
    image = image_file
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_file.shape}")
    
    # bbox 파일 로드
    data=result
    
    
    # 모든 bbox 그리기
    for d in data['detections']:
        # bbox 좌표 추출 [x, y, width, height]
        x, y, w, h = map(int, d['bbox'])
        x2, y2 = x + w, y + h
        
        # bbox 그리기
        cv2.rectangle(image, (x, y), (x2, y2), color, thickness)
        
        # 클래스 이름 표시 ----> 정상 카테고리는 모두 '정상'으로 표기
        if d['class_id']-1 < 16:                 ## 결과값 id가 1부터 시작 -> 인덱스를 위해 -1
            label='정상'
        else: label = CATEGORIES[d['class_id']-1]['name']
        if label:
            # 텍스트 크기 계산
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            
            # 텍스트 그리기
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image_rgb)
            draw= ImageDraw.Draw(img_pil)
            font=ImageFont.truetype(FONT_PATH, size=15)
            l, t, r ,b= draw.textbbox((x,y-5), label, font=font)
            text_width= r-l
            text_height= b-t
            pad=1
            draw.rectangle((x, y, x+text_width+pad, y+text_height+pad), fill=(0,255,0))
            draw.text((x,y), label, font=font)
            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # 이미지 저장
    # 경로에 폴더가 있는지 확인
    if os.path.exists(output_path): None
    else: os.mkdir(output_path)
    # 이미지 복원 후 저장
    image_result = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    save_result= cv2.imwrite(output_path+filename, image_result)
    print(f"bbox가 그려진 이미지가 저장되었습니다: {output_path}")
    return output_path+filename