import cv2
import json
import random
from pathlib import Path

def draw_bbox(image_path, json_path, output_path, color=(0, 255, 0), thickness=2):
    """
    이미지에 정답(GT) COCO JSON의 bbox를 그리고 지정된 출력 경로에 저장하는 함수
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # JSON 파일 로드
    with open(str(json_path), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 카테고리 정보 매핑 (id -> name)
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    # 모든 bbox 그리기
    for ann in data.get('annotations', []):
        x, y, w, h = map(int, ann['bbox'])
        x2, y2 = x + w, y + h
        
        cv2.rectangle(image, (x, y), (x2, y2), color, thickness)
        
        label = categories.get(ann['category_id'], f"Class_{ann['category_id']}")
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            text_y = max(y, text_height + 10)
            
            cv2.rectangle(image, (x, text_y - text_height - 10), (x + text_width, text_y), color, -1)
            cv2.putText(image, label, (x, text_y - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"시각화 완료 (산출물 저장소): {output_path}")


def test_bbox_drawing():
    """
    1.base_data/val/sd 내부에서 정상 3개, 불량 3개 샘플을 무작위 탐색하여 
    시각화 결과물을 dataset/5.models/detection_sd/utils_test_visual 폴더로 격리 저장하는 함수
    """
    # 프로젝트 루트 절대 경로 역추적
    project_root = Path(__file__).resolve().parents[2]
    
    # 원천 데이터 소스 경로
    image_base = project_root / 'dataset' / '1.base_data' / 'images' / 'train' / 'sd'
    json_base = project_root / 'dataset' / '1.base_data' / 'labels_json' / 'train' / 'sd'
    
    # 산출물 격리 저장소
    output_base = project_root / 'dataset' / '5.models' / 'detection_sd' / 'utils_test_visual'
    
    print(f"=== SD 공정 바운딩 박스 시각화 검증 테스트 엔진 가동 ===")
    print(f"원천 데이터 소스: {image_base}")
    print(f"산출물 격리 저장소: {output_base}\n")
    
    if not json_base.exists() or not image_base.exists():
        print("에러: 데이터 원천 경로가 존재하지 않습니다. 폴더 구조를 재확인하세요.")
        return

    all_json_files = list(json_base.glob('*.json'))
    if not all_json_files:
        print("안내: val/sd 폴더 내에 검증할 JSON 라벨 파일이 비어 있습니다.")
        return
        
    # 고정 포인트: SD 공정 파일명 규칙 프리픽스(SD_DEF_ / SD_NOR_)를 기준으로 그룹 분리
    defect_jsons = [j for j in all_json_files if j.name.startswith("SD_DEF")]
    normal_jsons = [j for j in all_json_files if j.name.startswith("SD_NOR")]
    
    test_targets = []
    
    # 불량 샘플 최대 3개 추출
    defect_size = min(3, len(defect_jsons))
    if defect_size > 0:
        test_targets.extend(random.sample(defect_jsons, defect_size))
        print(f"- 불량(Defect) 샘플 {defect_size}개 추출 완료")
    else:
        print("- 경고: 불량(SD_DEF_) 라벨 파일이 폴더에 존재하지 않습니다.")
        
    # 정상 샘플 최대 3개 추출
    normal_size = min(3, len(normal_jsons))
    if normal_size > 0:
        test_targets.extend(random.sample(normal_jsons, normal_size))
        print(f"- 정상(Normal) 샘플 {normal_size}개 추출 완료")
    else:
        print("- 경고: 정상(SD_NOR_) 라벨 파일이 폴더에 존재하지 않습니다.")
        
    print(f"\n총 {len(test_targets)}개의 샘플에 대해 시각화를 진행합니다.\n")
        
    for json_path in test_targets:
        img_name = json_path.with_suffix('.JPG').name
        img_path = image_base / img_name
        if not img_path.exists():
            img_name = json_path.with_suffix('.jpg').name
            img_path = image_base / img_name
            
        if not img_path.exists():
            print(f"경고: 라벨과 쌍을 이룰 원본 이미지를 찾지 못했습니다 -> {img_name}")
            continue
            
        dst_output_path = output_base / f"{img_path.stem}_bbox{img_path.suffix}"
            
        try:
            print(f"[TEST RUN] 시각화 가동 -> 소스 라벨: {json_path.name}")
            draw_bbox(image_path=img_path, json_path=json_path, output_path=dst_output_path)
        except Exception as e:
            print(f"Error 처리 중 문제 발생 ({img_name}): {str(e)}")

if __name__ == "__main__":
    test_bbox_drawing()