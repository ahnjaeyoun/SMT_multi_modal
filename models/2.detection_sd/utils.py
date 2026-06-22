import cv2
import json
import random
from pathlib import Path

def draw_bbox(image_path, json_path, output_path=None, color=(0, 255, 0), thickness=2):
    """
    이미지에 정답(GT) COCO JSON의 bbox를 그리고 저장하는 함수
    
    Args:
        image_path (str/Path): 원본 이미지 경로
        json_path (str/Path): bbox 정보가 있는 JSON 파일 경로
        output_path (str/Path, optional): 저장할 경로. None이면 파일명 뒤에 '_bbox'를 붙여서 저장
        color (tuple): BGR 색상 튜플 (default: 녹색)
        thickness (int): 선 두께
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
        # COCO 포맷: [x_min, y_min, width, height]
        x, y, w, h = map(int, ann['bbox'])
        x2, y2 = x + w, y + h
        
        # bbox 사각형 그리기
        cv2.rectangle(image, (x, y), (x2, y2), color, thickness)
        
        # 클래스 이름 레이블링
        label = categories.get(ann['category_id'], f"Class_{ann['category_id']}")
        if label:
            # OpenCV 텍스트 크기 계산 (가독성을 위한 배경 사각형용)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # 텍스트가 이미지 위쪽 경계를 벗어나지 않도록 보정
            text_y = max(y, text_height + 10)
            
            # 텍스트 배경 사각형 그리기
            cv2.rectangle(image, (x, text_y - text_height - 10), (x + text_width, text_y), color, -1)
            
            # 텍스트 쓰기 (배경 위에 검은색 글씨)
            cv2.putText(image, label, (x, text_y - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    # 저장 경로가 설정되지 않았다면 원본 위치에 후미 기입 방식으로 세팅
    if output_path is None:
        image_path = Path(image_path)
        output_path = image_path.parent / f"{image_path.stem}_bbox{image_path.suffix}"
    
    # 이미지 파일 영구 저장
    cv2.imwrite(str(output_path), image)
    print(f"🎯 바운딩 박스 시각화 이미지가 성공적으로 저장되었습니다 -> {output_path}")


def test_bbox_drawing():
    """
    프로젝트 아키텍처(1.base_data/val/pr) 내부에서 
    샘플 정상 및 불량 데이터를 자동 탐색하여 bbox 그리기 테스트를 수행하는 함수
    """
    # 📍 프로젝트 루트(C:\SMT_multi_modal) 절대 경로 역추적
    project_root = Path(__file__).resolve().parents[2]
    
    # 검증할 데이터 레이어 소스 경로 연동
    image_base = project_root / 'dataset' / '1.base_data' / 'images' / 'val' / 'pr'
    json_base = project_root / 'dataset' / '1.base_data' / 'labels_json' / 'val' / 'pr'
    
    print(f"=== 바운딩 박스 시각화 검증 테스트 엔진 가동 ===")
    print(f"검색 타겟 이미지 저장소: {image_base}")
    print(f"검색 타겟 라벨 JSON 저장소: {json_base}\n")
    
    if not json_base.exists() or not image_base.exists():
        print("에러: 데이터 원천 경로가 존재하지 않습니다. 전처리를 먼저 수행했는지 확인하세요.")
        return

    # JSON 라벨 명단 스캔
    all_json_files = list(json_base.glob('*.json'))
    
    if not all_json_files:
        print("안내: val/pr 폴더 내에 검증할 JSON 라벨 파일이 비어 있습니다.")
        return
        
    # 파일명 프리픽스(PR_DEF_ / PR_NOR_)를 기준으로 정상과 불량 샘플 분리
    defect_jsons = [j for j in all_json_files if j.name.startswith("PR_DEF_")]
    normal_jsons = [j for j in all_json_files if j.name.startswith("PR_NOR_")]
    
    # 테스트할 타겟 샘플 확정 (무작위로 각각 1개씩 추출, 없으면 전체 리스트에서 차용)
    test_targets = []
    if defect_jsons:
        test_targets.append(random.choice(defect_jsons))
    if normal_jsons:
        test_targets.append(random.choice(normal_jsons))
    if not test_targets:
        test_targets = all_json_files[:2] # 예외 케이스 처리용
        
    # 테스트 드로잉 루프 가동
    for json_path in test_targets:
        # JSON 명칭에 매핑되는 원천 이미지 파일 매칭 (대소문자 예외 처리)
        img_name = json_path.with_suffix('.JPG').name
        img_path = image_base / img_name
        if not img_path.exists():
            img_name = json_path.with_suffix('.jpg').name
            img_path = image_base / img_name
            
        if not img_path.exists():
            print(f"경고: 라벨과 쌍을 이룰 원본 이미지를 찾지 못했습니다 -> {img_name}")
            continue
            
        try:
            print(f"[TEST RUN] 시각화 대상 라벨: {json_path.name}")
            draw_bbox(image_path=img_path, json_path=json_path)
        except Exception as e:
            print(f"Error 처리 중 문제 발생 ({img_name}): {str(e)}")

if __name__ == "__main__":
    test_bbox_drawing()