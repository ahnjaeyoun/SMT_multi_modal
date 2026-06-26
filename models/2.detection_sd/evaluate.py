import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# 1. 시스템 환경 설정 및 초기화 레이어
# ==========================================

# 한글 깨짐 현상 방지
def setup_korean_font():
    """
    matplotlib 그래프 시각화 리포트 생성 시 한글 깨짐 현상을 방지하는 시스템 함수
    """
    try:
        font_list = [font.name for font in fm.fontManager.ttflist]
        # 운영체제별(Windows, Mac, Linux OS) 대표적인 한글 폰트 후보군 정의
        korean_fonts = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR', 'DejaVu Sans']
        for font in korean_fonts:
            if font in font_list:
                plt.rcParams['font.family'] = font          # 매칭되는 첫 번째 폰트를 전역 적용
                plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 예외 처리
                return
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    except Exception as e:
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 전역 폰트 로드 수행
setup_korean_font()

# 정제된 텍스트 출력을 위해 불필요한 서드파티 라이브러리(폰트 누락, Ultralytics 내장 안내) 경고 숨김 처리
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')

def get_project_root():
    env_root = os.getenv('PROJECT_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


# ==========================================
# 2. 파일 I/O 및 데이터 유틸리티 함수 레이어
# ==========================================

def get_results_dir(base_path):
    """
    평가 프로세스가 실행될 때마다 고유의 실험 통계를 유실 없이 격리 보관하기 위해 
    타임스탬프화된 폴더(eval_YYYYMMDD_HHMMSS)를 생성하는 함수
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_path) / f"eval_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True) # 하위 경로를 포함하여 폴더 자동 빌드
    return results_dir


def load_ground_truth_annotations(annotations_csv_path):
    """
    전처리기(preprocess.py) 단계에서 정합 가공된 원천 정답 파일(val_annotations.csv)을 읽어와
    수동 계산기(IoU 매칭 단락)가 조율할 수 있도록 해시 맵 데이터로 역직렬화하는 함수
    
    Args:
        annotations_csv_path: 검증 정답 데이터프레임 CSV 파일 경로
    Returns:
        dict: { 이미지명: [{'class_name': 클래스명, 'bbox': [x1, y1, x2, y2]}, ...] }
    """
    gt_annotations = defaultdict(list) # 키가 없으면 자동으로 빈 리스트 생성
    try:
        df = pd.read_csv(annotations_csv_path, encoding='utf-8')
        for _, row in df.iterrows():
            image_name = Path(row['image_path']).name # 절대 경로에서 파일 명칭만 추출
            gt_box = {
                'class_name': row['class_name'],
                # COCO 좌상단 및 우하단 픽셀 절대 좌표계 파싱
                'bbox': [float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])]
            }
            gt_annotations[image_name].append(gt_box) # 이미지별 객체 박스 축적
        print(f"GT 어노테이션 로드 완료: {len(gt_annotations)}개 이미지")
    except Exception as e:
        print(f"GT 로드 실패: {e}")
    return dict(gt_annotations)


def get_validation_images(dataset_yaml_path):
    """
    YOLO 설정 파일(dataset.yaml)을 동적으로 파싱하여 
    실제 로컬 드라이브에 복사된 검증 세트(val/images) 내부의 수치 연단용 이미지 목록을 수집하는 함수
    """
    dataset_yaml_path = Path(dataset_yaml_path)
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    val_path = dataset_config.get('val')        # YAML 내에 기입된 'val/images' 상대 패스 파싱
    dataset_root = dataset_config.get('path', '') # 데이터셋 최상위 마운트 절대 경로 파싱
    dataset_root = Path(dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml_path.parent / dataset_root).resolve()
    full_val_path = dataset_root / val_path
    
    # 확장자 예외 처리용 해시 셋 정의
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    val_images = []
    
    if full_val_path.exists():
        for ext in image_extensions:
            # 해당 확장자 패턴과 매칭되는 가용한 모든 파일 스캔
            val_images.extend(list(full_val_path.glob(f"*{ext}")))
    return sorted(val_images)


# ==========================================
# 3. 모델 추론 및 연산 핵심 인프라 레이어
# ==========================================

def predict_on_images(model, image_paths, conf_thres=0.25, iou_thres=0.45):
    """
    Ultralytics YOLO 자체 내장 평가 외에, 이미지 단위별 수동 IoU 정밀 오탐 매칭을 수행하기 위해
    개별 검증 이미지의 로형 픽셀 데이터 정보를 바탕으로 1차 예측 바운딩 박스를 추출하는 함수
    
    Args:
        model: YOLOv11 가중치 모델 객체
        image_paths: 순회할 이미지 파일 Path 리스트
    Returns:
        list: 각 이미지별 1차 원시 모델 예측 스코어 셋
    """
    all_results = []
    # tqdm 연동을 통해 대용량 SMT 이미지 스캔 스리풋 실시간 시각화 진행
    for img_path in tqdm(image_paths, desc="SD 이미지 추론 검증 중", unit="img"):
        # model.predict 엔진 구동 (테스트 환경이므로 파일 영구 저장은 False로 리소스 아낌)
        results = model.predict(source=str(img_path), conf=conf_thres, iou=iou_thres, verbose=False, save=False)
        image_result = {'image_path': str(img_path), 'image_name': img_path.name, 'predictions': []}
        
        # 모델이 박스를 하나라도 검출해 냈을 경우 레이블 복원 가공 진행
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for j in range(len(boxes)):
                image_result['predictions'].append({
                    'class_id': int(boxes.cls[j]), 
                    'class_name': model.names[int(boxes.cls[j])], # ID를 한글/영어 클래스명으로 맵핑 복원
                    'confidence': float(boxes.conf[j]),           # AI의 확신도 스코어
                    'bbox': boxes.xyxy[j].tolist()                # 검출된 픽셀 좌표 [x1, y1, x2, y2]
                })
        all_results.append(image_result)
    return all_results


def evaluate_model(model, dataset_yaml_path, conf_thres=0.25, iou_thres=0.45, save_json=True, results_dir=None):
    """
    YOLOv11 프레임워크 자체의 검증 파이프라인(model.val)을 공식 트리거하여
    mAP@0.5, mAP@0.5:0.95, Precision, Recall 종합 표준 평가지표를 자동 산출하는 함수
    """
    original_cwd = Path.cwd() # 현재 터미널 동작 경로 백업
    try:
        # Ultralytics의 임시 파일 생성 분산 버그를 막기 위해 실험 타겟 폴더로 잠시 워킹 디렉토리 이동
        os.chdir(results_dir)
        metrics = model.val(
            data=dataset_yaml_path, 
            split='val', 
            conf=conf_thres, 
            iou=iou_thres, 
            save_json=save_json, 
            verbose=True, 
            project=".", # 현재 폴더 패스 직접 고정
            name=""
        )
    finally:
        os.chdir(original_cwd) # 프로세스 안전을 위해 원래 터미널 경로로 원복 복귀
    return metrics, results_dir


def calculate_iou(box1, box2):
    """
    수동 성능 결합 레이어용 핵심 기하학 공식 연산기
    두 바운딩 박스가 교차(Intersection)하는 면적을 합집합(Union)의 면적으로 나누어 IoU 스코어를 연산하는 함수
    """
    # 교차하는 사각형의 좌상단, 우하단 좌표 도출
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    
    # 겹치는 영역이 없을 경우의 예외 차단
    if x2 <= x1 or y2 <= y1: return 0.0
    
    intersection = (x2 - x1) * (y2 - y1) # 교차 면적
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) # 사각형 1 면적
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) # 사각형 2 면적
    union = area1 + area2 - intersection # 합집합 면적
    
    return intersection / union if union > 0 else 0.0


def match_predictions_with_gt_for_map(image_predictions, gt_annotations, iou_threshold=0.5):
    """
    [핵심 검증 알고리즘]
    AI가 검출한 예측 박스들과 사람의 실제 정답(GT) 박스들을 IoU 기반으로 1:1 대조 정합하여
    실제 정탐(True Positive, TP)인지 오탐(False Positive, FP)인지 엄격히 판별 분류하는 알고리즘 함수
    """
    individual_results = []
    all_predictions, all_gt_boxes = [], []
    
    # 1차 원시 연산 어레이 플래싱 수집
    for img_result in image_predictions:
        for pred in img_result['predictions']:
            all_predictions.append({'image_name': img_result['image_name'], 'class_name': pred['class_name'], 'confidence': pred['confidence'], 'bbox': pred['bbox']})
    for name, boxes in gt_annotations.items():
        for gt in boxes:
            all_gt_boxes.append({'image_name': name, 'class_name': gt['class_name'], 'bbox': gt['bbox']})
            
    # 전체 등장하는 불량/정상 고유 클래스 셋 구성
    all_classes = set([p['class_name'] for p in all_predictions] + [g['class_name'] for g in all_gt_boxes])
    
    # 객체 검출 평가지표 연산 표준에 맞춰 클래스별 독립 매칭 연산 개시
    for class_name in all_classes:
        class_predictions = [p for p in all_predictions if p['class_name'] == class_name]
        class_gt_boxes = [g for g in all_gt_boxes if g['class_name'] == class_name]
        
        # 독점 매칭 방지를 위해 확신 스코어(Confidence)가 높은 박스 순으로 셔플 정렬
        class_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 중복 매칭 방지를 위해 각 정답 박스의 매칭 완료 플래그 상태 저장소 선언
        gt_matched = {f"{gt['image_name']}_{gt['bbox']}": False for gt in class_gt_boxes}
        
        for pred in class_predictions:
            best_iou, best_gt_key = 0.0, None
            # 동일한 이미지 내부에 든 정답 객체들하고만 루프 매칭 대조 진행
            for gt in class_gt_boxes:
                if gt['image_name'] != pred['image_name']: continue
                key = f"{gt['image_name']}_{gt['bbox']}"
                if gt_matched[key]: continue # 이미 다른 높은 확신도 박스에 매칭 선점당한 정답은 스킵
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou: best_iou, best_gt_key = iou, key
            
            # 지정한 기준(보통 0.5)을 넘겼고, 매칭할 정답이 남아 있다면 최종 정탐(TP) 확정 판정
            is_tp = best_iou >= iou_threshold and best_gt_key is not None
            if is_tp: gt_matched[best_gt_key] = True # 해당 정답 박스 사용 마킹 처리
            
            # 한 박스 단위별 세부 평가 로그 기록 저장
            individual_results.append({
                'image_name': pred['image_name'], 
                'pred_class': pred['class_name'], 
                'gt_class': pred['class_name'] if is_tp else None, # 오탐일 경우 정답 클래스는 무효(None)화
                'confidence': pred['confidence'], 
                'iou': best_iou, 
                'is_tp': is_tp,
                'pred_bbox_x1': pred['bbox'][0], 'pred_bbox_y1': pred['bbox'][1], 'pred_bbox_x2': pred['bbox'][2], 'pred_bbox_y2': pred['bbox'][3]
            })
    return individual_results


# ==========================================
# 4. 통합 리포트 마크다운/CSV 영구 저장 레이어
# ==========================================

def save_unified_results_with_yolo_metrics(individual_results, yolo_metrics, gt_annotations, save_path):
    """
    YOLOv11의 공식 종합 스코어(mAP) 데이터와 수동 연산한 개별 정성 박스 로그 정보를
    WMS/MLOps 대시보드가 읽을 수 있는 단일 표준 통합 CSV 포맷 파일로 가공 및 영구 보존하는 함수
    """
    save_dir = Path(save_path)
    # 공식 메트릭스 클래스 아티팩트에서 핵심 스코어 정밀 소수점 플로팅 변환 추출
    overall_ap50, overall_map = float(yolo_metrics.box.map50), float(yolo_metrics.box.map)
    overall_precision, overall_recall = float(yolo_metrics.box.mp), float(yolo_metrics.box.mr)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # 세부 클래스 단위별 성능 맵 세팅 진행
    class_map_dict = {}
    if hasattr(yolo_metrics.box, 'maps') and yolo_metrics.box.maps is not None:
        for cls_id, cls_name in yolo_metrics.names.items():
            if cls_id < len(yolo_metrics.box.maps):
                map_50_95 = float(yolo_metrics.box.maps[cls_id])
                class_map_dict[cls_name] = {
                    'class_total_gt': 0, 
                    'class_total_predictions': 0, 
                    # 단일 세부 클래스별 mAP@0.5 유기 비례식 추정 계산식 적용
                    'class_map_50': map_50_95 * (overall_ap50 / overall_map if overall_map > 0 else 0),
                    'class_map_50_95': map_50_95, 'class_precision': overall_precision, 'class_recall': overall_recall, 'class_f1': overall_f1
                }
                
    # 실제 수량 정량 통계 카운팅 연산 진행
    class_gt_counts, class_pred_counts = {}, {}
    for boxes in gt_annotations.values():
        for b in boxes: class_gt_counts[b['class_name']] = class_gt_counts.get(b['class_name'], 0) + 1
    for r in individual_results:
        class_pred_counts[r['pred_class']] = class_pred_counts.get(r['pred_class'], 0) + 1
        
    # 데이터베이스 적재 규격에 맞춰 통계 디렉토리 로깅 팩 구현
    unified_results = []
    for i, result in enumerate(individual_results):
        class_name = result['pred_class']
        class_info = class_map_dict.get(class_name, {'class_total_gt': 0, 'class_total_predictions': 0, 'class_map_50': 0.0, 'class_map_50_95': 0.0, 'class_precision': 0.0, 'class_recall': 0.0, 'class_f1': 0.0})
        unified_results.append({
            'detection_id': i + 1, 'image_name': result['image_name'], 'pred_class': result['pred_class'], 'gt_class': result.get('gt_class'),
            'confidence': result['confidence'], 'iou': result['iou'], 'is_tp': result['is_tp'],
            'pred_bbox_x1': result['pred_bbox_x1'], 'pred_bbox_y1': result['pred_bbox_y1'], 'pred_bbox_x2': result['pred_bbox_x2'], 'pred_bbox_y2': result['pred_bbox_y2'],
            'class_total_gt': class_info['class_total_gt'], 'class_total_predictions': class_info['class_total_predictions'], 'class_map_50': class_info['class_map_50'], 'class_map_50_95': class_info['class_map_50_95'],
            'overall_map_50': overall_ap50, 'overall_map_50_95': overall_map, 'overall_precision': overall_precision, 'overall_recall': overall_recall, 'overall_f1': overall_f1
        })
        
    # 최종 영구 파일 출력 (한글 깨짐 없는 인코딩 지정)
    pd.DataFrame(unified_results).to_csv(save_dir / 'detection_results.csv', index=False, encoding='utf-8-sig')
    print(f"종합 성능 분석 보고서가 생성되었습니다 -> {save_dir}")


# ==========================================
# 5. 오케스트레이션 메인 컨트롤러 레이어
# ==========================================

def main():
    start_time = datetime.now()
    print(f"\n평가 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 스크립트 실행 드라이브 위치에 구애받지 않고 유연하게 C:\SMT_multi_modal 최상위 경로 포착
    project_root = get_project_root()
    
    # 정교한 5.models 아키텍처 격리 디렉토리 구조 자동 연동 세팅
    model_sd_warehouse = project_root / 'dataset' / '5.models' / 'detection_sd'
    
    dataset_config_path = model_sd_warehouse / 'metadata' / 'dataset.yaml'
    gt_annotations_path = model_sd_warehouse / 'metadata' / 'val_annotations.csv' # 검증 세트 타겟팅
    base_results_dir = model_sd_warehouse / 'results_evaluation'
    
    # [MLOps 자동화 안착] results_train 내부를 역추적하여 가장 마지막(최신)에 가공 완료된 타임스탬프 실험 폴더의 best.pt를 자동 픽업
    model_train_base = model_sd_warehouse / 'results_train'
    latest_train_folders = sorted(list(model_train_base.glob('20*')))
    target_pt = latest_train_folders[-1] / 'weights' / 'best.pt' if latest_train_folders else project_root / 'models' / '2.detection_sd' / 'yolo11n.pt'
    
    print(f"=== SMT SD 공정 모델 최종 검증 및 평가 가동 ===")
    print(f"평가 대상 가중치: {target_pt}")
    
    # 핵심 추론 컴포넌트 엔진 인스턴스화
    model = YOLO(str(target_pt))
    results_dir = get_results_dir(base_results_dir)
    
    # 파이프라인 프로세스 순차 가동
    # 1단계: YOLO 내장 val 함수 기반 메트릭 산출
    results, _ = evaluate_model(model=model, dataset_yaml_path=dataset_config_path, save_json=True, results_dir=results_dir, conf_thres=0.5, iou_thres=0.5)
    
    # 2단계: 유효 이미지 순회 스캔 및 1차 예측 바운딩 박스 추론 수집
    val_images = get_validation_images(dataset_config_path)
    image_predictions = predict_on_images(model, val_images, conf_thres=0.5, iou_thres=0.5)
    
    # 3단계: 정답 셋 로드 수행
    gt_annotations = load_ground_truth_annotations(gt_annotations_path)
    
    # 4단계: IoU 정밀 오탐 스크리닝 매칭 매트릭스 구동
    individual_results = match_predictions_with_gt_for_map(image_predictions=image_predictions, gt_annotations=gt_annotations, iou_threshold=0.5)
    
    # 5단계: 대시보드 백업용 최종 통합 CSV 데이터 파일 빌드 출력
    save_unified_results_with_yolo_metrics(individual_results=individual_results, yolo_metrics=results, gt_annotations=gt_annotations, save_path=results_dir)

if __name__ == "__main__":
    main()
