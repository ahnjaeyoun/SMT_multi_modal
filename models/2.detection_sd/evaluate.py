import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        font_list = [font.name for font in fm.fontManager.ttflist]
        korean_fonts = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR', 'DejaVu Sans']
        
        for font in korean_fonts:
            if font in font_list:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                return
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다. 한글 표시에 문제가 있을 수 있습니다.")
        
    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

setup_korean_font()

warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')

def get_results_dir(base_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_path) / f"eval_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def load_ground_truth_annotations(annotations_csv_path):
    """Ground Truth 어노테이션 데이터를 로드"""
    gt_annotations = defaultdict(list)
    try:
        df = pd.read_csv(annotations_csv_path, encoding='utf-8')
        for _, row in df.iterrows():
            image_path = row['image_path']
            image_name = Path(image_path).name
            
            gt_box = {
                'class_name': row['class_name'],
                'bbox': [float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])]
            }
            gt_annotations[image_name].append(gt_box)
        print(f"GT 어노테이션 로드 완료: {len(gt_annotations)}개 이미지, 총 {len(df)}개 박스")
    except Exception as e:
        print(f"GT 어노테이션 로드 실패: {e}")
    return dict(gt_annotations)

def get_validation_images(dataset_yaml_path):
    """test셋 대신 프로젝트 구조에 맞춰 val 데이터셋의 경로와 이미지 수집"""
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    # 📍 아키텍처 규격에 따라 'val' 폴더 타겟팅
    val_path = dataset_config.get('val')
    print(f"평가 타겟 [val] 데이터셋 이미지 경로: {val_path}")
    
    if val_path is None:
        raise ValueError("데이터셋 YAML에서 val 경로를 찾을 수 없습니다.")
    
    dataset_root = dataset_config.get('path', '')
    full_val_path = Path(dataset_root) / val_path
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    val_images = []
    
    if full_val_path.exists():
        for ext in image_extensions:
            val_images.extend(list(full_val_path.glob(f"*{ext}")))
        print(f"검증 대상 이미지 파일 수: {len(val_images)}")
    else:
        print(f"경고: 검증 이미지 경로가 존재하지 않습니다: {full_val_path}")
    
    return sorted(val_images)

def predict_on_images(model, image_paths, conf_thres=0.25, iou_thres=0.45):
    """개별 이미지들에 대해 예측 수행"""
    all_results = []
    for img_path in tqdm(image_paths, desc="이미지 추론 평가 중", unit="img"):
        results = model.predict(
            source=str(img_path),
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
            save=False
        )
        
        image_result = {
            'image_path': str(img_path),
            'image_name': img_path.name,
            'predictions': []
        }
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for j in range(len(boxes)):
                pred = {
                    'class_id': int(boxes.cls[j]),
                    'class_name': model.names[int(boxes.cls[j])],
                    'confidence': float(boxes.conf[j]),
                    'bbox': boxes.xyxy[j].tolist()
                }
                image_result['predictions'].append(pred)
        all_results.append(image_result)
    print("예측 완료!")
    return all_results

def evaluate_model(model, dataset_yaml_path, conf_thres=0.25, iou_thres=0.45, save_json=True, results_dir=None):
    """학습된 모델의 성능 검증 평가 수행"""
    print("SMT PR 공정 DETECTION 모델 성능 정밀 평가를 시작합니다...")
    print(f"설정 피라미터: conf={conf_thres}, iou={iou_thres}")
    
    if results_dir is None:
        raise ValueError("결과 저장 디렉토리가 정의되지 않았습니다.")
    
    original_cwd = Path.cwd()
    try:
        os.chdir(results_dir)
        # 📍 val 데이터셋을 타겟으로 명시적 지정 평가
        metrics = model.val(
            data=dataset_yaml_path,
            split='val',
            conf=conf_thres,
            iou=iou_thres,
            save_json=save_json,
            verbose=True,
            project=".",
            name=""
        )
    finally:
        os.chdir(original_cwd)
    return metrics, results_dir

def calculate_iou(box1, box2):
    """두 박스 간의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def match_predictions_with_gt_for_map(image_predictions, gt_annotations, iou_threshold=0.5):
    """mAP 계산을 위한 예측값과 GT 매칭"""
    individual_results = []
    all_predictions = []
    all_gt_boxes = []
    
    for img_result in image_predictions:
        image_name = img_result['image_name']
        for pred in img_result['predictions']:
            all_predictions.append({
                'image_name': image_name,
                'class_name': pred['class_name'],
                'confidence': pred['confidence'],
                'bbox': pred['bbox']
            })
    
    for image_name, gt_boxes in gt_annotations.items():
        for gt_box in gt_boxes:
            all_gt_boxes.append({
                'image_name': image_name,
                'class_name': gt_box['class_name'],
                'bbox': gt_box['bbox']
            })
    
    all_classes = set([p['class_name'] for p in all_predictions] + [g['class_name'] for g in all_gt_boxes])
    
    for class_name in all_classes:
        class_predictions = [p for p in all_predictions if p['class_name'] == class_name]
        class_gt_boxes = [g for g in all_gt_boxes if g['class_name'] == class_name]
        class_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        gt_matched = {f"{gt['image_name']}_{gt['bbox']}": False for gt in class_gt_boxes}
        
        for pred in class_predictions:
            best_iou = 0.0
            best_gt_key = None
            
            for gt in class_gt_boxes:
                if gt['image_name'] != pred['image_name']:
                    continue
                gt_key = f"{gt['image_name']}_{gt['bbox']}"
                if gt_matched[gt_key]:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_key = gt_key
            
            is_tp = best_iou >= iou_threshold and best_gt_key is not None
            if is_tp:
                gt_matched[best_gt_key] = True
            
            individual_results.append({
                'image_name': pred['image_name'],
                'pred_class': pred['class_name'],
                'gt_class': pred['class_name'] if is_tp else None,
                'confidence': pred['confidence'],
                'iou': best_iou,
                'is_tp': is_tp,
                'pred_bbox_x1': pred['bbox'][0],
                'pred_bbox_y1': pred['bbox'][1],
                'pred_bbox_x2': pred['bbox'][2],
                'pred_bbox_y2': pred['bbox'][3]
            })
    return individual_results

def save_image_level_results(image_predictions, save_path, class_names):
    """이미지별 예측 결과를 요약 분석하여 리포트로 저장"""
    save_dir = Path(save_path)
    image_summary = []
    
    for img_result in image_predictions:
        img_name = img_result['image_name']
        predictions = img_result['predictions']
        total_predictions = len(predictions)
        class_counts = defaultdict(int)
        confidence_scores = []
        
        for pred in predictions:
            class_counts[pred['class_name']] += 1
            confidence_scores.append(pred['confidence'])
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        max_confidence = max(confidence_scores) if confidence_scores else 0
        
        image_summary.append({
            'image_name': img_name,
            'total_detections': total_predictions,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'detected_classes': ', '.join(class_counts.keys()),
            'class_distribution': dict(class_counts)
        })
        
    return image_summary

def save_unified_results_with_yolo_metrics(individual_results, yolo_metrics, gt_annotations, save_path):
    """YOLO 종합 메트릭과 개별 분석 매칭 결과를 병합 통합본 생성"""
    save_dir = Path(save_path)
    
    overall_ap50 = float(yolo_metrics.box.map50)
    overall_map = float(yolo_metrics.box.map)
    overall_precision = float(yolo_metrics.box.mp)
    overall_recall = float(yolo_metrics.box.mr)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\n[METRICS] YOLO 검증최종 스코어:")
    print(f" - mAP@0.5: {overall_ap50:.4f} | mAP@0.5:0.95: {overall_map:.4f}")
    print(f" - Precision: {overall_precision:.4f} | Recall: {overall_recall:.4f} | F1-Score: {overall_f1:.4f}")
    
    class_map_dict = {}
    if hasattr(yolo_metrics.box, 'maps') and yolo_metrics.box.maps is not None:
        class_maps = yolo_metrics.box.maps
        for cls_id, cls_name in yolo_metrics.names.items():
            if cls_id < len(class_maps):
                map_50_95 = float(class_maps[cls_id])
                map_50 = map_50_95 * (overall_ap50 / overall_map) if overall_map > 0 else 0
                class_map_dict[cls_name] = {
                    'class_total_gt': 0, 'class_total_predictions': 0,
                    'class_map_50': map_50, 'class_map_50_95': map_50_95,
                    'class_precision': overall_precision, 'class_recall': overall_recall, 'class_f1': overall_f1
                }
                
    class_gt_counts = {}
    class_pred_counts = {}
    for gt_boxes in gt_annotations.values():
        for gt_box in gt_boxes:
            class_gt_counts[gt_box['class_name']] = class_gt_counts.get(gt_box['class_name'], 0) + 1
            
    for result in individual_results:
        class_pred_counts[result['pred_class']] = class_pred_counts.get(result['pred_class'], 0) + 1
        
    for class_name in class_map_dict:
        class_map_dict[class_name]['class_total_gt'] = class_gt_counts.get(class_name, 0)
        class_map_dict[class_name]['class_total_predictions'] = class_pred_counts.get(class_name, 0)
        
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
        
    pd.DataFrame(unified_results).to_csv(save_dir / 'detection_results.csv', index=False, encoding='utf-8-sig')
    
    class_summary = []
    for cls_name, class_info in class_map_dict.items():
        class_summary.append({
            'class_name': cls_name, 'total_gt': class_gt_counts.get(cls_name, 0), 'total_predictions': class_pred_counts.get(cls_name, 0),
            'ap_at_iou_50': class_info['class_map_50'], 'map_50_95': class_info['class_map_50_95'],
            'precision': class_info['class_precision'], 'recall': class_info['class_recall'], 'f1_score': class_info['class_f1']
        })
    pd.DataFrame(class_summary).to_csv(save_dir / 'class_map_results.csv', index=False, encoding='utf-8-sig')
    print(f"종합 성능 분석 평가 보고서가 {save_dir} 폴더에 정상 저장되었습니다.")

def main():
    start_time = datetime.now()
    print(f"\n평가 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 📍 프로젝트 루트(C:\SMT_multi_modal) 자동 연동 경로 구조 설계
    project_root = Path(__file__).resolve().parents[2]
    model_pr_dir = project_root / 'models' / '3.detection_pr'
    
    # 가공된 의존성 패스 자동 연결
    dataset_config_path = model_pr_dir / 'dataset' / 'metadata' / 'dataset.yaml'
    gt_annotations_path = model_pr_dir / 'dataset' / 'metadata' / 'val_annotations.csv'  # test에서 val로 연동 보정
    base_results_dir = model_pr_dir / 'results_evaluation'
    
    # 가장 최근에 학습된 가중치 연동을 위한 best.pt 디렉토리 자동 스캔 추천
    # 수동 탐색용 디폴트 패스 지정 (train 파이프라인에서 생성한 최신 pt 폴더를 지정해 쓰시면 됩니다)
    model_path = model_pr_dir / 'results_train'
    latest_train_folders = sorted(list(model_path.glob('20*')))
    if latest_train_folders:
        target_pt = latest_train_folders[-1] / 'weights' / 'best.pt'
    else:
        # 폴더가 아직 없을 때를 대비한 기본 예외 디폴트 패스 지정
        target_pt = model_pr_dir / 'yolo11n.pt'
        
    print(f"평가 대상 타겟 가중치: {target_pt}")
    model = YOLO(str(target_pt))
    
    eval_args = {
        'conf_thres': 0.5,
        'iou_thres': 0.5,
    }
    
    results_dir = get_results_dir(base_results_dir)
    
    # 1. YOLO 기반 검증 평가 수행
    results, _ = evaluate_model(
        model=model,
        dataset_yaml_path=dataset_config_path,
        save_json=True,
        results_dir=results_dir,
        **eval_args
    )
    
    # 2. 개별 예측 이미지 스캔 (test에서 val로 타겟 스위칭)
    print("\n[PROCESS] 검증 데이터셋 이미지 개별 추론 스캔 중...")
    val_images = get_validation_images(dataset_config_path)
    image_predictions = predict_on_images(model, val_images, eval_args['conf_thres'], eval_args['iou_thres'])
    
    # 3. Ground Truth 대조 및 수동 매칭 스코어링 결합
    print("\n[PROCESS] GT 어노테이션 로드 및 정합성 검증 대조 중...")
    gt_annotations = load_ground_truth_annotations(gt_annotations_path)
    
    individual_results = match_predictions_with_gt_for_map(
        image_predictions=image_predictions,
        gt_annotations=gt_annotations,
        iou_threshold=0.5
    )
    
    # 4. 종합 아티팩트 레포트 영구 저장 저장
    save_unified_results_with_yolo_metrics(
        individual_results=individual_results,
        yolo_metrics=results,
        gt_annotations=gt_annotations,
        save_path=results_dir
    )
    
    save_image_level_results(image_predictions=image_predictions, save_path=results_dir, class_names=model.names)
    
    end_time = datetime.now()
    print(f"\n=== 평가 프로세스 전체 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (소요 시간: {end_time - start_time}) ===")

if __name__ == "__main__":
    main()