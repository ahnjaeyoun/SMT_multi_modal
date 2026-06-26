import os
import torch
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
    except Exception as e:
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

setup_korean_font()

# Ultralytics에서 발생하는 한글 관련 경고 무시
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')

def get_project_root():
    env_root = os.getenv('PROJECT_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]

def get_train_id():
    """학습 ID 생성 (타임스탬프 기반)"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_training_summary(results, val_results, training_args, save_dir):
    """
    학습 결과를 txt 파일로 저장합니다.
    """
    summary_path = save_dir / 'training_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== 학습 설정 ===\n")
        f.write(f"학습 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n데이터셋: {training_args['data']}\n")
        f.write(f"이미지 크기: {training_args['imgsz']} | 배치 크기: {training_args['batch']} | 에포크: {training_args['epochs']}\n\n")
        
        f.write("=== 훈련 세트 최종 스코어 ===\n")
        if results and hasattr(results, 'results_dict'):
            train_metrics = results.results_dict
            f.write(f"훈련 최종 Box Loss: {train_metrics.get('train/box_loss', 0):.4f}\n\n")
        
        f.write("=== 검증 세트 최종 스코어 ===\n")
        if val_results:
            val_metrics = val_results.results_dict
            f.write(f"검증 mAP@0.5: {val_metrics.get('metrics/mAP50(B)', 0):.4f}\n")
            f.write(f"검증 mAP@0.5:0.95: {val_metrics.get('metrics/mAP50-95(B)', 0):.4f}\n")
            f.write(f"검증 Precision: {val_metrics.get('metrics/precision(B)', 0):.4f}\n")
            f.write(f"검증 Recall: {val_metrics.get('metrics/recall(B)', 0):.4f}\n")
        print(f"학습 요약 아티팩트 저장 완료: {summary_path}")

# YOLOv11 모델을 학습시키고 검증 세트를 평가
def train_model(model, training_args):
    # 1. 학습 진행 (이 과정에서 자동으로 best.pt와 last.pt가 지정된 경로에 안전하게 저장됨)
    results = model.train(**training_args)
    
    # 디렉토리 경로 연산
    save_dir = Path(training_args['project']) / training_args['name']
    
    print("\n[VALIDATION] 최종 모델 검증 세트 평가 가동...")

    # 2. 검증 진행
    val_results = model.val(data=training_args['data'], split='val') 
    
    # 3. 로그 저장
    save_training_summary(results, val_results, training_args, save_dir)
    return results, val_results

def main():
    # 스크립트 위치에 종속되지 않고 최상위 루트 확보
    project_root = get_project_root()
    
    model_sd_warehouse = project_root / 'dataset' / '5.models' / 'detection_sd'
    
    # 전처리기(preprocess.py)가 생성해 둔 전용 데이터셋 YAML 패스 연결
    dataset_config_path = model_sd_warehouse / 'metadata' / 'dataset.yaml'
    
    # 사전 학습된 기본 가중치(yolo11n.pt) 위치 정의
    pretrained_model_path = project_root / 'models' / '2.detection_sd' / 'yolo11n.pt'
    
    # 학습 결과 산출물이 격리되어 쌓일 폴더 트리 매핑
    results_base = model_sd_warehouse / 'results_train'
    
    train_id = get_train_id()
    train_dir = results_base / train_id
    
    print(f"=== SMT Solder Deposit(SD) 사전 공정 학습 엔진 가동 ===")
    print(f"프로젝트 루트: {project_root}")
    print(f"데이터셋 YAML 패스: {dataset_config_path}")
    print(f"산출물 저장 루트: {train_dir}\n")
    
    # 가중치 인스턴스화 (초기 백본 사용)
    if not pretrained_model_path.exists():
        print(f"안내: 초기 백본 가중치가 해당 경로에 없어 다운로드를 시작합니다 -> {pretrained_model_path}")
    model = YOLO(str(pretrained_model_path))
    
    # 하이퍼파라미터 및 경로 정의 (안정화 튜닝 적용 완료)
    training_args = {
        'data': str(dataset_config_path),  # 데이터셋 설정 파일 경로
        'epochs': 200,                     # 전체 학습 에포크 수
        'batch': 16,                       # 안정적인 16 고정
        'imgsz': 512,                      # 입력 이미지 크기
        'patience': 50,                    # 조기 종료 조건
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4,                      
        'project': str(results_base),      
        'name': train_id,                  
        'exist_ok': True,                  
        'pretrained': True,                
        
        # 학습 안정화 코드
        'optimizer': 'auto',     # 데이터에 맞춰 가장 안정적인 옵티마이저 선택
        'amp': False,            # Mixed Precision 연산 끄기 (NaN/Inf 에러 방지)
        'lr0': 0.001,            # 초기 학습률을 낮게 설정하여 보폭을 안전하게 만듦
        'cos_lr': True,          # 코사인 학습률 스케줄러 적용
    }
    
    print("YOLOv11 모델 파인튜닝 학습을 처음부터 시작합니다...")
    train_model(model, training_args)
    print("\n[COMPLETE] 모델 파인튜닝 및 산출물 누적이 정상 종료되었습니다.")

if __name__ == "__main__":
    main()
