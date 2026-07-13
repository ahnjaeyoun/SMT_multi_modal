import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from pathlib import Path

# train.py에서 데이터셋과 모델 구조를 가져옵니다.
try:
    from train import DualEncodingModel, SMTMultimodalDataset
except ImportError:
    print("🚨 에러: train.py 파일에서 모델과 데이터셋 클래스를 불러올 수 없습니다.")
    raise

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    """혼동 행렬 시각화 및 저장"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """ROC Curve 시각화 및 저장"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (과탐률)')
    plt.ylabel('True Positive Rate (정탐률)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_model(model_path, test_csv, results_dir, device):
    print(f"🔍 [1/4] 평가 준비 중... (디바이스: {device})")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 모델 및 데이터 로드
    model = DualEncodingModel(num_classes=2).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 가중치 파일을 찾을 수 없습니다: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_dataset = SMTMultimodalDataset(test_csv, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"📊 테스트 데이터: 총 {len(test_dataset)}건")

    # 2. 결과 수집용 컨테이너
    all_preds, all_labels, all_probs = [], [], []
    all_process_types = []
    error_cases = [] # 오답 노트용

    print("🚀 [2/4] 테스트 데이터 추론 시작...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = batch['image'].to(device)
            sensor_data = batch['sensor_data'].to(device)
            labels = batch['label'].to(device)
            process_types = batch['process_type']
            
            outputs = model(images, sensor_data)
            
            # [핵심] Softmax를 통해 0~1 사이의 확률값 추출
            probs = F.softmax(outputs, dim=1)[:, 1] 
            # 2. [수정됨] 커스텀 임계값 설정 및 비교
            custom_threshold = 0.8788
            preds = (probs >= custom_threshold).long()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_process_types.extend(process_types)
            
            # 오답 데이터(FP, FN) 정보 저장
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                
                if true_label != pred_label:
                    # Dataset에서 원본 행을 추적하여 경로 추출
                    global_idx = batch_idx * test_loader.batch_size + i
                    orig_row = test_dataset.df.iloc[global_idx]
                    
                    error_type = "과탐(FP)" if pred_label == 1 else "미탐(FN)"
                    
                    error_cases.append({
                        'Error_Type': error_type,
                        'Process': process_types[i],
                        'True_Label': 'Defect' if true_label == 1 else 'Normal',
                        'Predicted_Label': 'Defect' if pred_label == 1 else 'Normal',
                        'Defect_Probability': f"{probs[i].item()*100:.2f}%",
                        'Image_Path': orig_row['image_path'],
                        'Sensor_Path': orig_row['sensor_path']
                    })

    print("📈 [3/4] 평가지표 계산 및 리포트 생성 중...")
    
    # 3. 평가지표 계산
    class_names = ['Normal(0)', 'Defect(1)']
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    
    # ROC 커브 및 최적의 Threshold 계산
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic을 이용한 최적의 임계값 찾기
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # 공정별 분리 분석
    df_results = pd.DataFrame({'Label': all_labels, 'Pred': all_preds, 'Process': all_process_types})
    process_reports = {}
    for p_type in df_results['Process'].unique():
        p_data = df_results[df_results['Process'] == p_type]
        if len(p_data) > 0:
            process_reports[p_type] = classification_report(p_data['Label'], p_data['Pred'], output_dict=True, zero_division=0)

    print("💾 [4/4] 결과물 파일 저장 중...")
    
    # [결과물 1] Confusion Matrix 이미지 저장
    plot_confusion_matrix(cm, class_names, os.path.join(results_dir, 'confusion_matrix.png'))
    
    # [결과물 2] ROC Curve 이미지 저장
    plot_roc_curve(fpr, tpr, roc_auc, os.path.join(results_dir, 'roc_curve.png'))
    
    # [결과물 3] 오답 노트 CSV 저장
    if error_cases:
        df_errors = pd.DataFrame(error_cases)
        # 치명적인 미탐(FN)이 위로 오도록 정렬
        df_errors = df_errors.sort_values(by='Error_Type', ascending=False)
        df_errors.to_csv(os.path.join(results_dir, 'misclassified_cases.csv'), index=False, encoding='utf-8-sig')
    
    # [결과물 4] 종합 평가 리포트 TXT 저장
    with open(os.path.join(results_dir, 'evaluation_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=== SMT 멀티모달 모델 종합 평가 리포트 ===\n")
        f.write(f"테스트 데이터 수: {len(all_labels)}건\n\n")
        
        f.write("1. 전체 분류 성능 (Classification Report)\n")
        f.write(report_str + "\n\n")
        
        f.write("2. 혼동 행렬 상세 (Confusion Matrix)\n")
        f.write(f"- 정상 데이터 정확히 맞춤 (TN): {cm[0][0]}건\n")
        f.write(f"- 정상인데 불량으로 판정 [과탐/FP]: {cm[0][1]}건 (주의 요망)\n")
        f.write(f"- 불량인데 정상으로 판정 [미탐/FN]: {cm[1][0]}건 (★치명적 결함)\n")
        f.write(f"- 불량 데이터 정확히 맞춤 (TP): {cm[1][1]}건\n\n")
        
        f.write("3. 확률 및 임계값 분석\n")
        f.write(f"- ROC-AUC Score: {roc_auc:.4f}\n")
        f.write(f"- 추천 최적 임계값 (Threshold): {optimal_threshold:.4f} (이 값 이상이면 불량으로 판정 시 효율 극대화)\n\n")
        
        f.write("4. 공정별 F1-Score (불량 검출 기준)\n")
        for p_type, p_report in process_reports.items():
            # 라벨 1(불량)이 데이터셋에 존재하는 경우에만 기록
            defect_f1 = p_report.get('1', {}).get('f1-score', 'N/A')
            f.write(f"- {p_type}: {defect_f1}\n")

    print(f"\n🎉 모든 평가가 완료되었습니다. 결과물은 [{results_dir}] 폴더를 확인하세요.")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    
    # test.csv 경로 (preprocess.py에서 생성된 test 메타데이터)
    test_csv_path = current_dir / "processed_data" / "test" / "metadata.csv"
    
    # 저장된 최고의 모델 가중치 경로
    # 모델 폴더를 명시적으로 찾도록 경로 설정
    model_weight_path = current_dir / "results_train" 
    
    # 가장 최근에 생성된 experiment_id 폴더를 찾아서 best_model.pth 선택
    if model_weight_path.exists():
        subdirs = [d for d in model_weight_path.iterdir() if d.is_dir()]
        if subdirs:
            latest_exp_dir = max(subdirs, key=os.path.getmtime)
            best_model_path = latest_exp_dir / "checkpoints" / "best_model.pth"
        else:
            best_model_path = model_weight_path / "best_model.pth" # 기본 백백 경로
    else:
        best_model_path = current_dir / "best_model.pth"

    # 평가 결과물이 저장될 폴더
    eval_results_dir = current_dir / "evaluation_results"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not test_csv_path.exists():
        print(f"🚨 에러: 테스트 데이터 목차({test_csv_path})가 없습니다.")
    else:
        evaluate_model(str(best_model_path), str(test_csv_path), str(eval_results_dir), device)