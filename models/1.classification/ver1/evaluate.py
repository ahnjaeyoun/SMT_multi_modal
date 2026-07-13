import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from pathlib import Path
from train import GEMSMultimodal, SMTMultimodalDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_csv = Path("processed_data/val/metadata.csv")
    if not val_csv.exists():
        print("🚨 에러: 검증용 메타데이터가 없습니다.")
        return

    # Albumentations이 적용된 Dataset (is_train=False)
    val_loader = DataLoader(SMTMultimodalDataset(val_csv, is_train=False), batch_size=16, shuffle=False)
    
    print("🧠 가중치 로드 중...")
    model = GEMSMultimodal().to(device)
    model.load_state_dict(torch.load("best_multimodal.pt", map_location=device, weights_only=True))
    model.eval()
    
    ground_truths = []
    predictions = []
    
    print("🎯 평가 시작...")
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device)
            sensors = batch['sensor_data'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(imgs, sensors)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
            
    f1 = f1_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions)
    recall = recall_score(ground_truths, predictions)
    cm = confusion_matrix(ground_truths, predictions)
    
    print("\n" + "="*50)
    print(" 🏆 GEMS 멀티모달 분류 모델 최종 평가 결과")
    print("="*50)
    print(f" F1-Score : {f1 * 100:.2f}%")
    print(f" Precision: {precision * 100:.2f}%")
    print(f" Recall   : {recall * 100:.2f}%")
    print("\n [Confusion Matrix]")
    print(f"  - TN(정상 판정): {cm[0][0]:4d} | FP(불량 오탐): {cm[0][1]:4d}")
    print(f"  - FN(정상 미탐): {cm[1][0]:4d} | TP(불량 판정): {cm[1][1]:4d}")
    print("="*50)

if __name__ == "__main__":
    main()