import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from pathlib import Path

# ==============================================================================
# 🚨 [중요] 외부 의존성 로드
# ==============================================================================
try:
    from timesnet import Model as TimesNet
except ImportError:
    print("🚨 에러: 'timesnet.py' 파일을 찾을 수 없습니다. train.py와 같은 폴더에 위치시켜 주세요.")
    raise

class Config:
    """TimesNet을 피처 추출기(Feature Extractor)로 사용하기 위한 설정"""
    def __init__(self):
        # [수정 1] anomaly_detection -> classification 모드로 변경하여 Dense Vector 추출
        self.task_name = 'classification'
        self.seq_len = 5
        self.label_len = 0
        self.pred_len = 0
        self.enc_in = 5       # 센서 차원 (온, 습, 진, 가, 소)
        self.d_model = 64     # 연산 효율성을 위한 내부 차원 축소
        self.d_ff = 128
        self.num_kernels = 6
        self.e_layers = 3
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.1
        self.top_k = 2
        # [핵심] TimesNet의 최종 출력을 512차원 벡터로 강제하여 Swin(768)과 균형을 맞춤
        self.num_class = 512  

# ==============================================================================
# 1. 멀티모달 데이터셋 클래스 (경로 하드코딩 수정본)
# ==============================================================================
class SMTMultimodalDataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        self.df = pd.read_csv(csv_path)
        # [수정 2] 현재 파일(train.py) 기준으로 프로젝트 루트 추적 (MLOps 이식용)
        self.project_root = Path(__file__).resolve().parents[2]
        
        if is_train:
            self.transform = A.Compose([
                A.Resize(256, 256), # Swin v2 T 기본 입력 사이즈인 256으로 최적화 (OOM 방지)
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 동적 상대 경로를 절대 경로로 안전하게 복원
        # 이전 디렉토리명 '1.classsification'이 메타데이터에 남아있는 경우를 대비해 현재 디렉토리명으로 자동 교정
        sensor_path_str = str(row['sensor_path']).replace('1.classsification', '2.classification')
        
        img_full_path = self.project_root / row['image_path']
        sensor_full_path = self.project_root / sensor_path_str
        
        image = np.array(Image.open(img_full_path).convert('RGB'))
        image_tensor = self.transform(image=image)['image']
        
        sensor_data = np.load(sensor_full_path)
        
        # [필수] 센서 데이터 스케일링 (간단한 Min-Max 또는 Standardizer 적용 필요 시 여기에 추가)
        sensor_tensor = torch.FloatTensor(sensor_data)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        process_type = row['process_type']
        
        return {
            'image': image_tensor,
            'sensor_data': sensor_tensor,
            'label': label,
            'process_type': process_type
        }

# ==============================================================================
# 2. 모델 아키텍처: Early Fusion (Swin V2 + TimesNet)
# ==============================================================================
class DualEncodingModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 비전: Swin Transformer V2 (출력 768차원, 사전학습 가중치 사용)
        self.image_encoder = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        self.image_encoder.head = nn.Identity()
        
        # 센서: TimesNet (출력 512차원으로 세팅됨)
        configs = Config()
        self.sensor_encoder = TimesNet(configs)
        
        # 결합: 이미지(768) + 센서(512) = 1280차원
        self.fusion = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, sensor_data):
        # 1. 이미지 특징 추출 -> [B, 768]
        image_features = self.image_encoder(image)
        
        # 2. 센서 특징 추출을 위한 Dummy Padding Mask 생성 (TimesNet 내부 에러 방지)
        B, seq_len = sensor_data.shape[0], sensor_data.shape[1]
        x_mark_enc = torch.ones(B, seq_len).to(sensor_data.device)
        
        # 3. 센서 특징 추출 -> [B, 512]
        sensor_features = self.sensor_encoder(sensor_data, x_mark_enc, None, None)
        
        # 4. 특징 결합 및 최종 예측
        combined_features = torch.cat([image_features, sensor_features], dim=1)
        output = self.fusion(combined_features)
        
        return output

# ==============================================================================
# 3. 조기 종료 (Early Stopping) 유틸리티
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ==============================================================================
# 4. 학습 파이프라인 (Gradient Clipping & Scheduler 탑재)
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir, resume_checkpoint=None):
    os.makedirs(results_dir, exist_ok=True)
    start_time = datetime.now()
    print(f"\n=== 학습 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    # [수정 3] 초고속 최적화 학습 엔진: OneCycleLR 스케줄러 적용
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        epochs=num_epochs, 
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3
    )
    
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # 이어하기 로직
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        experiment_id = checkpoint['experiment_id']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        experiment_dir = os.path.join(results_dir, experiment_id)
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        history_file = os.path.join(experiment_dir, f'training_history_{experiment_id}.json')
        
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                training_history = json.load(f)
        else:
            training_history = {'experiment_info': {'experiment_id': experiment_id}, 'epoch_results': []}
    else:
        best_val_acc = 0.0
        experiment_id = start_time.strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(results_dir, experiment_id)
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        training_history = {'experiment_info': {'experiment_id': experiment_id, 'best_val_acc': 0.0}, 'epoch_results': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            images = batch['image'].to(device)
            sensor_data = batch['sensor_data'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, sensor_data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # [수정 4] 학습 안정화를 위한 Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step() # Batch 단위 스케줄러 업데이트
            
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 검증 (Validation)
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        val_process_stats = {"사전공정": {"correct": 0, "total": 0}, "납땜공정": {"correct": 0, "total": 0}}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(device)
                sensor_data = batch['sensor_data'].to(device)
                labels = batch['label'].to(device)
                process_types = batch['process_type']
                
                outputs = model(images, sensor_data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += images.size(0)
                
                for i, p_type in enumerate(process_types):
                    if p_type in val_process_stats:
                        val_process_stats[p_type]["total"] += 1
                        if preds[i] == labels[i]:
                            val_process_stats[p_type]["correct"] += 1
                        
        v_loss = val_loss / val_total
        v_acc = val_corrects.double() / val_total
        print(f'Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}')
        
        # 최고 성능 달성 시 저장
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            training_history['experiment_info']['best_val_acc'] = float(best_val_acc)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"🌟 새로운 최고 성능 갱신! 모델 저장됨.")
            
        # 히스토리 저장
        epoch_result = {
            'epoch': epoch + 1,
            'train': {'loss': float(epoch_loss), 'accuracy': float(epoch_acc)},
            'validation': {'loss': float(v_loss), 'accuracy': float(v_acc)}
        }
        training_history['epoch_results'].append(epoch_result)
        with open(os.path.join(experiment_dir, f'training_history_{experiment_id}.json'), 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=4, ensure_ascii=False)
            
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'experiment_id': experiment_id,
            'best_val_acc': best_val_acc
        }, os.path.join(checkpoint_dir, 'last_checkpoint.pth'))
        
        # Early Stopping 체크
        early_stopping(v_loss)
        if early_stopping.early_stop:
            print("🛑 조기 종료(Early Stopping)가 발동되었습니다. 학습을 중단합니다.")
            break
        print()

    print(f"\n=== 학습 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

def main(resume_from=None):
    # preprocess.py 출력 경로
    current_dir = Path(__file__).resolve().parent
    train_csv = current_dir / "processed_data" / "train" / "metadata.csv"
    val_csv = current_dir / "processed_data" / "val" / "metadata.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        print(f"🚨 에러: {train_csv} 를 찾을 수 없습니다. 전처리를 먼저 수행하세요.")
        return

    results_dir = current_dir / "results_train"
    batch_size = 32 # GPU 메모리에 따라 조절
    num_epochs = 50
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 연산 장치: {device}")
    
    train_loader = DataLoader(SMTMultimodalDataset(train_csv, is_train=True), batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(SMTMultimodalDataset(val_csv, is_train=False), batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    
    model = DualEncodingModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 과적합 방지를 위한 Label Smoothing 적용
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n🔄 체크포인트 이어서 학습: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        num_epochs = max(0, num_epochs - start_epoch)
        if num_epochs == 0: return
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir, resume_checkpoint=resume_from)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 경로')
    args = parser.parse_args()
    main(resume_from=args.resume)