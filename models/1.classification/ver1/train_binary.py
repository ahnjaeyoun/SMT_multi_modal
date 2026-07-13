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
# 🚨 [중요] 외부 의존성 로드 (레퍼런스 코드 기준)
# 이 스크립트가 실행되는 폴더 안에 timesnet.py 가 반드시 있어야 합니다.
# ==============================================================================
try:
    from timesnet import Model as TimesNet
except ImportError:
    print("🚨 에러: 'timesnet.py' 파일을 찾을 수 없습니다. train.py와 같은 폴더에 위치시켜 주세요.")
    raise

class Config:
    """TimesNet 초기화를 위한 설정 클래스 (회원님 스펙 그대로 유지)"""
    def __init__(self):
        self.task_name = 'anomaly_detection'
        self.seq_len = 5
        self.label_len = 0
        self.pred_len = 0
        self.enc_in = 5
        self.c_out = 5
        self.d_model = 512
        self.d_ff = 1024
        self.num_kernels = 6
        self.e_layers = 4
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.1
        self.top_k = 2

# ==============================================================================
# 1. 멀티모달 데이터셋 클래스 (해상도 384 유지)
# ==============================================================================
class SMTMultimodalDataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        self.df = pd.read_csv(csv_path)
        
        if is_train:
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row['image_path']).convert('RGB'))
        image_tensor = self.transform(image=image)['image']
        sensor_data = np.load(row['sensor_path'])
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
# 2. 모델 아키텍처: Swin V2 + TimesNet
# ==============================================================================
class DualEncodingModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.image_encoder = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        self.image_encoder.head = nn.Identity()
        
        configs = Config()
        self.sensor_encoder = TimesNet(configs)
        
        self.fusion = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, sensor_data):
        image_features = self.image_encoder(image)
        sensor_reconstruction = self.sensor_encoder(sensor_data, None, None, None)
        reconstruction_error = torch.mean((sensor_data - sensor_reconstruction) ** 2, dim=(1, 2))
        sensor_features = reconstruction_error.unsqueeze(1).repeat(1, 512)
        combined_features = torch.cat([image_features, sensor_features], dim=1)
        output = self.fusion(combined_features)
        return output

# ==============================================================================
# 3. 학습 파이프라인 (OneCycleLR, Early Stopping, Gradient Clipping 적용)
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir, eval_interval=1, resume_checkpoint=None):
    os.makedirs(results_dir, exist_ok=True)
    start_time = datetime.now()
    print(f"\n=== 학습 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    # 🌟 최적화 1: AMP Scaler (메모리 절약 및 속도 향상)
    scaler = torch.amp.GradScaler('cuda')
    
    # 🌟 최적화 2: OneCycleLR 스케줄러 (가속도 전략)
    # 초기에는 학습률을 빠르게 올려서 수렴 속도를 극대화하고 후반에 안정화시킵니다.
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3, # 최대 학습률 지정
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs
    )

    # 🌟 최적화 3: Early Stopping 변수 세팅
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 5  # 5 에포크 동안 Loss 개선이 없으면 강제 종료
    trigger_times = 0

    experiment_id = start_time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(results_dir, experiment_id)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Epoch 루프 시작
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
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
            
            # AMP 자동 혼합 정밀도 캐스팅
            with torch.amp.autocast('cuda'):
                outputs = model(images, sensor_data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # 🌟 최적화 4: Gradient Clipping (기울기 폭발 방지)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Clipping을 위해 미리 unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # 매 배치마다 OneCycleLR 스텝 업데이트
            
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # ========================================================
        # 검증 (Validation) & Early Stopping 로직
        # ========================================================
        if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
            print(f"=== 검증 평가 (Epoch {epoch + 1}) ===")
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    images = batch['image'].to(device)
                    sensor_data = batch['sensor_data'].to(device)
                    labels = batch['label'].to(device)
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(images, sensor_data)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    val_corrects += torch.sum(preds == labels.data)
                    val_total += images.size(0)
            
            v_loss = val_loss / val_total
            v_acc = val_corrects.double() / val_total
            print(f'Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}')
            
            # 🌟 Early Stopping 판단 및 Best Model 저장
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_val_acc = v_acc
                trigger_times = 0
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"🌟 새로운 최고 성능 갱신! (Val Loss: {v_loss:.4f}) 모델을 저장합니다.")
            else:
                trigger_times += 1
                print(f"⚠️ 성능 개선 없음 ({trigger_times}/{patience})")
                if trigger_times >= patience:
                    print(f"🚨 {patience} 에포크 연속 개선이 없어 Early Stopping을 발동합니다. 학습 조기 종료!")
                    break # 에포크 루프 완전 탈출
                    
        # 매 에포크마다 안전하게 진행상황 저장
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'experiment_id': experiment_id,
            'best_val_acc': best_val_acc
        }, os.path.join(checkpoint_dir, 'last_checkpoint.pth'))
        print()

    end_time = datetime.now()
    print(f"\n=== 학습 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"총 소요 시간: {end_time - start_time}")

def main(resume_from=None):
    train_csv = Path("processed_data/train/metadata.csv")
    val_csv = Path("processed_data/val/metadata.csv")
    
    if not train_csv.exists() or not val_csv.exists():
        print("🚨 processed_data 안에 metadata.csv를 찾을 수 없습니다.")
        return

    results_dir = "results_train"
    
    # 파라미터 세팅
    batch_size = 16 
    num_epochs = 50
    learning_rate = 1e-4 # (스케줄러가 알아서 조절하므로 초기값 역할)
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 연산 장치: {device}")
    
    train_loader = DataLoader(
        SMTMultimodalDataset(train_csv, is_train=True), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        SMTMultimodalDataset(val_csv, is_train=False), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # num_classes=2 세팅
    model = DualEncodingModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n🔄 체크포인트에서 이어서 학습합니다: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if num_epochs - start_epoch <= 0: return
        num_epochs = num_epochs - start_epoch
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir, eval_interval=1, resume_checkpoint=resume_from)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SMT 불량 검출 멀티모달 모델 최적화 학습')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 파일 경로 지정 (이어하기용)')
    args = parser.parse_args()
    
    main(resume_from=args.resume)