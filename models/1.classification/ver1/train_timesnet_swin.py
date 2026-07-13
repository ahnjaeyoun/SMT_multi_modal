# ==============================================================================
# 🚀 [학습 속도 가속을 위한 핵심 튜닝 파라미터 안내]
# 아래 파라미터들을 조절하여 모델의 학습 속도와 성능(정확도)을 트레이드오프 할 수 있습니다.
#
# 1. 이미지 사이즈 (IMG_SIZE) - SMTMultimodalDataset 클래스 내부
#    - 현재값: 128 (초고속 학습 세팅) 
#
# 2. 센서 모델 다이어트 (Config 클래스) 
#    - 현재값: d_model=128, d_ff=256, e_layers=2 (가벼운 세팅)
#
# 3. 그래디언트 누적 (accumulation_steps) - train_model 함수 내부
#    - 현재값: 1 (누적 없음)
#
# 4. 검증 주기 (eval_interval) - main() 함수 하단
#    - 현재값: 5 (5 에포크마다 1번 검증)
#
# 5. 데이터 로더 최적화 (num_workers, persistent_workers)
#    - 현재값: num_workers=12 (윈도우 환경 안정성 최대치)
#
# 6. CuDNN 벤치마크 (torch.backends.cudnn.benchmark)
#    - 적용 완료: 최적의 Conv 알고리즘 탐색
#
# 7. 🌟 [핵심 변경] 모델 출력 클래스 수 (num_classes)
#    - 현재값: 32 (정상 16종 + 불량 16종 = 총 32개 멀티 클래스 분류 완벽 지원)
# ==============================================================================

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
# 🚨 외부 모듈 로드: timesnet.py가 같은 폴더에 있어야 합니다.
# ==============================================================================
try:
    from timesnet import Model as TimesNet
except ImportError:
    print("🚨 에러: 'timesnet.py' 파일을 찾을 수 없습니다. train.py와 같은 폴더에 위치시켜 주세요.")
    raise

class Config:
    def __init__(self):
        self.task_name = 'anomaly_detection'
        self.seq_len = 5
        self.label_len = 0
        self.pred_len = 0
        self.enc_in = 5
        self.c_out = 5
        # 🌟 속도 가속 파라미터 2: TimesNet 다이어트
        self.d_model = 128
        self.d_ff = 256
        self.num_kernels = 6
        self.e_layers = 2
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.1
        self.top_k = 2

# ==============================================================================
# 1. 멀티모달 데이터셋 
# ==============================================================================
class SMTMultimodalDataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        self.df = pd.read_csv(csv_path)
        
        # 🌟 속도 가속 파라미터 1: 이미지 사이즈 대폭 축소 (224 -> 128)
        IMG_SIZE = 224
        
        if is_train:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
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
        
        # 라벨 값이 0~31로 들어옴 (CrossEntropyLoss가 자동 처리)
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
    def __init__(self, num_classes=32):
        super().__init__()
        self.image_encoder = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        self.image_encoder.head = nn.Identity()
        
        configs = Config()
        self.sensor_encoder = TimesNet(configs)
        
        # 융합 차원 업데이트: Swin(768) + TimesNet 복사본(d_model=128)
        self.fusion = nn.Sequential(
            nn.Linear(768 + 128, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes) # 여기서 32개의 뉴런으로 뻗어나감
        )

    def forward(self, image, sensor_data):
        image_features = self.image_encoder(image)
        sensor_reconstruction = self.sensor_encoder(sensor_data, None, None, None)
        reconstruction_error = torch.mean((sensor_data - sensor_reconstruction) ** 2, dim=(1, 2))
        
        # 128 차원으로 복사 (변경된 d_model에 맞춤)
        sensor_features = reconstruction_error.unsqueeze(1).repeat(1, 128)
        
        combined_features = torch.cat([image_features, sensor_features], dim=1)
        output = self.fusion(combined_features)
        return output

# ==============================================================================
# 3. 학습 및 검증 파이프라인 (Gradient Accumulation + 최신 AMP 적용)
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir, eval_interval=5, resume_checkpoint=None):
    os.makedirs(results_dir, exist_ok=True)
    start_time = datetime.now()
    print(f"\n=== 학습 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    # 최신 PyTorch API로 변경된 AMP Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # 🌟 속도 가속 파라미터 3: Gradient Accumulation 끄기 (빠른 업데이트)
    accumulation_steps = 1
    
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
            training_history = {
                'experiment_info': {'experiment_id': experiment_id, 'total_epochs': checkpoint['epoch'] + num_epochs, 'best_val_acc': best_val_acc},
                'epoch_results': []
            }
    else:
        best_val_acc = 0.0
        experiment_id = start_time.strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(results_dir, experiment_id)
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        training_history = {
            'experiment_info': {'experiment_id': experiment_id, 'total_epochs': num_epochs, 'best_val_acc': 0.0, 'best_epoch': 0},
            'epoch_results': []
        }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        process_stats = {"사전공정": {"correct": 0, "total": 0}, "납땜공정": {"correct": 0, "total": 0}}
        
        # 에포크 시작 전 기울기 초기화
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc='Training')):
            images = batch['image'].to(device)
            sensor_data = batch['sensor_data'].to(device)
            labels = batch['label'].to(device)
            process_types = batch['process_type']
            
            # 최신 PyTorch API로 변경된 autocast
            with torch.amp.autocast('cuda'):
                outputs = model(images, sensor_data)
                _, preds = torch.max(outputs, 1) # 32개 확률 중 가장 높은 번호를 선택!
                loss = criterion(outputs, labels)
                
                # Gradient Accumulation을 위해 loss를 누적 횟수만큼 나눔
                scaled_loss = loss / accumulation_steps
            
            # AMP 기반 역전파 (메모리에 차곡차곡 쌓음)
            scaler.scale(scaled_loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 🌟 처방 2: 기울기 폭발 방지 (Gradient Clipping 추가!)
                scaler.unscale_(optimizer) # unscale 먼저 실행
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 기울기가 1.0을 넘으면 깎아버림
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            
            for i, p_type in enumerate(process_types):
                process_stats[p_type]["total"] += 1
                if preds[i] == labels[i]:
                    process_stats[p_type]["correct"] += 1
                    
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        process_accuracies = {}
        for p_type, stats in process_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            process_accuracies[p_type] = {"accuracy": float(acc), "correct": stats["correct"], "total": stats["total"]}
            print(f'{p_type} Acc: {acc:.4f} ({stats["correct"]}/{stats["total"]})')
            
        epoch_result = {
            'epoch': epoch + 1,
            'train': {'loss': float(epoch_loss), 'accuracy': float(epoch_acc), 'process_stats': process_accuracies}
        }
        
        if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
            print(f"\n=== 검증 평가 (Epoch {epoch + 1}) ===")
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            val_total = 0
            val_process_stats = {"사전공정": {"correct": 0, "total": 0}, "납땜공정": {"correct": 0, "total": 0}}
            

            # 검증 전 캐시 비우기
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    images = batch['image'].to(device)
                    sensor_data = batch['sensor_data'].to(device)
                    labels = batch['label'].to(device)
                    process_types = batch['process_type']
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(images, sensor_data)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    val_corrects += torch.sum(preds == labels.data)
                    val_total += images.size(0)
                    
                    for i, p_type in enumerate(process_types):
                        val_process_stats[p_type]["total"] += 1
                        if preds[i] == labels[i]:
                            val_process_stats[p_type]["correct"] += 1
                            
            v_loss = val_loss / val_total
            v_acc = val_corrects.double() / val_total
            print(f'Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}')
            
            val_process_accuracies = {}
            for p_type, stats in val_process_stats.items():
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                val_process_accuracies[p_type] = {"accuracy": float(acc), "correct": stats["correct"], "total": stats["total"]}
                print(f'{p_type} Val Acc: {acc:.4f} ({stats["correct"]}/{stats["total"]})')
                
            epoch_result['validation'] = {'loss': float(v_loss), 'accuracy': float(v_acc), 'process_stats': val_process_accuracies}
            
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                training_history['experiment_info']['best_val_acc'] = float(best_val_acc)
                training_history['experiment_info']['best_epoch'] = epoch + 1
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"🌟 새로운 최고 성능! Val Acc: {v_acc:.4f}")
                
        training_history['epoch_results'].append(epoch_result)
        
        results_file = os.path.join(experiment_dir, f'training_history_{experiment_id}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=4, ensure_ascii=False)
            
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
    
    # 🌟 배치 사이즈 복구
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.5e-5
    
    # 캐시 비우기 (기존 찌꺼기 방지)
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 연산 장치: {device}")
    
    # 🌟 속도 가속 파라미터 6: CuDNN 벤치마크 활성화 (공짜 속도 부스트)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 🌟 윈도우 환경 병목 방지: num_workers를 20에서 8로 하향 조정 (프리징 및 OOM 방지)
    train_loader = DataLoader(
        SMTMultimodalDataset(train_csv, is_train=True), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=12, 
        pin_memory=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        SMTMultimodalDataset(val_csv, is_train=False), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    # 🌟 핵심 수정: 32개 멀티 클래스(정상 16종 + 불량 16종)로 출력 뉴런 수 완벽 확장!
    model = DualEncodingModel(num_classes=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)    
    if resume_from and os.path.exists(resume_from):
        print(f"\n🔄 체크포인트에서 이어서 학습합니다: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        if num_epochs - start_epoch <= 0: return
        num_epochs = num_epochs - start_epoch
    
    # 🌟 속도 가속 파라미터 4: eval_interval=5 (5 에포크마다 1번씩만 평가)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir, eval_interval=5, resume_checkpoint=resume_from)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SMT 불량 검출 멀티모달 모델 학습 (32 Multi-class 분류)')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 파일 경로 지정 (이어하기용)')
    args = parser.parse_args()
    
    main(resume_from=args.resume)