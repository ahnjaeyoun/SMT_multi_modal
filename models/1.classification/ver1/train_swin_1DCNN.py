# ==============================================================================
# 🚀 [학습 속도 가속을 위한 핵심 튜닝 파라미터 안내]
# 아래 파라미터들을 조절하여 모델의 학습 속도와 성능(정확도)을 트레이드오프 할 수 있습니다.
#
# 1. 이미지 사이즈 (IMG_SIZE) - SMTMultimodalDataset 클래스 내부
#    - 현재값: 160 (안정형 타협점) / 원본: 224
#
# 2. 센서 모델 스펙 (Config 클래스) 
#    - 현재값: d_model=256, d_ff=512, e_layers=2 (절반 다이어트)
#
# 3. 그래디언트 누적 (accumulation_steps) - train_model 함수 내부
#    - 현재값: 1 (누적 없음) 
#
# 4. 검증 주기 (eval_interval) - main() 함수 하단 train_model 인자
#    - 현재값: 5 (5 에포크마다 1번 검증) 
#
# 5. 🌟 데이터 로더 최적화 (num_workers, persistent_workers) - main() 함수 내부
#    - 현재값: num_workers=4 (윈도우 멀티프로세싱 병목 현상 방지를 위해 12에서 4로 하향)
#
# 6. CuDNN 벤치마크 (torch.backends.cudnn.benchmark)
#    - 적용 완료: 최적의 Conv 알고리즘 탐색
#
# 7. 비전 모델 아키텍처 (vision_model_type) - main() 함수 내부
#    - 현재값: 'efficientnet' (최고속 CNN 유지)
#
# 8. 🌟 배치 사이즈 (batch_size) - main() 함수 내부
#    - 현재값: 32 (안정성을 위해 64에서 32로 추가 하향)
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
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
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
        # 🌟 약간의 다이어트 적용 (512 -> 256, 1024 -> 512, 4 -> 2)
        self.d_model = 256
        self.d_ff = 512
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
        
        # 🌟 224는 무겁고 128은 가벼우니, 중간 타협점인 160 사용
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
        label = torch.tensor(row['label'], dtype=torch.long)
        process_type = row['process_type']
        
        return {
            'image': image_tensor,
            'sensor_data': sensor_tensor,
            'label': label,
            'process_type': process_type
        }

# ==============================================================================
# 2. 모델 아키텍처: Vision Model + TimesNet 동적 결합
# ==============================================================================
class DualEncodingModel(nn.Module):
    def __init__(self, num_classes=2, vision_type='efficientnet'):
        super().__init__()
        self.vision_type = vision_type
        
        configs = Config()
        self.d_model = configs.d_model 
        
        if vision_type == 'swin':
            self.image_encoder = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
            self.image_encoder.head = nn.Identity()
            vision_out_dim = 768
        elif vision_type == 'efficientnet':
            self.image_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            self.image_encoder.classifier = nn.Identity()
            vision_out_dim = 1280
        else:
            raise ValueError("지원하지 않는 vision_type 입니다.")
        
        self.sensor_encoder = TimesNet(configs)
        
        self.fusion = nn.Sequential(
            nn.Linear(vision_out_dim + self.d_model, 512),
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
        
        sensor_features = reconstruction_error.unsqueeze(1).repeat(1, self.d_model)
        
        combined_features = torch.cat([image_features, sensor_features], dim=1)
        output = self.fusion(combined_features)
        return output

# ==============================================================================
# 3. 학습 및 검증 파이프라인
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir, eval_interval=5, resume_checkpoint=None):
    os.makedirs(results_dir, exist_ok=True)
    start_time = datetime.now()
    print(f"\n=== 학습 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    scaler = torch.amp.GradScaler('cuda')
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
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc='Training')):
            images = batch['image'].to(device)
            sensor_data = batch['sensor_data'].to(device)
            labels = batch['label'].to(device)
            process_types = batch['process_type']
            
            with torch.amp.autocast('cuda'):
                outputs = model(images, sensor_data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                scaled_loss = loss / accumulation_steps
            
            scaler.scale(scaled_loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
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
    
    # 🌟 안정적인 학습을 위한 배치 사이즈 32 (스와핑 원천 차단)
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0001
    
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 연산 장치: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 🌟 윈도우 프리징 방지: num_workers를 4로 유지
    train_loader = DataLoader(
        SMTMultimodalDataset(train_csv, is_train=True), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        SMTMultimodalDataset(val_csv, is_train=False), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    vision_model_type = 'efficientnet'
    model = DualEncodingModel(num_classes=2, vision_type=vision_model_type).to(device)
    
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
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir, eval_interval=5, resume_checkpoint=resume_from)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SMT 불량 검출 멀티모달 모델 학습 (TimesNet 기반)')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 파일 경로 지정 (이어하기용)')
    args = parser.parse_args()
    
    main(resume_from=args.resume)