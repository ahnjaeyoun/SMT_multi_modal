import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from PIL import Image

# 1. 데이터를 PyTorch에 올려주는 클래스
class MultimodalDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        sensor = np.load(row['sensor_npy']) # (15, 5) 데이터 로드
        label = row['label']

        if self.transform: 
            img = self.transform(img)
            
        sensor = torch.tensor(sensor).float().transpose(0, 1) # (5, 15)로 차원 변경
        return img, sensor, torch.tensor(label).float()

# 2. 하이브리드(멀티모달) AI 모델 아키텍처
class GEMSMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        # 비전: Swin Transformer V2
        self.swin = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        self.swin.head = nn.Identity() # 맨 마지막 층 떼어내기 (특징만 추출)
        
        # 센서: 1D CNN
        self.sensor_net = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 합치기 (Fusion)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # 1개의 값 출력 (0에 가까우면 정상, 1에 가까우면 불량)
        )

    def forward(self, img, sensor):
        img_f = self.swin(img)
        sen_f = self.sensor_net(sensor)
        fused = torch.cat((img_f, sen_f), dim=1) # 데이터 이어붙이기
        out = self.classifier(fused)
        return out.squeeze(1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 학습 장치: {device}")

    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(MultimodalDataset("dataset_csv/train.csv", transform), batch_size=16, shuffle=True)
    val_loader = DataLoader(MultimodalDataset("dataset_csv/val.csv", transform), batch_size=16, shuffle=False)

    model = GEMSMultimodal().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    epochs = 15
    best_loss = float('inf')

    print("🔥 학습을 시작합니다!")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, sensors, labels in train_loader:
            imgs, sensors, labels = imgs.to(device), sensors.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, sensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, sensors, labels in val_loader:
                imgs, sensors, labels = imgs.to(device), sensors.to(device), labels.to(device)
                outputs = model(imgs, sensors)
                val_loss += criterion(outputs, labels).item()
                
        val_loss /= len(val_loader)
        print(f"에포크 [{epoch+1}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_multimodal.pt")
            print("  => 💾 가중치 저장 완료 (best_multimodal.pt)")

if __name__ == "__main__":
    main()