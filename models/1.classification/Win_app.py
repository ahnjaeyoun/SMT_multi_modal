import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
from pathlib import Path
import platform
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 모델 아키텍처만 가져오기
try:
    from train import DualEncodingModel
except ImportError:
    print("🚨 에러: train.py 모듈을 불러올 수 없습니다.")
    sys.exit(1)

class RawInferenceDataset:
    """metadata.csv(정답 라벨) 없이 이미지와 센서 폴더만 직접 스캔하여 불러오는 클래스"""
    def __init__(self, img_dir, sensor_dir):
        self.img_dir = Path(img_dir)
        self.sensor_dir = Path(sensor_dir)
        self.samples = []
        
        # 실제 모델 추론 시와 동일한 이미지 전처리 적용
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 이미지 폴더 내 모든 jpg, png 검색
        img_paths = list(self.img_dir.glob("**/*.jpg")) + list(self.img_dir.glob("**/*.png")) + list(self.img_dir.glob("**/*.JPG"))
        
        for img_path in img_paths:
            stem = img_path.stem
            
            # 매칭되는 센서 데이터 검색 (.npy 또는 .csv)
            npy_path = self.sensor_dir / f"{stem}.npy"
            csv_path = self.sensor_dir / f"{stem}.csv"
            
            sensor_path = None
            if npy_path.exists():
                sensor_path = npy_path
            elif csv_path.exists():
                sensor_path = csv_path
                
            if sensor_path:
                # 파일 이름에서 공정 유추 (PR_xxx -> 사전공정, SD_xxx -> 납땜공정)
                process = "사전공정" if stem.lower().startswith('pr') else "납땜공정"
                self.samples.append({
                    'image_path': img_path,
                    'sensor_path': sensor_path,
                    'process': process
                })
                
        print(f"✅ 폴더 스캔 완료: 매칭된 데이터 쌍(이미지+센서) 총 {len(self.samples)}개 찾음.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 로드 및 전처리
        image = np.array(Image.open(sample['image_path']).convert('RGB'))
        image_tensor = self.transform(image=image)['image']
        
        # 센서 로드
        if sample['sensor_path'].suffix == '.npy':
            sensor_data = np.load(sample['sensor_path'])
        else:
            # csv인 경우 (숫자 데이터만 추출하여 5x5 로 캐스팅)
            df = pd.read_csv(sample['sensor_path'])
            # 5x5 크기로 맞추기
            sensor_data = df.values[:5, :5].astype(np.float32)
            
        sensor_tensor = torch.FloatTensor(sensor_data)
        
        return {
            'image': image_tensor,
            'sensor_data': sensor_tensor,
            'process_type': sample['process'],
            'image_name': sample['image_path'].name
        }

class SMTVisualizerApp:
    def __init__(self):
        print("초기 설정 및 데이터 로딩 중...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_dir = Path(__file__).resolve().parent
        
        # 프로젝트 루트 탐색
        self.project_root = self.current_dir.parents[1]
        
        # =========================================================================
        # 📂 [경로 설정] 정답 목차(metadata.csv)를 무시하고, 원시 데이터 폴더를 직접 스캔
        # 테스트 이미지들이 모여있는 폴더 경로
        self.test_img_dir = self.project_root / "dataset" / "1.basedata" / "images" / "test"
        
        # 테스트 센서 데이터(.npy 또는 .csv)들이 모여있는 폴더 경로
        self.test_sensor_dir = self.current_dir / "processed_data" / "test" / "sensors"
        # =========================================================================
        
        self.model_weight_path = self.current_dir / "results_train"
        
        # 1. 데이터셋 로드 (폴더 직접 스캔)
        self.dataset = RawInferenceDataset(self.test_img_dir, self.test_sensor_dir)
        if len(self.dataset) == 0:
            print("🚨 에러: 매칭되는 이미지-센서 데이터 쌍을 찾을 수 없습니다. 폴더 경로를 확인하세요.")
            sys.exit(1)
            
        # 2. 모델 로드
        self.best_model_path = self._get_best_model_path()
        if not self.best_model_path.exists():
            print(f"🚨 에러: 모델 가중치 파일({self.best_model_path})을 찾을 수 없습니다.")
            sys.exit(1)
            
        self.model = DualEncodingModel(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ 모델 로드 완료 ({self.device} 환경)")
        
        # 센서 데이터 라벨 (순서: 온도, 습도, 진동, 가스, 소음 등)
        self.sensor_names = ['Temp(온도)', 'Humidity(습도)', 'Vibration(진동)', 'Gas(가스)', 'Noise(소음)']
        self.current_idx = 0
        
        # 3. UI 셋업 및 첫 화면 출력
        self.setup_ui()
        self.update_plot(self.current_idx)
        
    def _get_best_model_path(self):
        if self.model_weight_path.exists():
            subdirs = [d for d in self.model_weight_path.iterdir() if d.is_dir()]
            if subdirs:
                latest_exp_dir = max(subdirs, key=os.path.getmtime)
                return latest_exp_dir / "checkpoints" / "best_model.pth"
            else:
                return self.model_weight_path / "best_model.pth"
        return self.current_dir / "best_model.pth"

    def setup_ui(self):
        # Matplotlib 대시보드 형태 설정
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 스타일 적용 후 한글 폰트 다시 설정
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('SMT Defect Analysis - Inference Mode (No Ground Truth)')
        
        # 그리드 레이아웃 설정
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[1.2, 1])
        
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_sensor = self.fig.add_subplot(gs[0, 1:])
        self.ax_prob = self.fig.add_subplot(gs[1, 0])
        self.ax_info = self.fig.add_subplot(gs[1, 1:])
        self.ax_info.axis('off')
        
        # 네비게이션 버튼
        ax_prev = plt.axes([0.7, 0.03, 0.12, 0.06])
        ax_next = plt.axes([0.85, 0.03, 0.12, 0.06])
        ax_rand = plt.axes([0.55, 0.03, 0.12, 0.06])
        
        self.btn_prev = Button(ax_prev, '◀ 이전 샘플', color='lightgray')
        self.btn_prev.on_clicked(self.prev_sample)
        
        self.btn_next = Button(ax_next, '다음 샘플 ▶', color='lightgray')
        self.btn_next.on_clicked(self.next_sample)

        self.btn_rand = Button(ax_rand, '🎲 랜덤 샘플링', color='lightblue')
        self.btn_rand.on_clicked(self.random_sample)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
    def get_prediction(self, image, sensor):
        with torch.no_grad():
            img_batch = image.unsqueeze(0).to(self.device)
            sensor_batch = sensor.unsqueeze(0).to(self.device)
            
            outputs = self.model(img_batch, sensor_batch)
            probs = F.softmax(outputs, dim=1)[0]
            
            return probs.cpu().numpy()

    def update_plot(self, idx):
        # 데이터 1개 추출
        data = self.dataset[idx]
        image = data['image']
        sensor = data['sensor_data']
        process = data['process_type']
        img_name = data['image_name']
        
        # 모델 추론 진행
        probs = self.get_prediction(image, sensor)
        pred_label = np.argmax(probs)
        
        # --- 1. 이미지 업데이트 ---
        img_np = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        self.ax_img.clear()
        self.ax_img.imshow(img_np)
        self.ax_img.set_title(f"입력: {img_name}", fontweight='bold')
        self.ax_img.axis('off')
        
        # --- 2. 센서 데이터 그래프 업데이트 ---
        self.ax_sensor.clear()
        sensor_np = sensor.numpy()
        
        markers = ['o', 's', '^', 'D', '*']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i in range(5):
            if i < sensor_np.shape[1]:
                self.ax_sensor.plot(sensor_np[:, i], marker=markers[i], color=colors[i], 
                                    linewidth=2, markersize=8, label=self.sensor_names[i])
            
        self.ax_sensor.legend(loc='upper right')
        self.ax_sensor.set_title("5-Step 시계열 멀티 센서 데이터 변화량", fontweight='bold')
        self.ax_sensor.set_xlabel("Time Step (시점)")
        self.ax_sensor.set_ylabel("Sensor Value")
        self.ax_sensor.set_xticks(range(sensor_np.shape[0]))
        
        # --- 3. 판별 확률 바 차트 업데이트 ---
        self.ax_prob.clear()
        classes = ['Normal(정상)', 'Defect(불량)']
        
        # 정답 비교 없이, 모델이 가장 높게 판단한 라벨을 파란색으로 강조
        bar_colors = ['#3498db' if pred_label == i else '#bdc3c7' for i in range(2)]
        
        bars = self.ax_prob.bar(classes, probs, color=bar_colors, alpha=0.9)
        self.ax_prob.set_ylim(0, 1.1)
        self.ax_prob.set_title("AI 모델 예측 확률", fontweight='bold')
        self.ax_prob.set_ylabel("Probability")
        
        for bar in bars:
            yval = bar.get_height()
            self.ax_prob.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, 
                              f'{yval*100:.1f}%', va='bottom', ha='center', fontweight='bold')
            
        # --- 4. 요약 정보 텍스트 업데이트 (정답 비교 없음) ---
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        pred_text = 'Defect (불량) 🚨' if pred_label == 1 else 'Normal (정상) ✅'
        bg_color = '#fadbd8' if pred_label == 1 else '#e8f8f5'
        
        info_str = f"""
        [ 입력 데이터 정보 ]
        ▶ 파일명 : {img_name}
        ▶ 유추된 공정 : {process}

        """
        self.ax_info.text(0.05, 0.5, info_str, fontsize=15, va='center', ha='left', 
                         bbox=dict(facecolor=bg_color, alpha=0.9, edgecolor='gray', boxstyle='round,pad=1.5'))
        
        # 화면 갱신
        self.fig.canvas.draw_idle()

    def next_sample(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        self.update_plot(self.current_idx)
        
    def prev_sample(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.dataset)
        self.update_plot(self.current_idx)
        
    def random_sample(self, event):
        self.current_idx = np.random.randint(0, len(self.dataset))
        self.update_plot(self.current_idx)

if __name__ == "__main__":
    app = SMTVisualizerApp()
    plt.show()
