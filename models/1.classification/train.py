"""
SMT 공정 불량 원인 분석 모델 학습 스크립트.

SMTCauseAnalyzer(EfficientNet-B3 + PatchTST)를 학습시킨다.
16개 불량 유형 각각을 독립적인 이진 문제로 다루는 멀티라벨 분류 방식이며,
BCEWithLogitsLoss + AdamW + CosineAnnealingLR + AMP 조합을 사용한다.

출력:
  dataset/1.base_data/result/<timestamp>/checkpoints/best_model.pth  val Macro F1 최고 가중치
  dataset/1.base_data/result/<timestamp>/checkpoints/last.pth        이어서 학습할 때 사용
  dataset/1.base_data/result/<timestamp>/history.csv                 에포크별 지표

실행:
  python train.py
  python train.py --resume ../../dataset/1.base_data/result/<exp_id>/checkpoints/last.pth
"""

import argparse
import contextlib
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Transformer 내부에서 발생하는 비필수 경고 억제 (성능·결과에 무관한 구현 경고)
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

from model import SMTCauseAnalyzer, DEFECT_NAMES

IMG_SIZE     = 224   # EfficientNet-B3 권장 입력 크기
BATCH_SIZE   = 8     # 배치 크기 (CPU 환경 메모리 부족으로 8로 설정)
NUM_EPOCHS   = 50    # 전체 학습 에포크 수
LR           = 1e-4  # 초기 학습률 (AdamW)
WEIGHT_DECAY = 1e-2  # L2 정규화 (과적합 방지)
NUM_WORKERS  = 0     # Windows CPU 환경에서는 0 권장 (멀티프로세싱 충돌 방지)
EVAL_EVERY   = 5     # 몇 에포크마다 검증을 수행할지

# metadata.csv 의 라벨 컬럼명 목록 (model.py 의 DEFECT_NAMES 와 순서 일치)
LABEL_COLS = [f"label_{n}" for n in DEFECT_NAMES]


# 데이터셋
class SMTDataset(Dataset):
    """
    preprocess.py 가 생성한 metadata.csv 를 읽어 모델 입력 형태로 변환한다.

    반환 형태:
      image : (3, IMG_SIZE, IMG_SIZE)  — 정규화된 이미지 텐서
      sensor: (5, 5)                   — 센서 시계열 (SEQ_LEN=5, 채널=5)
      label : (16,)                    — 불량 유형 멀티라벨 (float, 0 또는 1)

    데이터 증강 (학습 시만 적용):
      - HorizontalFlip  : SMT 보드는 좌우 대칭 구조이므로 무방
      - VerticalFlip    : 상하 반전도 허용 (회전 불변 특성 강화)
      - BrightnessContrast: 조명 변화에 강인하게
      - Affine           : 약간의 이동·확대축소·회전 변화에 강인하게 (ShiftScaleRotate 대체)
    """

    def __init__(self, csv_path: str, is_train: bool = True):
        self.df = pd.read_csv(csv_path)

        if is_train:
            # 학습용: 데이터 증강 포함
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.Affine(
                    translate_percent=0.05,  # 이미지 크기의 5% 이내 이동
                    scale=(0.9, 1.1),        # 10% 이내 확대/축소
                    rotate=(-15, 15),        # ±15도 회전
                    p=0.3,
                ),
                # ImageNet 통계로 정규화 (EfficientNet 사전학습 때 사용된 값)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # 검증용: 증강 없이 리사이즈 + 정규화만
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # 이미지 로드 및 변환
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        image = self.transform(image=image)["image"]        # (3, H, W)

        # 센서 시계열 로드: .npy → (SEQ_LEN=5, 채널=5)
        sensor = torch.FloatTensor(np.load(row["sensor_path"]))

        # 멀티라벨 벡터 (16,): CSV 에서 label_미납, label_납부족, ... 컬럼 읽기
        label = torch.FloatTensor(row[LABEL_COLS].values.astype(float))

        return image, sensor, label


# 평가 지표 계산
def calc_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    멀티라벨 분류 지표를 계산한다.

    Args:
        logits   : (N, 16) — 모델 출력 (sigmoid 적용 전 logit)
        labels   : (N, 16) — 정답 라벨 (0 또는 1)
        threshold: 양성(불량) 판정 임계값

    Returns:
        exact_match: 16개 라벨이 모두 일치한 샘플 비율 (엄격한 지표)
        macro_f1   : 16개 클래스 F1 의 평균 (클래스 불균형에 robust)

    계산 방식:
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 × Precision × Recall / (Precision + Recall)
        ε=1e-8 로 0 나누기 방지
    """
    # logit → 이진 예측
    preds = (torch.sigmoid(logits) >= threshold).float()

    # Exact Match: 16개 라벨 전부 맞아야 1
    exact_match = (preds == labels).all(dim=1).float().mean().item()

    # 라벨별(dim=0) TP/FP/FN 계산
    tp = (preds * labels).sum(dim=0)                    # (16,)
    fp = (preds * (1 - labels)).sum(dim=0)              # (16,)
    fn = ((1 - preds) * labels).sum(dim=0)              # (16,)

    precision = tp / (tp + fp + 1e-8)                   # (16,)
    recall    = tp / (tp + fn + 1e-8)                   # (16,)
    f1_per    = 2 * precision * recall / (precision + recall + 1e-8)  # (16,)
    macro_f1  = f1_per.mean().item()

    return exact_match, macro_f1


# 학습 루프
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    한 에포크 동안 학습을 수행한다.

    AMP(Automatic Mixed Precision) 사용:
      - torch.amp.autocast: FP16 연산으로 속도 향상
      - GradScaler: FP16 에서 발생하는 gradient underflow 방지

    Returns:
        avg_loss   : 에포크 평균 손실
        exact_match: Exact Match Accuracy
        macro_f1   : Macro F1
    """
    model.train()  # Dropout 활성화, BatchNorm 학습 통계 사용 — eval()과 반대
    total_loss = 0.0
    all_logits, all_labels = [], []  # 에포크 전체 결과를 모아 지표를 한 번에 계산

    for images, sensors, labels in tqdm(loader, desc="  Train", leave=False):
        # 데이터를 GPU(또는 CPU)로 이동 — 모델과 같은 장치에 있어야 연산 가능
        images  = images.to(device)
        sensors = sensors.to(device)
        labels  = labels.to(device)

        # 이전 배치의 기울기를 초기화 — PyTorch는 기울기를 누적하므로 매 배치 시작 시 반드시 필요
        optimizer.zero_grad()

        # AMP: CUDA 환경에서는 FP16 연산으로 속도 향상, CPU에서는 일반 FP32 사용
        amp_ctx = torch.amp.autocast("cuda") if torch.cuda.is_available() else contextlib.nullcontext()
        with amp_ctx:
            logits, _ = model(images, sensors)   # _ 는 센서 기여도(학습 시 미사용)
            loss = criterion(logits, labels)

        # GradScaler가 비활성화(enabled=False)된 CPU 환경에서도 API 형태는 동일하게 유지됨
        # → CUDA/CPU 분기 없이 같은 코드로 실행 가능 (CPU에서는 scale 값이 1.0으로 고정)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 배치 손실 누적 (샘플 수 가중치)
        total_loss += loss.item() * images.size(0)

        # 지표 계산을 위해 logit/라벨 저장 (CPU 로 이동하여 GPU 메모리 절약)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

    # 전체 에포크 logit/라벨 합산 후 지표 계산
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    exact, f1  = calc_metrics(all_logits, all_labels)

    return total_loss / len(loader.dataset), exact, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    검증 데이터로 모델 성능을 평가한다.
    @torch.no_grad() 로 gradient 계산 비활성화 → 메모리 절약 + 속도 향상

    Returns:
        avg_loss   : 검증 평균 손실
        exact_match: Exact Match Accuracy
        macro_f1   : Macro F1
    """
    model.eval()  # BatchNorm·Dropout 등을 평가 모드로 전환 (학습 때와 동작이 달라짐)
    total_loss = 0.0
    all_logits, all_labels = [], []  # 전체 배치 결과를 모아서 한 번에 지표 계산하기 위해 리스트로 수집

    for images, sensors, labels in tqdm(loader, desc="  Val  ", leave=False):
        images  = images.to(device)
        sensors = sensors.to(device)
        labels  = labels.to(device)

        # AMP: CUDA 환경에서는 FP16 연산으로 속도 향상, CPU에서는 일반 FP32 사용
        amp_ctx = torch.amp.autocast("cuda") if torch.cuda.is_available() else contextlib.nullcontext()
        with amp_ctx:
            logits, _ = model(images, sensors)   # _ 는 센서 기여도(평가 시 미사용)
            loss = criterion(logits, labels)

        # 배치 손실을 샘플 수 기준으로 누적 (나중에 전체 평균을 내기 위해 샘플 수를 곱함)
        total_loss += loss.item() * images.size(0)

        # GPU 메모리 절약을 위해 CPU로 옮긴 뒤 리스트에 추가
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    # 배치별로 쌓아둔 리스트를 하나의 텐서로 합침 (N, 16) 형태
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    exact, f1  = calc_metrics(all_logits, all_labels)  # 전체 데이터 기준 지표 계산

    # 샘플 수로 나눠 평균 손실을 구하고 지표와 함께 반환
    return total_loss / len(loader.dataset), exact, f1


# 메인
def main(resume: str = None):
    """
    전체 학습 파이프라인을 실행한다.
    resume 경로가 주어지면 체크포인트에서 이어서 학습하고,
    없으면 새 실험 디렉터리를 생성해 처음부터 학습한다.
    """
    # 전처리 결과 파일 확인
    base = Path(__file__).parent.parent.parent / "dataset" / "1.base_data" / "processed_data"
    train_csv = base / "train" / "metadata.csv"
    val_csv   = base / "val"   / "metadata.csv"
    if not train_csv.exists() or not val_csv.exists():
        print("[오류] metadata.csv 가 없습니다. 먼저 preprocess.py 를 실행하세요.")
        return

    # 학습 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device}")

    # CuDNN 벤치마크: 고정 크기 입력일 때 가장 빠른 Conv 알고리즘 자동 탐색
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # DataLoader 생성
    train_loader = DataLoader(
        SMTDataset(train_csv, is_train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,                  # 학습: 매 에포크 랜덤 순서
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),  # CPU→GPU 전송 속도 향상 (CUDA 시에만)
        persistent_workers=False,      # num_workers=0 이면 반드시 False
    )
    val_loader = DataLoader(
        SMTDataset(val_csv, is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,                 # 검증: 순서 고정
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    # 모델 / 손실 / 옵티마이저 / 스케줄러 초기화
    model = SMTCauseAnalyzer().to(device)

    # BCEWithLogitsLoss: 16개 불량 유형 각각을 "있다/없다" 이진 문제로 독립 처리
    # sigmoid + BCE를 하나로 합쳐 수치적으로 안정적으로 계산 (직접 sigmoid 후 BCE보다 안전)
    criterion = nn.BCEWithLogitsLoss()

    # AdamW: L2 정규화를 weight 에만 정확히 적용 (bias 제외)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingLR: 학습률을 코사인 곡선 모양으로 서서히 줄임
    # 초반엔 크게 움직여 빠르게 수렴하고, 후반엔 작게 움직여 정밀하게 조정
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # GradScaler: CUDA 환경에서만 FP16 underflow 방지 역할 / CPU에서는 비활성화됨
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # 실험 디렉터리 설정
    start_epoch = 0
    best_val_f1 = 0.0
    exp_id      = datetime.now().strftime("%Y%m%d_%H%M%S")  # ex: 20250623_143022
    exp_dir     = Path(__file__).parent.parent.parent / "dataset" / "1.base_data" / "result" / exp_id
    ckpt_dir    = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 체크포인트에서 재개
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])          # 모델 가중치 복원
        optimizer.load_state_dict(ckpt["optimizer"])  # 옵티마이저 상태 복원
        scheduler.load_state_dict(ckpt["scheduler"])  # 스케줄러 상태 복원
        start_epoch = ckpt["epoch"]                   # 이어서 시작할 에포크
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        exp_id      = ckpt.get("exp_id", exp_id)      # 재개 시 새 폴더가 아닌 원래 실험 폴더에 이어서 저장
        exp_dir     = Path(__file__).parent.parent.parent / "dataset" / "1.base_data" / "result" / exp_id
        ckpt_dir    = exp_dir / "checkpoints"
        print(f"체크포인트 로드: epoch {start_epoch} | best F1 {best_val_f1:.4f}")

    history = []  # 에포크별 지표 기록 (history.csv 용) — 학습이 중단돼도 복기할 수 있도록 매 에포크 직후 저장

    # 학습 루프
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}  lr={current_lr:.2e}")

        # 학습 1 에포크
        tr_loss, tr_exact, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        # 스케줄러로 학습률 갱신 (에포크 종료 후)
        scheduler.step()

        elapsed = time.time() - epoch_start
        remaining = elapsed * (NUM_EPOCHS - epoch - 1)
        print(f"  Train  loss={tr_loss:.4f}  exact={tr_exact:.4f}  F1={tr_f1:.4f}"
              f"  [{elapsed:.0f}s 경과 | 잔여 {remaining/60:.1f}분]")

        # 에포크 기록 초기화
        row = {
            "epoch":       epoch + 1,
            "train_loss":  tr_loss,
            "train_exact": tr_exact,
            "train_f1":    tr_f1,
        }

        # EVAL_EVERY 주기마다 검증하되, 마지막 에포크는 주기와 무관하게 반드시 검증
        # (EVAL_EVERY 배수가 아닌 에포크에서 학습이 끝날 때 검증 누락 방지)
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            vl_loss, vl_exact, vl_f1 = evaluate(
                model, val_loader, criterion, device
            )
            print(f"  Val    loss={vl_loss:.4f}  exact={vl_exact:.4f}  F1={vl_f1:.4f}")
            row.update({
                "val_loss":  vl_loss,
                "val_exact": vl_exact,
                "val_f1":    vl_f1,
            })

            # val F1 이 이전 최고보다 높으면 best 모델 저장
            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                print(f"  ✅ best 갱신 → F1={best_val_f1:.4f}")

        # 에포크 기록 누적 후 CSV 즉시 저장 (학습 중 확인 가능)
        history.append(row)
        pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)

        # 마지막 체크포인트 저장 (이어서 학습할 때 사용)
        torch.save({
            "epoch":       epoch + 1,         # 다음 재개 시 이 에포크부터 시작
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "exp_id":      exp_id,            # 같은 실험 폴더에 이어서 저장하기 위해
        }, ckpt_dir / "last.pth")

    best_ckpt = ckpt_dir / "best_model.pth"
    print(f"\n학습 완료 | best val F1: {best_val_f1:.4f}")
    print(f"결과 저장 위치: {exp_dir.resolve()}")
    print(f"\n평가 실행 명령어:")
    print(f"  python evaluate.py --checkpoint {best_ckpt}")


# 스크립트 직접 실행 시 진입점 — python train.py [--resume ...] 형태로 호출
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMT 불량 원인 분석 모델 학습")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="이어서 학습할 체크포인트 경로 (ex: ../../dataset/1.base_data/result/20250623_143022/checkpoints/last.pth)",
    )
    args = parser.parse_args()
    main(resume=args.resume)
