# 학습 파라미터 설정
import torch
from ultralytics import YOLO
import os
import json



training_args = {
    'data': 'data.yaml',  # 데이터셋 설정 파일 경로
    'epochs': 100,  # 전체 학습 에포크
    'batch': -1,    # 배치 크기
    'imgsz': 512,   # 입력 이미지 크기
    'patience': 50,  # Early stopping patience
    'device': 1 if torch.cuda.is_available() else 'cpu',  # GPU 사용 여부
    'workers': 8,   # 데이터 로더 워커 수
    'name': 'test_3',  # 실험 이름 (학습 ID 사용)
    'exist_ok': True,  # 기존 실험 결과 덮어쓰기 허용
    'pretrained': True,  # 사전 학습된 가중치 사용
    'optimizer': 'AdamW',  # 옵티마이저 선택
    'lr0': 0.001,  # 초기 학습률
    # 'weight_decay': 0.05,  # 가중치 감쇠
    # 'cos_lr': True,  # Cosine 학습률 스케줄러 사용
}

if __name__=='__main__':
    # freeze_support()...?
    model= YOLO('./yolo11n.pt')

    model.train(**training_args)


