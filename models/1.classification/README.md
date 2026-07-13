# SMT 멀티모달 분류 모델 실행 가이드

이 문서는 SMT 공정의 이미지 및 시계열 센서 데이터를 활용하여 양/불량을 판정하는 멀티모달 분류 모델(`1.classification`)의 실행 과정을 단계별로 설명합니다.

## 1. 환경 설정

본 모델은 `requirements.txt`에 명시된 의존성을 필요로 합니다.
제공된 가상환경(`.venv_cls`)을 활성화하거나 다음 명령어로 패키지를 설치하십시오.

```bash
cd f:/workspace1/AI_Workspace/models/1.classification
pip install -r requirements.txt
```

## 2. 데이터 전처리 (Preprocessing)

원시 데이터셋(`dataset/1.basedata`)에서 학습에 필요한 이미지와 센서 데이터를 추출하고 병합하기 위해 전처리를 수행합니다.

```bash
python process.py
```
- **역할**: 분리된 `train`, `val`, `test` 폴더를 스캔하여 `.json`에서 5-step 센서 데이터를 `.npy` 형태로 저장하고, 모델이 읽을 수 있는 통합 `metadata.csv`를 생성합니다.
- **결과물**: `processed_data/` 폴더 내에 `train/`, `val/`, `test/` 하위 폴더와 통계 보고서(`dataset_statistics.txt`)가 생성됩니다.

## 3. 모델 학습 (Training)

전처리된 데이터를 바탕으로 Swin Transformer V2 (이미지)와 TimesNet (센서)을 결합한 멀티모달 모델을 학습시킵니다.

```bash
python train.py
```
- **역할**: 설정된 기본 파라미터(Batch Size: 32, Epochs: 50)에 따라 모델을 학습하며, 검증 손실(Validation Loss) 기준 Early Stopping이 적용되어 있습니다.
- **체크포인트 이어서 학습 (선택)**: 기존에 중단된 학습을 이어가려면 `--resume` 옵션을 사용합니다.
  ```bash
  python train.py --resume results_train/experiment_.../checkpoints/last_checkpoint.pth
  ```
- **결과물**: `results_train/` 폴더 내에 실험별로 최고 성능 모델 가중치(`best_model.pth`), 마지막 체크포인트, 학습 곡선 등이 저장됩니다.

## 4. 모델 평가 (Evaluation)

학습 완료 후, `test` 데이터셋을 사용하여 모델의 성능을 최종 평가합니다.

```bash
python evaluate.py
```
- **역할**: `results_train/`에서 가장 최근에 학습된 `best_model.pth`를 자동으로 불러와 테스트 셋에 대한 평가를 진행합니다.
- **결과물**: `evaluation_results/` 폴더 내에 아래 평가 항목들이 저장됩니다.
  - 종합 평가 리포트 (`evaluation_summary.txt`)
  - 혼동 행렬 (Confusion Matrix)
  - ROC Curve 등 평가 지표

## 5. GUI 시각화 앱 실행 (Inference App)

모델이 개별 샘플에 대해 어떻게 예측하는지 직관적으로 시각화하여 확인하려면 제공된 윈도우 앱을 실행합니다.

```bash
python Win_app.py
```
- **역할**: Matplotlib 기반의 시각화 도구가 열리며, 입력 제품 이미지와 5-Step 시계열 센서 데이터 변화량 그래프, 그리고 모델의 AI 예측 확률 바 차트를 보여줍니다. 
- **기능**: UI 상의 버튼을 통해 이전 샘플, 다음 샘플, 랜덤 샘플링 등을 직관적으로 살펴볼 수 있습니다.
