"""
전처리 스크립트 (preprocess.py)

원본 이미지(.jpg)와 JSON(탐지 결과+센서 시계열)을 읽어
모델 학습에 필요한 형태로 변환한다.

출력:
  processed_data/
    sensors/       센서 시계열 .npy 파일
    train/metadata.csv
    val/metadata.csv

라벨 설계:
  category_id 1~16  정상 → 무시
  category_id 17~32 불량 → 16-dim 이진 벡터 (멀티라벨)
  불량이 없는 이미지는 제외

실행:
  cd models/1.classification
  python preprocess.py
"""

import sys
import json as _json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# models/1.classification/ 기준에서 dataset 폴더까지의 상대 경로
BASE_DIR    = Path("../../dataset/1.base_data")
IMAGE_DIR   = BASE_DIR / "image"   # 원본 이미지 폴더
JSON_DIR    = BASE_DIR / "json"    # 탐지 결과 + 센서 JSON 폴더
OUT_DIR     = BASE_DIR / "processed_data"  # 전처리 결과 저장 루트

SEQ_LEN     = 5   # JSON 의 sensor_sequence 에서 사용할 타임스텝 수
SENSOR_KEYS = [   # JSON 의 각 타임스텝에서 추출할 키 (순서가 채널 순서)
    "temperature",   # 온도 (°C)
    "humidity",      # 습도 (%)
    "vibration",     # 진동 (m/s²)
    "acceleration",  # 가속도 (g)
    "noise",         # 소음 (dB)
]

# category_id 17 → index 0 (미납), ..., category_id 32 → index 15 (납고드름)
DEFECT_NAMES = [
    "미납", "납부족", "납쇼트", "납볼", "납좌표밀림", "납형성불량",
    "냉납", "밀림", "쇼트", "오삽", "미삽", "역삽",
    "뒤집힘", "일어섬", "납금감/핀홀", "납고드름",
]


def parse_file(stem: str, npy_dir: Path):
    """
    파일명 stem(확장자 제외)을 받아 하나의 데이터 레코드를 생성한다.

    처리 순서:
      1. JSON 파일과 이미지 파일 존재 확인
      2. annotations 에서 불량(category_id 17~32) 추출 → 16-dim 멀티라벨
      3. sensor_sequence 에서 5채널 × 5 timestep 추출 → .npy 저장
      4. 이미지/센서 경로 + 라벨을 딕셔너리로 반환

    Args:
        stem   : 파일명 (확장자 제외), ex) "PR_DEF_MF_A_20250902-171623_01484"
        npy_dir: 센서 .npy 를 저장할 디렉터리

    Returns:
        레코드 딕셔너리 또는 None (유효하지 않은 파일)
    """
    json_path  = JSON_DIR  / f"{stem}.json"
    image_path = IMAGE_DIR / f"{stem}.jpg"

    # jpg 소문자가 없으면 대문자 JPG 시도 (파일명 케이스 대응)
    if not image_path.exists():
        image_path = IMAGE_DIR / f"{stem}.JPG"

    # 둘 중 하나라도 없으면 스킵
    if not json_path.exists() or not image_path.exists():
        return None

    # JSON 파싱
    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)

    # 불량 라벨 구성 — 16종 불량을 한 번에 표현하는 16자리 0/1 배열
    # 예) 미납 + 납쇼트 동시 발생 → [1, 0, 1, 0, 0, ..., 0]
    # category_id 17~32만 불량, 1~16은 정상이므로 정상 annotation은 무시
    label = [0] * 16
    has_defect = False
    for ann in data.get("annotations", []):
        cid = ann.get("category_id", 0)
        if 17 <= cid <= 32:
            label[cid - 17] = 1   # ex) category_id=22 → index=5 (납형성불량)
            has_defect = True

    # 불량 annotation이 하나도 없으면 (정상 이미지) 제외
    if not has_defect:
        return None

    # 센서 시계열 추출
    # JSON 구조: sensor_data[0].sensor_sequence = [{timestamp, temperature, ...}, ...]
    sensor_data = data.get("sensor_data", [])
    if not sensor_data:
        return None

    seq = sensor_data[0].get("sensor_sequence", [])
    if len(seq) < SEQ_LEN:
        # 시퀀스 길이가 SEQ_LEN 미만이면 사용 불가
        return None

    # 앞에서 SEQ_LEN 개 타임스텝만 사용, 5개 센서 채널 추출
    # 결과 shape: (SEQ_LEN, 5) = (5, 5)
    arr = np.array(
        [[step.get(k, 0.0) for k in SENSOR_KEYS] for step in seq[:SEQ_LEN]],
        dtype=np.float32,
    )

    # .npy 파일로 저장 (모델 학습 시 np.load 로 빠르게 로드)
    npy_path = npy_dir / f"{stem}.npy"
    np.save(npy_path, arr)

    # 파일명에서 공정 구분 파싱 (PR_DEF_* → "PR", SD_DEF_* → "SD")
    process = stem.split("_")[0]

    # 최종 레코드 반환 — image_path, sensor_path, process, label_* 컬럼으로 구성
    return {
        "image_path":  str(image_path.resolve()),   # 이미지 절대 경로
        "sensor_path": str(npy_path.resolve()),      # 센서 .npy 절대 경로
        "process":     process,                      # 공정 구분 (PR / SD)
        # 불량 유형별 라벨 컬럼: label_미납, label_납부족, ...
        **{f"label_{name}": label[i] for i, name in enumerate(DEFECT_NAMES)},
    }


def main():
    # 입력 폴더 유효성 검사
    for d in [IMAGE_DIR, JSON_DIR]:
        if not d.exists():
            print(f"[오류] 폴더를 찾을 수 없습니다: {d.resolve()}")
            sys.exit(1)

    # 기존 processed_data 가 있으면 초기화 (재실행 시 충돌 방지)
    if OUT_DIR.exists():
        print(f"기존 {OUT_DIR.resolve()} 폴더를 초기화합니다...")
        shutil.rmtree(OUT_DIR)

    # 센서 .npy 저장 폴더 생성
    npy_dir = OUT_DIR / "sensors"
    npy_dir.mkdir(parents=True)

    # 전체 JSON 파일 목록 수집
    stems = sorted(p.stem for p in JSON_DIR.glob("*.json"))
    print(f"총 JSON 파일: {len(stems)}개")

    # 파일별 파싱
    records = []
    skip    = 0
    for stem in stems:
        r = parse_file(stem, npy_dir)
        if r:
            records.append(r)
        else:
            skip += 1

    print(f"유효: {len(records)}개 | 제외(정상/누락/짧은 시퀀스): {skip}개")

    df = pd.DataFrame(records)

    # 불량 유형별 분포 출력
    label_cols = [f"label_{n}" for n in DEFECT_NAMES]
    print("\n[불량 유형 분포]")
    for col in label_cols:
        print(f"  {col[6:]:12s}: {int(df[col].sum()):5d}건")

    # Train / Val 분할 (80/20) — process 컬럼 기준 stratify로 PR/SD 비율 보존
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["process"]
    )

    # 센서 정규화 — train 통계로 fit, train/val 모두 transform
    # .npy 파일을 직접 덮어써서 정규화된 값으로 교체
    sensor_paths_train = train_df["sensor_path"].tolist()
    sensor_paths_val   = val_df["sensor_path"].tolist()

    # train 센서 전체 로드: (N_train, SEQ_LEN, 5) → (N_train*SEQ_LEN, 5)
    train_arrays = np.stack([np.load(p) for p in sensor_paths_train])  # (N, SEQ_LEN, 5)
    N, T, C = train_arrays.shape
    scaler = StandardScaler()
    scaler.fit(train_arrays.reshape(-1, C))   # (N*T, 5) 로 펼쳐서 채널별 통계 계산

    def normalize_and_save(paths: list, arrays: np.ndarray):
        n = arrays.shape[0]
        normalized = scaler.transform(arrays.reshape(-1, C)).reshape(n, T, C)
        for path, arr in zip(paths, normalized):
            np.save(path, arr.astype(np.float32))

    normalize_and_save(sensor_paths_train, train_arrays)

    val_arrays = np.stack([np.load(p) for p in sensor_paths_val])
    normalize_and_save(sensor_paths_val, val_arrays)

    # scaler 통계 저장 (서빙/평가 시 동일 정규화 재현용)
    scaler_stats = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "sensor_keys": SENSOR_KEYS,
    }
    with open(OUT_DIR / "scaler_stats.json", "w", encoding="utf-8") as f:
        _json.dump(scaler_stats, f, indent=2)
    print(f"\n센서 정규화 완료 (StandardScaler, train 통계 기준)")
    print(f"  mean : {[round(v, 4) for v in scaler.mean_]}")
    print(f"  scale: {[round(v, 4) for v in scaler.scale_]}")

    # 폴더 생성 및 CSV 저장 (utf-8-sig: Excel 한글 깨짐 방지)
    (OUT_DIR / "train").mkdir()
    (OUT_DIR / "val").mkdir()
    train_df.to_csv(OUT_DIR / "train" / "metadata.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUT_DIR  / "val"   / "metadata.csv", index=False, encoding="utf-8-sig")

    print(f"\ntrain: {len(train_df)}개 | val: {len(val_df)}개")
    pr_train = (train_df["process"] == "PR").sum()
    sd_train = (train_df["process"] == "SD").sum()
    pr_val   = (val_df["process"]   == "PR").sum()
    sd_val   = (val_df["process"]   == "SD").sum()
    print(f"  train — PR: {pr_train}개, SD: {sd_train}개")
    print(f"  val   — PR: {pr_val}개,  SD: {sd_val}개")
    print(f"저장 완료: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
