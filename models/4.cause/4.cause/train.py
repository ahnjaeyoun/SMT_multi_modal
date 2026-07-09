"""
XGBoost 모델 학습 및 저장

동작 흐름:
  1. JSON 파일 파싱
  2. 공정(PR/SD) × 불량 유형별로 XGBoost 분류기 학습
  3. 학습된 모델을 result_cause/models/ 에 저장
  4. StandardScaler 를 result_cause/scalers/ 에 저장
     (analyze.py 에서 동일한 정규화를 재현하기 위해 필요)

실행:
  cd models/4.cause
  python train.py

출력 파일:
  result_cause/
  ├── skipped_files.txt           ← 파싱 제외된 파일 목록
  ├── models/
  │   ├── PR_미납.json            ← 공정_불량명 형태의 XGBoost 모델
  │   └── ...  (총 18개)
  └── scalers/
      ├── PR_scaler.pkl           ← PR 공정 StandardScaler
      └── SD_scaler.pkl           ← SD 공정 StandardScaler
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2] / "dataset" / "1.base_data"
JSON_DIR   = BASE_DIR / "json"
RESULT_DIR = Path(__file__).resolve().parent / "result_cause"

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
SENSOR_KEYS = ["temperature", "humidity", "vibration", "acceleration", "noise"]

# 공정별 불량 유형 — (불량명, category_id)
PROCESS_DEFECTS = {
    "PR": [
        ("미납",       17),  # 납이 아예 없음
        ("납부족",     18),  # 납 양이 부족함
        ("납쇼트",     19),  # 납이 옆으로 번져 단락 발생
        ("납볼",       20),  # 납이 구슬처럼 튀어오름
        ("납좌표밀림", 21),  # 납 위치가 틀어짐
        ("납형성불량", 22),  # 납 모양이 올바르지 않음
    ],
    "SD": [
        ("미납",        17),  # 납이 아예 없음
        ("납볼",        20),  # 납이 구슬처럼 튀어오름
        ("냉납",        23),  # 납이 제대로 녹지 않아 접합 불량
        ("밀림",        24),  # 부품이 밀려 위치 이탈
        ("쇼트",        25),  # 두 단자가 납으로 연결되어 단락
        ("오삽",        26),  # 잘못된 부품이 삽입됨
        ("미삽",        27),  # 부품이 삽입되지 않음
        ("역삽",        28),  # 부품이 반대 방향으로 삽입됨
        ("뒤집힘",      29),  # 부품이 뒤집혀 실장됨
        ("일어섬",      30),  # 부품 한쪽이 들려 올라감 (툼스톤)
        ("납금감/핀홀", 31),  # 납 표면에 구멍(핀홀)이 생김
        ("납고드름",    32),  # 납이 고드름처럼 뾰족하게 굳음
    ],
}

# 분석 대상 category_id 전체 집합 (포함 여부를 O(1) 로 확인하기 위해 set 사용)
ALL_DEFECT_IDS = {cid for defs in PROCESS_DEFECTS.values() for _, cid in defs}


def parse_json(path):
    """JSON 파일 하나를 읽어 공정·파일명·피처·불량 ID 집합을 반환한다.

    Returns:
        성공: (record_dict, None)
        실패: (None, 제외_이유_문자열)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, "JSON 파싱 실패 (파일 손상 또는 형식 오류)"

    process = path.stem.split("_")[0]
    if process not in PROCESS_DEFECTS:
        return None, f"공정 식별 불가 (파일명 접두사: '{process}', PR/SD 아님)"

    defect_ids = set()
    for ann in data.get("annotations", []):
        cid = ann.get("category_id", 0)
        if cid in ALL_DEFECT_IDS:
            defect_ids.add(cid)

    if not defect_ids:
        return None, "불량 어노테이션 없음 (정상 이미지 — category_id 17~32 미존재)"

    sensor_data = data.get("sensor_data", [])
    if not sensor_data:
        return None, "센서 데이터 없음 (sensor_data 키 누락 또는 빈 리스트)"

    seq = sensor_data[0].get("sensor_sequence", [])
    if not seq:
        return None, "센서 시퀀스 없음 (sensor_sequence 가 비어 있음)"

    # 센서별 평균(mean)과 최대(max) → 피처 10개
    features = []
    for key in SENSOR_KEYS:
        vals = [step[key] for step in seq if key in step]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            features.append(float(arr.mean()))
            features.append(float(arr.max()))
        else:
            features.extend([0.0, 0.0])

    return {
        "filename":   path.stem,
        "process":    process,
        "features":   features,
        "defect_ids": defect_ids,
    }, None


def train_one(X, y):
    """불량 하나에 대한 XGBoost 이진 분류기를 학습하고 반환한다.

    Args:
        X: 정규화된 센서 피처 행렬 (N × 10)
        y: 이진 레이블 (1=해당 불량 있음, 0=없음)
    """
    pos_count = int(y.sum())
    neg_count = int((y == 0).sum())

    # 불량 샘플이 정상 샘플보다 훨씬 적으면 모델이 항상 "정상"으로만 예측할 위험이 있음
    # scale_pos_weight 로 불량 샘플에 더 높은 가중치를 부여해 균형을 맞춤
    scale_pos_weight = max(1, neg_count / pos_count)

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        base_score=0.5,  # XGBoost 3.x + SHAP 호환성 버그 우회
    )
    clf.fit(X, y)
    return clf


def main():
    if not JSON_DIR.exists():
        print(f"[오류] JSON 폴더를 찾을 수 없습니다: {JSON_DIR}")
        return

    (RESULT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "scalers").mkdir(parents=True, exist_ok=True)

    # ── 1. JSON 파싱 ───────────────────────────────────────────────────
    json_files = sorted(JSON_DIR.glob("*.json"))
    print(f"JSON 파일 {len(json_files)}개 파싱 중...")

    records = {"PR": [], "SD": []}
    skipped = []

    for path in json_files:
        record, reason = parse_json(path)
        if record:
            records[record["process"]].append(record)
        else:
            skipped.append((path.name, reason))

    print(f"  PR: {len(records['PR'])}개 | SD: {len(records['SD'])}개 | 제외: {len(skipped)}개\n")

    skip_path = RESULT_DIR / "skipped_files.txt"
    with open(skip_path, "w", encoding="utf-8") as f:
        f.write(f"제외된 파일 목록 (총 {len(skipped)}개)\n")
        f.write("=" * 70 + "\n")
        for name, reason in skipped:
            f.write(f"{name}  # {reason}\n")
    print(f"제외 파일 목록 저장: {skip_path}\n")

    # ── 2. 공정별 학습 ─────────────────────────────────────────────────
    for process, defect_list in PROCESS_DEFECTS.items():
        recs = records[process]
        if not recs:
            print(f"[경고] {process} 공정 데이터 없음 — 스킵")
            continue

        X_all = np.array([r["features"] for r in recs], dtype=np.float32)

        # StandardScaler 학습 후 저장
        # analyze.py 에서 동일한 스케일러로 변환해야 SHAP 결과가 일관성을 유지함
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        scaler_path = RESULT_DIR / "scalers" / f"{process}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"[{process}] 스케일러 저장: {scaler_path}")

        print(f"{'='*55}")
        print(f"  [{process}] 공정 모델 학습  (샘플 {len(recs)}개)")
        print(f"{'='*55}")

        for defect_name, cid in defect_list:
            y = np.array(
                [1 if cid in r["defect_ids"] else 0 for r in recs], dtype=np.int32
            )
            pos_count = int(y.sum())

            if pos_count < 2:
                print(f"  {defect_name:12s}: 불량 샘플 부족 ({pos_count}건) — 스킵")
                continue

            clf = train_one(X_scaled, y)

            # 파일명에 쓸 수 없는 '/' 를 '_' 로 치환
            safe_name  = defect_name.replace("/", "_")
            model_path = RESULT_DIR / "models" / f"{process}_{safe_name}.json"
            clf.save_model(str(model_path))

            print(f"  {defect_name:12s}: {pos_count:4d}건 학습 완료 → {model_path.name}")

        print()

    print("모든 모델 학습 및 저장 완료.")
    print(f"  모델 위치:     {RESULT_DIR / 'models'}")
    print(f"  스케일러 위치: {RESULT_DIR / 'scalers'}")


if __name__ == "__main__":
    main()
