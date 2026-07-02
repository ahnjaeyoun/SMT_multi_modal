"""
SHAP 기반 센서 영향도 분석 (분석 전용)

사전 조건:
  train.py 를 먼저 실행해서 아래 파일들이 생성되어 있어야 합니다.
    result_cause/models/PR_미납.json  등 공정×불량별 XGBoost 모델
    result_cause/scalers/PR_scaler.pkl  등 공정별 StandardScaler

동작 흐름:
  1. JSON 파일 파싱
  2. result_cause/scalers/ 에서 공정별 StandardScaler 로드
  3. result_cause/models/ 에서 공정×불량별 XGBoost 모델 로드
  4. SHAP TreeExplainer 로 양성(불량 있음) 샘플의 센서별 기여도 산출
  5. 결과를 result_cause/PR|SD/<불량명>.json 과 summary.json 으로 저장

실행:
  cd models/4.cause
  python train.py    # 먼저 학습
  python analyze.py  # 그 다음 분석

출력 파일:
  dataset/1.base_data/result_cause/
  ├── summary.json          ← 공정×불량 전체 요약
  ├── skipped_files.txt     ← 파싱 제외 파일 목록
  ├── PR/
  │   ├── 미납.json         ← 샘플별 SHAP 포함
  │   └── ...
  └── SD/
      └── ...
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import shap
import xgboost as xgb

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
RESULT_DIR = Path(__file__).resolve().parent / "result_cause"
JSON_DIR   = RESULT_DIR / "json"
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


def normalize_sensor(raw):
    """피처 10개의 |SHAP| 배열을 센서 5개의 정규화된 중요도 딕셔너리로 변환한다.

    피처 구조: [temp_mean, temp_max, hum_mean, hum_max, ...]
    센서 i 의 기여도 = (raw[i*2] + raw[i*2+1]) / 전체합
    """
    total = raw.sum()
    if total == 0:
        return {}

    norm = raw / total
    importance = {}
    for i, sensor in enumerate(SENSOR_KEYS):
        base = i * 2
        importance[sensor] = round(float(norm[base] + norm[base + 1]), 4)

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def run_shap(clf, X, y, filenames):
    """저장된 모델로 SHAP 분석을 수행하고 요약·샘플별 중요도를 반환한다.

    Args:
        clf:       train.py 에서 저장한 XGBoost 모델 (이미 학습 완료)
        X:         공정별 StandardScaler 로 정규화된 센서 피처 (N × 10)
        y:         이진 레이블 (1=해당 불량 있음, 0=없음)
        filenames: 각 샘플의 파일명
    """
    pos_mask  = y == 1
    pos_count = int(pos_mask.sum())
    if pos_count == 0:
        return None

    # 저장된 모델을 그대로 사용 — 추가 학습 없음
    explainer = shap.TreeExplainer(clf)

    # 불량이 실제로 발생한 샘플에 대해서만 SHAP 계산
    X_pos     = X[pos_mask]
    shap_vals = explainer.shap_values(X_pos)  # shape: (불량샘플수, 10)

    # 전체 요약: 불량 샘플들의 |SHAP| 평균 → 센서 5개 기여도
    mean_abs_all       = np.abs(shap_vals).mean(axis=0)
    summary_importance = normalize_sensor(mean_abs_all)
    if not summary_importance:
        return None

    # 샘플 하나하나에 대해 센서 중요도를 따로 계산
    pos_filenames = [fn for fn, flag in zip(filenames, y) if flag == 1]
    samples = []

    for fn, sv in zip(pos_filenames, shap_vals):
        sample_importance = normalize_sensor(np.abs(sv))
        if not sample_importance:
            continue
        critical = next(iter(sample_importance))
        samples.append({
            "filename":          fn,
            "critical_sensor":   critical,
            "sensor_importance": sample_importance,
        })

    return {
        "summary_importance": summary_importance,
        "samples":            samples,
    }


def main():
    if not JSON_DIR.exists():
        print(f"[오류] JSON 폴더를 찾을 수 없습니다: {JSON_DIR}")
        return

    models_dir  = RESULT_DIR / "models"
    scalers_dir = RESULT_DIR / "scalers"
    if not models_dir.exists() or not scalers_dir.exists():
        print("[오류] 저장된 모델/스케일러가 없습니다. 먼저 train.py 를 실행하세요.")
        return

    (RESULT_DIR / "PR").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "SD").mkdir(parents=True, exist_ok=True)

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

    # ── 2. 공정별 SHAP 분석 ────────────────────────────────────────────
    summary_all = {}

    for process, defect_list in PROCESS_DEFECTS.items():
        recs = records[process]
        if not recs:
            print(f"[경고] {process} 공정 데이터 없음 — 스킵")
            continue

        scaler_path = scalers_dir / f"{process}_scaler.pkl"
        if not scaler_path.exists():
            print(f"[경고] {process} 스케일러 없음 ({scaler_path}) — train.py 를 먼저 실행하세요")
            continue

        # train.py 가 저장한 스케일러 로드 — 학습 때와 동일한 정규화 적용
        scaler    = joblib.load(scaler_path)
        X_all     = np.array([r["features"] for r in recs], dtype=np.float32)
        X_scaled  = scaler.transform(X_all)   # fit 이 아닌 transform 만 수행
        filenames = [r["filename"] for r in recs]

        summary_all[process] = {}
        print(f"{'='*55}")
        print(f"  [{process}] 공정 SHAP 분석  (샘플 {len(recs)}개)")
        print(f"{'='*55}")

        for defect_name, cid in defect_list:
            safe_name  = defect_name.replace("/", "_")
            model_path = models_dir / f"{process}_{safe_name}.json"

            if not model_path.exists():
                print(f"  {defect_name:12s}: 모델 없음 ({model_path.name}) — 스킵")
                continue

            # 저장된 XGBoost 모델 로드
            clf = xgb.XGBClassifier()
            clf.load_model(str(model_path))

            y = np.array(
                [1 if cid in r["defect_ids"] else 0 for r in recs], dtype=np.int32
            )
            pos_count = int(y.sum())

            shap_result = run_shap(clf, X_scaled, y, filenames)
            if shap_result is None:
                print(f"  {defect_name:12s}: 양성 샘플 없음 — 스킵")
                continue

            summary_imp = shap_result["summary_importance"]
            critical    = next(iter(summary_imp))

            # 불량별 개별 JSON 저장 (요약 + 샘플별 상세)
            defect_path = RESULT_DIR / process / f"{safe_name}.json"
            defect_data = {
                "process":           process,
                "defect":            defect_name,
                "sample_count":      pos_count,
                "critical_sensor":   critical,
                "sensor_importance": summary_imp,
                "samples":           shap_result["samples"],
            }
            with open(defect_path, "w", encoding="utf-8") as f:
                json.dump(defect_data, f, ensure_ascii=False, indent=2)

            summary_all[process][defect_name] = {
                "sample_count":      pos_count,
                "critical_sensor":   critical,
                "sensor_importance": summary_imp,
            }

            print(
                f"  {defect_name:12s}: {pos_count:4d}건 | "
                f"치명 센서 = {critical} ({summary_imp[critical]:.1%})"
            )

        print()

    # ── 3. 전체 요약 저장 ─────────────────────────────────────────────
    summary_path = RESULT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_all, f, ensure_ascii=False, indent=2)

    print(f"전체 요약 저장 완료: {summary_path}")
    print(f"불량별 상세 저장 위치: {RESULT_DIR}/PR|SD/<불량명>.json")


if __name__ == "__main__":
    main()
