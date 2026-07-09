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

import joblib       # 파이썬 객체(모델, 스케일러 등)를 파일로 저장하고 불러오는 라이브러리
import numpy as np
import shap         # 머신러닝 모델의 예측 결과를 각 피처별로 얼마나 기여했는지 설명하는 라이브러리
import xgboost as xgb  # 빠르고 정확한 트리 기반 분류기 (SHAP 공식 지원)

# SHAP, XGBoost 라이브러리 등에서 출력되는 불필요한 경고 메시지를 숨김
warnings.filterwarnings("ignore")


# 경로 설정
# Path(__file__).resolve() → 이 스크립트 파일의 절대 경로
# .parents[2] → 두 단계 위 폴더 = 프로젝트 루트(SMT_multimodal/)
BASE_DIR   = Path(__file__).resolve().parents[2] / "dataset" / "1.base_data"
# 원본 JSON 라벨 파일들이 모여 있는 폴더
JSON_DIR   = BASE_DIR / "json"
# train.py 가 모델/스케일러를 저장한 폴더이자, 이 파일의 분석 결과도 저장할 폴더
# Path(__file__).resolve().parent → 현재 스크립트가 있는 폴더 (models/4.cause/)
RESULT_DIR = Path(__file__).resolve().parent / "result_cause"


# 상수 정의

# 센서 종류 5가지 — JSON 파일 안의 키 이름과 정확히 일치해야 함
SENSOR_KEYS = ["temperature", "humidity", "vibration", "acceleration", "noise"]

# 공정별로 발생할 수 있는 불량 유형 정의 — (불량명, category_id)
# PR 공정: 리플로 전 납 도포 단계 → 납 모양/위치 불량 6종
# SD 공정: 납땜 단계 → 부품 실장 관련 불량 포함 12종
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

# PROCESS_DEFECTS 에 있는 모든 category_id 를 집합(set)으로 한꺼번에 모아둠
# 집합은 "포함 여부 확인"이 O(1) 로 매우 빠름 → parse_json 에서 반복 조회 시 유리
ALL_DEFECT_IDS = {cid for defs in PROCESS_DEFECTS.values() for _, cid in defs}


def parse_json(path):
    """JSON 파일 하나를 읽어 공정·파일명·피처·불량 ID 집합을 반환한다.

    Returns:
        성공: (record_dict, None)
        실패: (None, 제외_이유_문자열)
    """
    # JSON 파일 열기 — 파일이 깨졌거나 형식이 잘못된 경우 예외 발생
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # 제외 이유: 파일을 읽는 도중 오류 발생 (파일 손상, 잘못된 JSON 형식 등)
        return None, "JSON 파싱 실패 (파일 손상 또는 형식 오류)"

    # 파일명 첫 번째 토큰으로 공정 구분 (예: "PR_DEF_NM_A_..." → "PR")
    process = path.stem.split("_")[0]
    if process not in PROCESS_DEFECTS:
        # 제외 이유: 파일명이 PR_ 또는 SD_ 로 시작하지 않아 어느 공정인지 알 수 없음
        return None, f"공정 식별 불가 (파일명 접두사: '{process}', PR/SD 아님)"

    # annotations 리스트에서 불량 category_id 만 골라 집합으로 수집
    defect_ids = set()
    for ann in data.get("annotations", []):
        cid = ann.get("category_id", 0)
        if cid in ALL_DEFECT_IDS:
            defect_ids.add(cid)

    if not defect_ids:
        # 제외 이유: 불량 어노테이션(category_id 17~32)이 하나도 없는 정상 이미지
        # 원인 분석은 불량 샘플만 필요하므로 정상 이미지는 포함하지 않음
        return None, "불량 어노테이션 없음 (정상 이미지 — category_id 17~32 미존재)"

    # sensor_data 키에서 센서 블록 가져오기
    sensor_data = data.get("sensor_data", [])
    if not sensor_data:
        # 제외 이유: JSON 에 sensor_data 키 자체가 없거나 빈 리스트
        # 센서 값이 없으면 피처를 만들 수 없어 SHAP 분석 불가
        return None, "센서 데이터 없음 (sensor_data 키 누락 또는 빈 리스트)"

    # 첫 번째 센서 블록의 시계열(시간 순서대로 측정값이 나열된 리스트) 가져오기
    seq = sensor_data[0].get("sensor_sequence", [])
    if not seq:
        # 제외 이유: sensor_data 는 있지만 그 안의 sensor_sequence 가 비어 있음
        # 실제 측정값이 한 개도 없으면 평균·최대값 계산 자체가 불가능
        return None, "센서 시퀀스 없음 (sensor_sequence 가 비어 있음)"

    # 센서별 평균(mean)과 최대(max) 계산 → 피처 벡터 10개 구성
    # 예: [온도평균, 온도최대, 습도평균, 습도최대, ...]
    features = []
    for key in SENSOR_KEYS:
        # 시퀀스의 각 타임스텝에서 해당 센서 값만 수집 (키가 없는 타임스텝은 건너뜀)
        vals = [step[key] for step in seq if key in step]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            features.append(float(arr.mean()))  # 시퀀스 전체 평균
            features.append(float(arr.max()))   # 시퀀스 전체 최대값
        else:
            # 해당 센서 데이터가 아예 없으면 0으로 채움
            features.extend([0.0, 0.0])

    return {
        "filename":   path.stem,    # 확장자 없는 파일명 (샘플 식별자로 사용)
        "process":    process,
        "features":   features,
        "defect_ids": defect_ids,
    }, None


def normalize_sensor(raw):
    """피처 10개의 |SHAP| 배열을 센서 5개의 정규화된 중요도 딕셔너리로 변환한다.

    피처 배열 구조: [temp_mean, temp_max, hum_mean, hum_max, vib_mean, vib_max, ...]
    센서 i 의 기여도 = (raw[i*2] + raw[i*2+1]) / 전체합
    즉, 같은 센서의 mean 기여도와 max 기여도를 합산해 센서 하나의 최종 중요도로 만듦

    반환값은 중요도가 높은 순서(내림차순)로 정렬된 딕셔너리
    """
    total = raw.sum()
    if total == 0:
        # 모든 SHAP 값이 0이면 중요도 계산 불가 → 빈 딕셔너리 반환
        return {}

    # 합이 1이 되도록 정규화 → 각 피처의 상대적 기여 비율
    norm = raw / total

    # 피처 10개를 센서 5개로 합산
    importance = {}
    for i, sensor in enumerate(SENSOR_KEYS):
        base = i * 2  # mean 인덱스 (max 인덱스는 base+1)
        importance[sensor] = round(float(norm[base] + norm[base + 1]), 4)

    # 중요도가 높은 순서대로 정렬 (가장 영향력 큰 센서가 맨 앞)
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def run_shap(clf, X, y, filenames):
    """저장된 XGBoost 모델로 SHAP 분석을 수행하고 요약·샘플별 중요도를 반환한다.

    train.py 에서 이미 학습된 모델을 받아 SHAP 만 수행한다.
    이 함수 안에서는 모델 학습을 절대 하지 않는다.

    Args:
        clf:       train.py 에서 저장한 XGBoost 모델 (이미 학습 완료)
        X:         공정별 StandardScaler 로 정규화된 센서 피처 행렬 (N × 10)
        y:         이진 레이블 (1=해당 불량 있음, 0=없음), 길이 N
        filenames: 각 샘플의 파일명 (길이 N), 샘플별 결과에 붙이는 식별자

    Returns:
        {
          "summary_importance": {sensor: ratio, ...},  # 불량 샘플 전체 평균 SHAP
          "samples": [                                  # 샘플 하나하나의 상세 결과
            {"filename": ..., "critical_sensor": ..., "sensor_importance": {...}},
            ...
          ]
        }
        또는 None (불량 샘플이 하나도 없을 때)
    """
    # 불량이 있는 샘플(y=1)만 True 인 마스크 배열
    pos_mask  = y == 1
    pos_count = int(pos_mask.sum())
    if pos_count == 0:
        # 이 불량에 해당하는 샘플이 하나도 없으면 SHAP 계산 불가
        return None

    # train.py 에서 저장한 모델을 그대로 사용 — 추가 학습 없음
    # TreeExplainer: XGBoost 전용으로 빠르고 정확한 SHAP 계산기
    explainer = shap.TreeExplainer(clf)

    # 불량 샘플(X_pos)에 대해서만 SHAP 값 계산
    # "불량이 실제로 발생했을 때 각 센서가 얼마나 영향을 미쳤는가"를 보기 위함
    # shap_vals shape: (불량샘플 수, 피처 수=10)
    X_pos     = X[pos_mask]
    shap_vals = explainer.shap_values(X_pos)

    # 전체 요약: 불량 샘플 전체의 |SHAP| 평균 → 센서 5개 중요도로 변환
    # 절댓값을 쓰는 이유: 양수(불량 기여)든 음수(불량 억제)든 영향력 크기만 봄
    mean_abs_all       = np.abs(shap_vals).mean(axis=0)  # shape: (10,)
    summary_importance = normalize_sensor(mean_abs_all)
    if not summary_importance:
        return None

    # 샘플별 상세: 불량 샘플 하나하나에 대해 센서 중요도를 따로 계산
    # 어떤 이미지에서 어떤 센서가 문제였는지 개별 추적 가능
    pos_filenames = [fn for fn, flag in zip(filenames, y) if flag == 1]
    samples = []

    for fn, sv in zip(pos_filenames, shap_vals):
        # sv: 이 샘플 하나의 SHAP 값 벡터, shape: (10,)
        sample_importance = normalize_sensor(np.abs(sv))
        if not sample_importance:
            continue
        # 내림차순 정렬이므로 첫 번째 키가 이 샘플에서 가장 영향력 큰 센서
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
    # JSON 폴더가 존재하는지 먼저 확인
    if not JSON_DIR.exists():
        print(f"[오류] JSON 폴더를 찾을 수 없습니다: {JSON_DIR}")
        return

    # train.py 가 저장한 모델/스케일러 폴더가 있는지 확인
    # 없으면 analyze.py 는 실행할 수 없음 — 반드시 train.py 를 먼저 실행해야 함
    models_dir  = RESULT_DIR / "models"
    scalers_dir = RESULT_DIR / "scalers"
    if not models_dir.exists() or not scalers_dir.exists():
        print("[오류] 저장된 모델/스케일러가 없습니다. 먼저 train.py 를 실행하세요.")
        return

    # 결과 저장 폴더 생성 (이미 있으면 그대로 유지)
    (RESULT_DIR / "PR").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "SD").mkdir(parents=True, exist_ok=True)

    # 1. 전체 JSON 파싱
    # json/ 폴더 안의 모든 .json 파일을 이름 순으로 읽음
    json_files = sorted(JSON_DIR.glob("*.json"))
    print(f"JSON 파일 {len(json_files)}개 파싱 중...")

    # PR / SD 공정별로 유효한 레코드를 분리해서 저장
    records = {"PR": [], "SD": []}
    skipped = []  # 제외된 파일과 그 이유를 함께 기록

    for path in json_files:
        record, reason = parse_json(path)
        if record:
            records[record["process"]].append(record)
        else:
            # 제외된 파일은 이름과 이유를 함께 보관
            skipped.append((path.name, reason))

    print(f"  PR: {len(records['PR'])}개 | SD: {len(records['SD'])}개 | 제외: {len(skipped)}개\n")

    # 제외된 파일 목록과 이유를 txt 파일로 저장 (디버깅·검수용)
    skip_path = RESULT_DIR / "skipped_files.txt"
    with open(skip_path, "w", encoding="utf-8") as f:
        f.write(f"제외된 파일 목록 (총 {len(skipped)}개)\n")
        f.write("=" * 70 + "\n")
        for name, reason in skipped:
            f.write(f"{name}  # {reason}\n")
    print(f"제외 파일 목록 저장: {skip_path}\n")

    # 2. 공정별 SHAP 분석
    summary_all = {}  # 나중에 summary.json 으로 저장할 전체 요약

    for process, defect_list in PROCESS_DEFECTS.items():
        recs = records[process]
        if not recs:
            print(f"[경고] {process} 공정 데이터 없음 — 스킵")
            continue

        # train.py 가 저장한 스케일러 파일 로드
        # 학습 때와 동일한 정규화를 분석 때도 적용해야 SHAP 값이 올바르게 나옴
        scaler_path = scalers_dir / f"{process}_scaler.pkl"
        if not scaler_path.exists():
            print(f"[경고] {process} 스케일러 없음 ({scaler_path}) — train.py 를 먼저 실행하세요")
            continue

        # joblib.load: 파이썬 객체를 파일에서 복원 (train.py 에서 joblib.dump 로 저장한 것)
        scaler    = joblib.load(scaler_path)
        X_all     = np.array([r["features"] for r in recs], dtype=np.float32)
        # fit 없이 transform 만 수행 — 학습 때 계산한 평균·표준편차 그대로 적용
        X_scaled  = scaler.transform(X_all)
        filenames = [r["filename"] for r in recs]

        summary_all[process] = {}
        print(f"{'='*55}")
        print(f"  [{process}] 공정 SHAP 분석  (샘플 {len(recs)}개)")
        print(f"{'='*55}")

        # 공정에 속한 불량 유형 각각에 대해 SHAP 분석 수행
        for defect_name, cid in defect_list:
            # 파일명에 쓸 수 없는 문자('/')를 '_'로 치환 (예: "납금감/핀홀" → "납금감_핀홀")
            safe_name  = defect_name.replace("/", "_")
            # train.py 가 저장한 불량별 XGBoost 모델 파일 경로
            model_path = models_dir / f"{process}_{safe_name}.json"

            if not model_path.exists():
                # 해당 불량의 모델 파일이 없으면 스킵 (train.py 에서 샘플 부족으로 학습 안 됐을 수 있음)
                print(f"  {defect_name:12s}: 모델 없음 ({model_path.name}) — 스킵")
                continue

            # XGBoost 모델을 파일에서 복원
            # xgb.XGBClassifier() 빈 객체를 만든 뒤 load_model 로 가중치를 불러옴
            clf = xgb.XGBClassifier()
            clf.load_model(str(model_path))

            # 현재 불량(cid)이 해당 샘플에 존재하면 1, 없으면 0 으로 이진 레이블 생성
            y = np.array(
                [1 if cid in r["defect_ids"] else 0 for r in recs], dtype=np.int32
            )
            pos_count = int(y.sum())  # 이 불량이 실제로 발생한 샘플 수

            # SHAP 분석 실행
            shap_result = run_shap(clf, X_scaled, y, filenames)
            if shap_result is None:
                print(f"  {defect_name:12s}: 양성 샘플 없음 — 스킵")
                continue

            summary_imp = shap_result["summary_importance"]
            # 내림차순 정렬이므로 첫 번째 키가 전체 평균 기준 가장 중요한 센서
            critical = next(iter(summary_imp))

            # 불량별 개별 JSON 저장 (요약 + 샘플별 상세)
            defect_path = RESULT_DIR / process / f"{safe_name}.json"
            defect_data = {
                "process":           process,
                "defect":            defect_name,
                "sample_count":      pos_count,
                "critical_sensor":   critical,
                "sensor_importance": summary_imp,             # 전체 평균 기반 요약
                "samples":           shap_result["samples"],  # 샘플 하나하나의 상세 결과
            }
            with open(defect_path, "w", encoding="utf-8") as f:
                # ensure_ascii=False: 한글이 \uXXXX 로 깨지지 않고 그대로 저장됨
                # indent=2: 들여쓰기 2칸으로 사람이 읽기 좋게 저장
                json.dump(defect_data, f, ensure_ascii=False, indent=2)

            # summary.json 에는 샘플 목록 없이 요약 정보만 기록
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

    # 3. 전체 요약 저장
    # 공정×불량 조합 전체를 한 파일에 담은 요약본
    # 빠른 조회나 대시보드 연동 시 이 파일 하나만 읽으면 됨
    summary_path = RESULT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_all, f, ensure_ascii=False, indent=2)

    print(f"전체 요약 저장 완료: {summary_path}")
    print(f"불량별 상세 저장 위치: {RESULT_DIR}/PR|SD/<불량명>.json")


if __name__ == "__main__":
    # 이 파일을 직접 실행할 때만 main() 호출
    # 다른 파일에서 import 했을 때는 자동으로 실행되지 않음
    main()
