"""
SMT 공정 이상 탐지 (Anomaly Detection)

정상(NOR) 데이터의 센서 평균/표준편차를 공정(PR/SD)별로 산출하고,
불량(DEF) 데이터의 센서값이 정상 기준에서 얼마나 벗어났는지 Z-score와
Mahalanobis 거리로 측정한다.

출력: dataset/1.base_data/anomaly_detection_result.json
"""

import json
import os
import math
from collections import defaultdict  # 키가 없어도 자동으로 기본값을 만들어주는 딕셔너리 (NOR 통계 집계용)
from glob import glob                # 특정 패턴에 맞는 파일 경로 목록을 반환하는 함수


# 경로 설정
# os.path.dirname(__file__) → 현재 스크립트가 있는 폴더 (models/4.cause/)
# "..", ".." 로 두 단계 올라가면 프로젝트 루트 → dataset/1.base_data/ 로 이동
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "1.base_data")
# 원본 JSON 라벨 파일들이 모여 있는 폴더
JSON_DIR    = os.path.join(BASE_DIR, "json")
# 이상 탐지 결과를 저장할 파일 경로 (현재 스크립트와 같은 폴더 = models/4.cause/)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "anomaly_detection_result.json")

# 분석할 센서 종류 5가지 — JSON 파일 안의 키 이름과 정확히 일치해야 함
SENSORS = ["temperature", "humidity", "vibration", "acceleration", "noise"]

# Z-score 이상 판별 임계값
# Z-score 란: (측정값 - 정상 평균) / 정상 표준편차
# 절댓값이 2.0 초과 = 정상 범위에서 표준편차의 2배 이상 벗어남 → 이상으로 판정
Z_THRESHOLD = 2.0

def load_json(path: str) -> dict | None:
    """JSON 파일을 읽어 딕셔너리로 반환한다. 인코딩을 자동으로 탐색한다.

    한국어가 포함된 파일은 인코딩 방식이 utf-8, cp949 등 다를 수 있으므로
    여러 인코딩을 순서대로 시도해 성공한 방식으로 읽는다.
    """
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # 해당 인코딩으로 읽기 실패하면 다음 인코딩으로 시도
            continue
    # 모든 인코딩 시도 실패 시 None 반환
    return None


def extract_sensor_sequences(data: dict) -> list[dict]:
    """JSON 데이터에서 모든 센서 측정값 스텝을 꺼내 평탄화(flatten)해서 반환한다.

    JSON 구조:
      sensor_data → 여러 센서 블록 → 각 블록의 sensor_sequence → 타임스텝(step) 목록
    이 함수는 중첩된 구조를 풀어 [{sensor: value, ...}, ...] 형태의 평탄한 리스트로 만든다.
    5개 센서 값 중 하나라도 None 인 스텝은 제외한다.
    """
    steps = []
    for sd in data.get("sensor_data", []):
        for step in sd.get("sensor_sequence", []):
            # 5개 센서 키의 값을 딕셔너리로 수집
            row = {s: step.get(s) for s in SENSORS}
            # 값이 하나라도 None 이면 불완전한 스텝이므로 제외
            if all(v is not None for v in row.values()):
                steps.append(row)
    return steps


def compute_normal_stats() -> dict:
    """정상(NOR) 파일들의 센서 평균과 표준편차를 공정(PR/SD)별로 계산한다.

    반환 형태:
    {
        "PR": {"temperature": {"mean": x, "std": y}, ..., "sample_count": n},
        "SD": {...}
    }

    표준편차를 직접 구하는 방법:
      분산(variance) = E[X²] - E[X]²
      표준편차(std)  = sqrt(분산)
    데이터를 두 번 순회하지 않아도 되므로 대용량에 유리하다.
    """
    # 공정별, 센서별 누적 합산값 저장소 (defaultdict → 없는 키도 자동 0 초기화)
    sums    = defaultdict(lambda: defaultdict(float))   # 값의 합
    sq_sums = defaultdict(lambda: defaultdict(float))   # 값의 제곱합 (분산 계산용)
    counts  = defaultdict(int)                          # 타임스텝 총 개수

    # 파일명에 "_NOR_" 이 포함된 JSON 파일만 선택 (정상 데이터)
    nor_files = glob(os.path.join(JSON_DIR, "*_NOR_*.json"))
    print(f"[NOR] 파일 수 (파일명 기준): {len(nor_files)}")

    skipped = 0  # 유효하지 않아 제외된 파일 수
    for fp in nor_files:
        # 파일명 첫 두 글자로 공정 구분 (예: "PR_NOR_..." → "PR")
        process = os.path.basename(fp)[:2]
        data = load_json(fp)
        if data is None:
            skipped += 1
            continue

        # 파일명이 NOR 이어도 내부 annotation 에 불량 category_id(17~32)가 있으면 제외
        # 라벨 오류나 혼합 이미지를 걸러내는 검증 단계
        ann_ids = {ann.get("category_id", 0) for ann in data.get("annotations", [])}
        if any(cid >= 17 for cid in ann_ids):
            skipped += 1
            continue

        # 유효한 파일의 모든 타임스텝을 순회하며 합산
        for step in extract_sensor_sequences(data):
            for s in SENSORS:
                v = step[s]
                sums[process][s]    += v        # 합산 (평균 계산용)
                sq_sums[process][s] += v * v    # 제곱 합산 (분산 계산용)
            counts[process] += 1  # 타임스텝 1개 처리 완료

    print(f"[NOR] 라벨 검증 후 제외된 파일: {skipped}개")

    # 수집한 합산값으로 평균과 표준편차 계산
    stats = {}
    for process in ("PR", "SD"):
        n = counts[process]
        if n == 0:
            continue  # 해당 공정 데이터가 아예 없으면 스킵
        stats[process] = {}
        for s in SENSORS:
            mean     = sums[process][s] / n                     # 평균 = 합 / 개수
            variance = sq_sums[process][s] / n - mean ** 2      # 분산 = E[X²] - E[X]²
            # max(..., 1e-12): 분산이 0 이 되면 나눗셈 불가 → 아주 작은 값으로 대체
            std = math.sqrt(max(variance, 1e-12))
            stats[process][s] = {"mean": round(mean, 6), "std": round(std, 6)}
        stats[process]["sample_count"] = n
        print(f"  [{process}] 정상 샘플 수: {n:,}")
        for s in SENSORS:
            m  = stats[process][s]["mean"]
            sd = stats[process][s]["std"]
            print(f"    {s:15s}: mean={m:.4f}, std={sd:.4f}")

    return stats


def build_thresholds(normal_stats: dict) -> dict:
    """정상 기준 통계에서 센서별 임계값을 실제 센서 단위로 변환해 반환한다.

    Z-score 임계값(2.0)을 실제 측정값으로 환산하는 공식:
      상한 임계값 = 정상 평균 + Z_THRESHOLD × 정상 표준편차
      하한 임계값 = 정상 평균 - Z_THRESHOLD × 정상 표준편차

    이렇게 하면 화면에 "습도 상한: 61.64%" 처럼 실제 단위로 표시할 수 있다.
    """
    thresholds = {}
    for process, stats in normal_stats.items():
        if process == "sample_count":
            continue
        thresholds[process] = {}
        for s in SENSORS:
            mean = stats[s]["mean"]
            std  = stats[s]["std"]
            thresholds[process][s] = {
                "lower": round(mean - Z_THRESHOLD * std, 6),  # 이상 판정 하한값
                "mean":  round(mean, 6),
                "upper": round(mean + Z_THRESHOLD * std, 6),  # 이상 판정 상한값
                "std":   round(std, 6),
            }
    return thresholds


def main():
    print("=" * 60)
    print("SMT 센서 이상 탐지 시작")
    print("=" * 60)

    # 1. 정상 기준 통계 산출
    # 정상(NOR) 파일들의 센서 평균·표준편차를 계산해 이후 이상 판별의 기준으로 사용
    print("\n[1단계] 정상 데이터 통계 산출 중...")
    normal_stats = compute_normal_stats()

    # 2. Z=2.0 임계값을 실제 센서 단위로 변환
    # "Z=2.0 초과" 라는 추상적 기준을 "온도 27.4°C 초과" 처럼 실제 측정 단위로 환산
    print("\n[2단계] 실제 단위 임계값 계산 중...")
    thresholds = build_thresholds(normal_stats)
    for process, sensors in thresholds.items():
        print(f"\n  [{process}] 이상 판정 구간 (Z={Z_THRESHOLD:.1f} 기준)")
        for s, v in sensors.items():
            print(f"    {s:15s}: {v['lower']:.4f} ~ {v['upper']:.4f}  (평균 {v['mean']:.4f})")

    # 3. 결과 저장
    # 파일별 상세 결과(results)는 제외하고 정상 기준과 임계값만 저장
    output = {
        "z_threshold": Z_THRESHOLD,   # 이상 판정에 사용된 Z-score 기준값
        "thresholds":  thresholds,    # 실제 센서 단위로 환산된 이상 판정 임계값
        "normal_baseline": normal_stats,  # 공정별 정상 센서 평균·표준편차
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # ensure_ascii=False: 한글이 \uXXXX 로 깨지지 않고 그대로 저장됨
        # indent=2: 들여쓰기 2칸으로 사람이 읽기 좋게 저장
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[결과 저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    # 이 파일을 직접 실행할 때만 main() 호출
    # 다른 파일에서 import 했을 때는 자동으로 실행되지 않음
    main()
