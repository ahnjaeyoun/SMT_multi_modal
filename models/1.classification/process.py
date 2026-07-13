import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# [논리적 수정 1] 절대 경로 하드코딩 제거 및 프로젝트 루트 동적 추적
# 현재 스크립트(models/2.classification/preprocess.py)를 기준으로 2단계 위를 프로젝트 루트로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
BASE_DATA_DIR = PROJECT_ROOT / "dataset" / "1.basedata"
OUT_DIR = Path(__file__).resolve().parent / "processed_data"

def process_split(split_name, seq_length=5):
    """미리 분리된 train / val / test 폴더를 스캔하여 센서 추출 및 메타데이터 생성"""
    img_root = BASE_DATA_DIR / "images" / split_name
    lbl_root = BASE_DATA_DIR / "labels_json" / split_name
    
    split_out_dir = OUT_DIR / split_name
    npy_dir = split_out_dir / "sensors"
    npy_dir.mkdir(parents=True, exist_ok=True)
    
    if not img_root.exists() or not lbl_root.exists():
        print(f"⚠️ {split_name} 폴더를 찾을 수 없습니다. 경로를 확인하세요: {img_root}")
        return None
        
    records = []
    split_stats = {
        'total': 0, 'valid': 0, 'invalid': 0,
        'reasons': {'missing_json': 0, 'corrupted_json': 0, 'empty_sensor_field': 0, 'short_sequence': 0},
        'classes': {'normal': 0, 'defect': 0},
        'processes': {'pr': 0, 'sd': 0}
    }
    
    print(f"🔍 [{split_name.upper()}] 데이터셋 스캔 및 센서 변환 중...")
    
    # 이미지 확장자 통합 검색
    img_paths = list(img_root.glob("**/*.jpg")) + list(img_root.glob("**/*.JPG")) + list(img_root.glob("**/*.jpeg"))
    
    for img_path in img_paths:
        split_stats['total'] += 1
        stem_name = img_path.stem
        
        parts = img_path.name.split('_')
        if len(parts) < 3: 
            split_stats['invalid'] += 1
            continue
            
        process_type = parts[0].lower() # pr 또는 sd
        condition = parts[1].lower()    # nor 또는 def
        label = 1 if 'def' in condition else 0
        
        # 상대 경로 구조 유지하여 JSON 매칭
        rel_path = img_path.parent.relative_to(img_root)
        json_path = lbl_root / rel_path / f"{stem_name}.json"
        
        if not json_path.exists(): 
            split_stats['invalid'] += 1
            split_stats['reasons']['missing_json'] += 1
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                j_data = json.load(f)
            except Exception:
                split_stats['invalid'] += 1
                split_stats['reasons']['corrupted_json'] += 1
                continue
                
        # 센서 데이터 검증
        if not j_data.get('sensor_data') or not j_data['sensor_data'][0].get('sensor_sequence'):
            split_stats['invalid'] += 1
            split_stats['reasons']['empty_sensor_field'] += 1
            continue
            
        sensor_seq = j_data['sensor_data'][0]['sensor_sequence']
        
        # 시퀀스 길이 검증
        if len(sensor_seq) < seq_length:
            split_stats['invalid'] += 1
            split_stats['reasons']['short_sequence'] += 1
            continue 
            
        seq_array = []
        for step in sensor_seq[:seq_length]:
            seq_array.append([
                step.get('temperature', 0),
                step.get('humidity', 0),
                step.get('vibration', 0),
                step.get('acceleration', 0),
                step.get('noise', 0)
            ])
            
        seq_array = np.array(seq_array, dtype=np.float32)
        
        npy_path = npy_dir / f"{stem_name}.npy"
        np.save(npy_path, seq_array)
        
        # 통계 집계 (도메인 지식: PR=사전공정, SD=납땜공정 반영)
        split_stats['valid'] += 1
        split_stats['classes']['defect' if label == 1 else 'normal'] += 1
        if process_type in ['pr', 'sd']:
            split_stats['processes'][process_type] += 1
        
        # [논리적 수정 2] MLOps 환경 이식을 위한 프로젝트 루트 기준 상대 경로 추출
        # 저장되는 경로 예시: dataset/1.base_data/images/train/pr/...
        rel_img_path = os.path.relpath(img_path, PROJECT_ROOT).replace('\\', '/')
        rel_npy_path = os.path.relpath(npy_path, PROJECT_ROOT).replace('\\', '/')
        
        records.append({
            'image_path': rel_img_path,
            'sensor_path': rel_npy_path,
            'label': label,
            'process_type': '사전공정' if process_type == 'pr' else '납땜공정' if process_type == 'sd' else '알수없음'
        })

    if records:
        csv_path = split_out_dir / "metadata.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   -> 📦 {split_name}: {len(records)}건 변환 및 목차 생성 완료.")

    return split_stats

def write_statistics_txt(stats_dict, out_dir):
    """수집된 통계를 바탕으로 dataset_statistics.txt 파일 생성"""
    txt_path = out_dir / "dataset_statistics.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== GEMS 멀티모달 데이터셋 전처리 통계 ===\n\n")
        
        total_valid = 0
        for split_name, stats in stats_dict.items():
            if not stats: continue
            total_valid += stats['valid']
            
            f.write(f"[{split_name.upper()} 데이터셋]\n")
            f.write(f"- 총 스캔된 이미지: {stats['total']}개\n")
            f.write(f"- 유효한 데이터 쌍: {stats['valid']}개\n")
            f.write(f"- 제외된 데이터: {stats['invalid']}개\n")
            
            if stats['invalid'] > 0:
                f.write("  [제외 사유]\n")
                for reason, count in stats['reasons'].items():
                    if count > 0: f.write(f"    * {reason}: {count}개\n")
                    
            f.write("  [클래스 분포]\n")
            f.write(f"    * 정상(Normal): {stats['classes']['normal']}개\n")
            f.write(f"    * 불량(Defect): {stats['classes']['defect']}개\n")
            
            f.write("  [공정 분포]\n")
            f.write(f"    * 사전공정(PR): {stats['processes']['pr']}개\n")
            f.write(f"    * 납땜공정(SD): {stats['processes']['sd']}개\n")
            f.write("-" * 45 + "\n\n")
            
        f.write(f"==> 전체 사용 가능한 유효 데이터 총합: {total_valid}개\n")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    
    # [논리적 수정 3] Test 데이터셋 파이프라인 누락 복구
    all_stats['train'] = process_split("train")
    all_stats['val'] = process_split("val")
    all_stats['test'] = process_split("test")
    
    write_statistics_txt(all_stats, OUT_DIR)
    print(f"\n📊 전처리 통계 보고서가 {OUT_DIR}/dataset_statistics.txt 에 저장되었습니다.")

if __name__ == "__main__":
    main()