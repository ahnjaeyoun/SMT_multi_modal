import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import shutil

# 데이터가 이미 분리되어 있는 Base Data 경로
BASE_DATA_DIR = Path("../../dataset/1.basedata")
OUT_DIR = Path("processed_data")

def process_single_file(args):
    """(병렬 처리용) 단일 파일을 처리하는 코어 워커 함수"""
    img_path, img_root, lbl_root, npy_dir, seq_length = args
    
    # 결과 반환용 딕셔너리
    result = {
        'status': 'invalid',
        'reason': None,
        'process_type': None,
        'label_type': None,
        'category_id': None, # 세부 클래스 ID 추적용 추가
        'record': None
    }
    
    stem_name = img_path.stem
    parts = img_path.name.split('_')
    if len(parts) < 3: 
        result['reason'] = 'invalid_filename'
        return result
    
    process_type = parts[0].lower() # pr 또는 sd
    
    rel_path = img_path.parent.relative_to(img_root)
    json_path = lbl_root / rel_path / f"{stem_name}.json"
    
    if not json_path.exists(): 
        result['reason'] = 'missing_json'
        return result
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            j_data = json.load(f)
    except Exception:
        result['reason'] = 'corrupted_json'
        return result
        
    # ========================================================
    # 🌟 핵심 수정 1: JSON에서 32개 세부 클래스(category_id) 추출
    # ========================================================
    if not j_data.get('annotations') or len(j_data['annotations']) == 0:
        result['reason'] = 'missing_annotations'
        return result
        
    # 첫 번째 어노테이션의 category_id를 가져옴 (1~32)
    category_id = j_data['annotations'][0]['category_id']
    
    # PyTorch 모델 학습을 위해 0부터 시작하도록 1을 빼줌 (0~31)
    label = category_id - 1
    
    result['process_type'] = process_type
    result['category_id'] = category_id
    # 1~16은 정상, 17~32는 불량 (JSON 명세 기준)
    result['label_type'] = 'normal' if category_id <= 16 else 'defect'

    # ========================================================
    # 🌟 핵심 수정 2: 센서 데이터 시계열 추출 (기존 구조 유지)
    # ========================================================
    if not j_data.get('sensor_data') or len(j_data['sensor_data']) == 0 or not j_data['sensor_data'][0].get('sensor_sequence'):
        result['reason'] = 'empty_sensor_field'
        return result
        
    sensor_seq = j_data['sensor_data'][0]['sensor_sequence']
    
    if len(sensor_seq) < seq_length:
        result['reason'] = 'short_sequence'
        return result
        
    seq_array = []
    for step in sensor_seq[:seq_length]: 
        seq_array.append([
            step.get('temperature', 0.0),
            step.get('humidity', 0.0),
            step.get('vibration', 0.0),
            step.get('acceleration', 0.0),
            step.get('noise', 0.0)
        ])
        
    seq_array = np.array(seq_array, dtype=np.float32)
    
    # 병렬 처리 시 충돌 방지를 위해 exist_ok=True 처리
    process_npy_dir = npy_dir / rel_path
    process_npy_dir.mkdir(parents=True, exist_ok=True)
    npy_path = process_npy_dir / f"{stem_name}.npy"
    np.save(npy_path, seq_array)
    
    result['status'] = 'valid'
    result['record'] = {
        'image_path': str(img_path.resolve()),
        'sensor_path': str(npy_path.resolve()),
        'label': label, # (0~31)의 세부 클래스 번호가 들어감!
        'process_type': '사전공정' if process_type == 'pr' else '납땜공정'
    }
    
    return result

def process_split(split_name, seq_length=5):
    """미리 분리된 train / val 폴더를 스캔하여 센서 추출 및 메타데이터 생성"""
    if not BASE_DATA_DIR.exists():
        print(f"🚨 치명적 에러: 최상위 데이터 폴더를 찾을 수 없습니다.\n   -> 경로를 확인하세요: {BASE_DATA_DIR.resolve()}")
        sys.exit(1)

    img_root = BASE_DATA_DIR / "images" / split_name
    lbl_root = BASE_DATA_DIR / "labels_json" / split_name
    
    if not img_root.exists():
        print(f"🚨 치명적 에러: '{split_name}' 이미지 폴더를 찾을 수 없습니다.\n   -> 경로: {img_root.resolve()}")
        sys.exit(1)
        
    if not lbl_root.exists():
        print(f"🚨 치명적 에러: '{split_name}' 라벨(JSON) 폴더를 찾을 수 없습니다.\n   -> 경로: {lbl_root.resolve()}")
        sys.exit(1)
    
    split_out_dir = OUT_DIR / split_name
    npy_dir = split_out_dir / "sensors"
    npy_dir.mkdir(parents=True, exist_ok=True)
        
    records = []
    split_stats = {
        'total': 0, 'valid': 0, 'invalid': 0,
        'reasons': {'missing_json': 0, 'corrupted_json': 0, 'missing_annotations': 0, 'empty_sensor_field': 0, 'short_sequence': 0, 'invalid_filename': 0},
        'classes': {'normal': 0, 'defect': 0},
        'processes': {'pr': 0, 'sd': 0}
    }
    
    print(f"\n🔍 [{split_name.upper()}] 병렬 처리 엔진 가동 중...")
    
    img_paths = list(img_root.glob("**/*.jpg")) + list(img_root.glob("**/*.JPG")) + list(img_root.glob("**/*.jpeg"))
    if len(img_paths) == 0:
        print(f"🚨 치명적 에러: '{img_root.resolve()}' 폴더 내에 이미지 파일이 단 하나도 없습니다!")
        sys.exit(1)
    
    # 인자 패키징
    args_list = [(img_path, img_root, lbl_root, npy_dir, seq_length) for img_path in img_paths]
    
    # CPU 풀파워 멀티프로세싱 가동
    max_workers = os.cpu_count() or 4
    print(f"⚡ CPU 코어 {max_workers}개를 풀가동하여 초고속 변환을 시작합니다.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_file, args_list), total=len(args_list), desc=f"{split_name.upper()} 병렬 전처리", unit="파일"))
        
    # 결과 집계
    for res in results:
        split_stats['total'] += 1
        if res['status'] == 'valid':
            split_stats['valid'] += 1
            split_stats['classes'][res['label_type']] += 1
            split_stats['processes'][res['process_type']] += 1
            records.append(res['record'])
        else:
            split_stats['invalid'] += 1
            if res['reason'] in split_stats['reasons']:
                split_stats['reasons'][res['reason']] += 1
            else:
                split_stats['reasons'][res['reason']] = 1

    if records:
        pd.DataFrame(records).to_csv(split_out_dir / "metadata.csv", index=False, encoding='utf-8-sig')
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
                    
            f.write("  [클래스 분포 (1~16:정상 / 17~32:불량)]\n")
            f.write(f"    * 정상(Normal): {stats['classes']['normal']}개\n")
            f.write(f"    * 불량(Defect): {stats['classes']['defect']}개\n")
            
            f.write("  [공정 분포]\n")
            f.write(f"    * 사전공정(PR): {stats['processes']['pr']}개\n")
            f.write(f"    * 납땜공정(SD): {stats['processes']['sd']}개\n")
            f.write("-" * 45 + "\n\n")
            
        f.write(f"==> 전체 사용 가능한 유효 데이터 총합: {total_valid}개\n")

def main():
    import multiprocessing
    multiprocessing.freeze_support()

    if OUT_DIR.exists():
        print(f"🗑️ 기존 {OUT_DIR.resolve()} 폴더를 삭제하고 초기화합니다...")
        shutil.rmtree(OUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    
    all_stats['train'] = process_split("train")
    all_stats['val'] = process_split("val")
    
    write_statistics_txt(all_stats, OUT_DIR)
    print(f"\n📊 전처리 통계 보고서가 {OUT_DIR.resolve()}\\dataset_statistics.txt 에 저장되었습니다.")

if __name__ == "__main__":
    main()