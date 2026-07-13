import os
import json
import glob
import pandas as pd
import numpy as np

# 경로 설정 (스크립트 위치 기반 상대 경로)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

BASE_DIR = os.path.join(PROJECT_ROOT, "dataset", "1.basedata", "labels_json")
OUTPUT_DIR = SCRIPT_DIR

# 결과 저장용 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 1. 라벨 추출 (Category ID: 1~16 정상, 17~32 불량)
    label = 0
    if 'annotations' in data and len(data['annotations']) > 0:
        for ann in data['annotations']:
            if 17 <= ann.get('category_id', 0) <= 32:
                label = 1
                break
                
    # 2. 센서 시퀀스 데이터 파싱
    features = {'filename': os.path.basename(json_path), 'label': label}
    
    if 'sensor_data' in data and len(data['sensor_data']) > 0:
        sequence = data['sensor_data'][0].get('sensor_sequence', [])
        
        if not sequence:
            return None
            
        temps = [item.get('temperature', 0) for item in sequence]
        hums = [item.get('humidity', 0) for item in sequence]
        vibs = [item.get('vibration', 0) for item in sequence]
        accs = [item.get('acceleration', 0) for item in sequence]
        noises = [item.get('noise', 0) for item in sequence]
        
        # 3. Feature Engineering (기초 통계량 추출)
        features.update({
            'temp_mean': np.mean(temps),
            'temp_std': np.std(temps),
            'temp_max': np.max(temps),
            'temp_min': np.min(temps),
            
            'hum_mean': np.mean(hums),
            'hum_std': np.std(hums),
            
            'vib_mean': np.mean(vibs),
            'vib_std': np.std(vibs),
            'vib_max': np.max(vibs),
            
            'acc_mean': np.mean(accs),
            'acc_std': np.std(accs),
            'acc_max': np.max(accs),
            
            'noise_mean': np.mean(noises),
            'noise_std': np.std(noises),
            'noise_max': np.max(noises),
        })
        return features
    return None

def main():
    print("센서 데이터 추출 시작...")
    
    all_json_files = []
    # train, val 디렉토리 내의 모든 json (pr, sd 모두 포함)
    for split in ['train', 'val']:
        pattern = os.path.join(BASE_DIR, split, '**', '*.json')
        all_json_files.extend(glob.glob(pattern, recursive=True))
        
    print(f"발견된 JSON 파일 개수: {len(all_json_files)}개")
    
    records = []
    for path in all_json_files:
        feats = extract_features_from_json(path)
        if feats is not None:
            records.append(feats)
            
    df = pd.DataFrame(records)
    
    output_path = os.path.join(OUTPUT_DIR, 'sensor_features.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"추출 완료! 데이터 저장: {output_path}")
    print(df.head())
    print(f"\n라벨 분포:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    main()
