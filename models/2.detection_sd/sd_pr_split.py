import os
import shutil
from pathlib import Path
from tqdm import tqdm

def reorganize_dataset(project_root):
    root = Path(project_root)
    
    # 원본 데이터 폴더와 타겟 분할 명칭 매핑
    # (만약 'Validation' 폴더가 있다면 'val'로 자동 맵핑되도록 구성)
    data_splits = {
        'Training': 'train',
        'Validation': 'val'
    }
    
    for raw_split_name, target_split in data_splits.items():
        src_base = root / 'dataset' / '1.base_data' / raw_split_name
        
        if not src_base.exists():
            print(f"건너뜀: [{raw_split_name}] 폴더를 찾을 수 없습니다.")
            continue
            
        print(f"\n=== 🚀 [{raw_split_name} -> {target_split}] 데이터 분류 이동 시작 ===")
        
        # 1. 이동할 목적지(Target) 폴더 경로 세팅
        dest_img_pr = root / 'dataset' / '1.base_data' / 'images' / target_split / 'pr'
        dest_img_sd = root / 'dataset' / '1.base_data' / 'images' / target_split / 'sd'
        dest_lbl_pr = root / 'dataset' / '1.base_data' / 'labels_json' / target_split / 'pr'
        dest_lbl_sd = root / 'dataset' / '1.base_data' / 'labels_json' / target_split / 'sd'
        
        # 목적지 폴더들이 없다면 일괄 생성
        for d in [dest_img_pr, dest_img_sd, dest_lbl_pr, dest_lbl_sd]:
            d.mkdir(parents=True, exist_ok=True)
            
        # ---------------------------------------------------------
        # 2. 이미지 데이터 처리 (01.원천데이터)
        # ---------------------------------------------------------
        raw_img_dir = src_base / '01.원천데이터'
        if raw_img_dir.exists():
            print("\n[1/2] 이미지 데이터 필터링 및 복사 중...")
            for folder in raw_img_dir.iterdir():
                if not folder.is_dir(): continue
                
                # YOLO에 쓰이지 않는 '센서데이터' 폴더는 가볍게 무시
                if "이미지데이터" not in folder.name:
                    continue
                    
                # 유저 정의 룰: 사전공정 = pr, 납땜공정 = sd
                if "사전공정" in folder.name:
                    target_dir = dest_img_pr
                elif "납땜공정" in folder.name:
                    target_dir = dest_img_sd
                else:
                    continue
                    
                # 지원하는 이미지 확장자만 추출
                files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                
                # tqdm 프로그레스바로 진행률 표시
                for f in tqdm(files, desc=folder.name, leave=False):
                    shutil.copy2(f, target_dir / f.name)

        # ---------------------------------------------------------
        # 3. 라벨 데이터 처리 (02.라벨링데이터)
        # ---------------------------------------------------------
        raw_lbl_dir = src_base / '02.라벨링데이터'
        if raw_lbl_dir.exists():
            print("\n[2/2] 라벨 JSON 데이터 필터링 및 복사 중...")
            for folder in raw_lbl_dir.iterdir():
                if not folder.is_dir(): continue
                
                # 유저 정의 룰: 사전공정 = pr, 납땜공정 = sd
                if "사전공정" in folder.name:
                    target_dir = dest_lbl_pr
                elif "납땜공정" in folder.name:
                    target_dir = dest_lbl_sd
                else:
                    continue
                    
                # 엑셀(xlsx) 등은 무시하고 오직 JSON 파일만 추출
                files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == '.json']
                
                for f in tqdm(files, desc=folder.name, leave=False):
                    shutil.copy2(f, target_dir / f.name)

    print("\n=== 🎉 데이터 재배치가 모두 완료되었습니다! ===")
    print("이제 5.models 의 preprocess.py를 실행할 준비가 완벽히 끝났습니다.")

if __name__ == "__main__":
    # 로컬 아키텍처 환경 변수 정의
    WINDOWS_PROJECT_ROOT = r"C:\SMT_multi_modal"
    reorganize_dataset(WINDOWS_PROJECT_ROOT)