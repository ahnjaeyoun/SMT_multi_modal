import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_val_to_test(project_root, target_process='sd', ratio=0.5, random_seed=42):
    """
    val 폴더의 이미지와 JSON 라벨을 읽어, 
    원인(Cause)별로 균등하게 섞은 뒤 지정된 비율(50%)만큼 test 폴더로 이동시키는 원샷 함수
    """
    # 난수 고정 (매번 똑같이 섞이도록 설정)
    random.seed(random_seed)
    root = Path(project_root)
    
    # 원본(val) 경로
    val_img_dir = root / 'dataset' / '1.base_data' / 'images' / 'val' / target_process
    val_lbl_dir = root / 'dataset' / '1.base_data' / 'labels_json' / 'val' / target_process
    
    # 이동할 목적지(test) 경로
    test_img_dir = root / 'dataset' / '1.base_data' / 'images' / 'test' / target_process
    test_lbl_dir = root / 'dataset' / '1.base_data' / 'labels_json' / 'test' / target_process
    
    print(f"=== SMT {target_process.upper()} 공정 [Val -> Test] 데이터 분할(비율: {ratio}) 시작 ===")
    
    # 유효성 검사
    if not val_lbl_dir.exists() or not val_img_dir.exists():
        print(f"에러: 원본 val 폴더를 찾을 수 없습니다. 경로를 확인하세요.\n{val_lbl_dir}")
        return

    # test 폴더 자동 생성
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. 파일 자동 그룹화 (수동 cause 리스트 필요 없음!)
    # 파일명 예시: SD_DEF_NM_A_... -> 분리해서 'DEF_NM', 'NOR_IC' 등으로 자동 그룹핑
    category_files = defaultdict(list)
    
    for json_path in val_lbl_dir.glob('*.json'):
        name_parts = json_path.stem.split('_')
        
        # 파일명 규칙에 따라 (예: SD / DEF / NM / ...)
        if len(name_parts) >= 3:
            category = f"{name_parts[1]}_{name_parts[2]}"  # 예: 'DEF_NM' (불량_미납)
        else:
            category = "UNKNOWN"
            
        category_files[category].append(json_path)

    print(f"총 {len(category_files)}개의 세부 카테고리를 자동으로 식별했습니다.")
    for cat, files in category_files.items():
        print(f" - [{cat}]: {len(files)}개")

    # 2. 그룹별로 섞고 이동(Move) 수행
    move_count = 0
    print("\n데이터 이동을 시작합니다...")
    
    for category, files in category_files.items():
        # 편향되지 않게 리스트를 무작위로 섞음
        random.shuffle(files)
        
        # 50% 분할 지점 정수형(int)으로 정확히 계산
        target_num = int(len(files) * ratio)
        
        # 처음부터 target_num까지 슬라이싱(자르기)
        test_files = files[:target_num]
        
        for json_path in test_files:
            # 매칭되는 이미지 찾기 (대소문자 지원)
            img_name = json_path.with_suffix('.JPG').name
            img_path = val_img_dir / img_name
            if not img_path.exists():
                img_path = val_img_dir / json_path.with_suffix('.jpg').name
                
            if img_path.exists():
                # JSON 라벨과 이미지를 둘 다 test 폴더로 이동 (Move)
                shutil.move(str(json_path), str(test_lbl_dir / json_path.name))
                shutil.move(str(img_path), str(test_img_dir / img_path.name))
                move_count += 1
            else:
                print(f"경고: 라벨과 쌍을 이룰 원본 이미지가 없습니다. (이동 스킵) -> {json_path.name}")

    print(f"\n=== 스플릿 완료! 총 {move_count}세트(이미지+라벨)가 Test 폴더로 이동되었습니다. ===")


if __name__ == "__main__":
    # 로컬 경로 변수 설정
    LOCAL_PROJECT_ROOT = r"C:\SMT_multi_modal"
    
    # val 데이터를 0.5(50%) 비율로 잘라서 test에 넣기
    split_val_to_test(project_root=LOCAL_PROJECT_ROOT, target_process='sd', ratio=0.5)