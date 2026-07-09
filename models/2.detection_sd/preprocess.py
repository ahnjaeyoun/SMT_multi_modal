import os
import json
import glob
import shutil
import random
from pathlib import Path
import numpy as np
import pandas as pd

class YOLODatasetPreprocessor:
    def __init__(self, project_root, random_seed=42):
        # 프로젝트 루트 기반 절대 경로 자동 추적
        self.project_root = Path(project_root)
        
        # base_data의 물리적 격리 구조 매핑
        self.base_data_path = self.project_root / 'dataset' / '1.base_data'
        self.raw_image_base = self.base_data_path / 'images'
        self.raw_label_base = self.base_data_path / 'labels_json'
        
        # 타겟 저장소: models/3.detection_pr/dataset
        self.dataset_path = self.project_root / 'models' / '3.detection_pr' / 'dataset'
        self.metadata_path = self.dataset_path / 'metadata'
        
        self.random_seed = random_seed
        self.classes = {}  # category_id를 key로 하는 딕셔너리
        
        # 통계 스키마 유지 (test 제외)
        self.statistics = {
            'total_images': 0,
            'total_annotations': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'class_distribution': {
                'total': {},
                'train': {},
                'val': {}
            },
            'split_distribution': {},
            'image_resolution': set(),
            'random_seed': random_seed,
            'reasons': {
                'missing_image': 0,
                'no_annotations': 0
            }
        }
        
        # 매핑되지 않는 파일 기록 보관소
        self.unmapped_files = []
        
        # 백업 기록용 어노테이션 데이터셋 생성 (test 제외)
        self.annotations_data = {
            'train': [],
            'val': []
        }
        
        self.resolution_stats = {}
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # YOLOv11 표준 학습 디렉토리 구조 자동 생성
        self.create_directory_structure()
        
    def create_directory_structure(self):

        # YOLO 학습을 위한 디렉토리 구조 생성 (train/val 집중)
        for split in ['train', 'val']:
            (self.dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
            
    def load_categories(self, json_file):
        # JSON 파일에서 카테고리 정보 로드
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for category in data.get('categories', []):
                self.classes[category['id']] = category['name']
                
    def process_annotation(self, json_path, target_img_name, split_name):
        # JSON 파일에서 annotation 정보 추출 후 YOLO 스케일 포맷으로 변환
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not self.classes:
            self.load_categories(json_path)
            
        # 원본 이미지 메타 매핑
        img_info = None
        for img in data.get('images', []):
            if img['file_name'] == target_img_name:
                img_info = img
                break
                
        if img_info is None:
            # 안전장치: 단일 이미지 셋 구조일 경우 첫 번째 인덱스 자동 차용
            if data.get('images'):
                img_info = data['images'][0]
            else:
                return None, []
            
        img_width = img_info['width']
        img_height = img_info['height']
        
        resolution = f"{img_width}x{img_height}"
        self.statistics['image_resolution'].add(resolution)
        self.statistics['total_annotations'] += len(data.get('annotations', []))
        
        if resolution not in self.resolution_stats:
            self.resolution_stats[resolution] = 0
        self.resolution_stats[resolution] += 1

        yolo_annotations = []
        for ann in data.get('annotations', []):
            category_id = ann['category_id']
            class_name = self.classes[category_id]
            
            self.statistics['class_distribution']['total'][class_name] = \
                self.statistics['class_distribution']['total'].get(class_name, 0) + 1
            
            x, y, w, h = ann['bbox']
            
            # YOLO 정규화 좌표계 변환식 적용
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            yolo_annotations.append({
                'category_id': category_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'original_bbox': [x, y, x+w, y+h]
            })
            
        return target_img_name, yolo_annotations
        
    def save_yolo_annotation(self, annotations, output_path, split_name, img_name, src_img_path):
        # YOLO 포맷 텍스트 파일 저장 및 통합 백업용 데이터 수집
        with open(output_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                category_id = ann['category_id']
                class_name = self.classes[category_id]
                self.statistics['class_distribution'][split_name][class_name] = \
                    self.statistics['class_distribution'][split_name].get(class_name, 0) + 1
                
                x1, y1, x2, y2 = ann['original_bbox']
                self.annotations_data[split_name].append({
                    'image_path': str(src_img_path),
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'class_name': class_name
                })
                
                # YOLO 클래스 인덱스는 0부터 시작하므로 category_id - 1 보정 처리
                f.write(f"{category_id-1} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")

    def process_pipeline(self):
        # 기존 물리 디렉토리 구조(train/val) 및 pr 세부 폴더 패스를 그대로 살려 전처리 일괄 수행
        print("=== SMT Post-Reflow(PR) 데이터셋 전처리 파이프라인 가공 시작 ===")
        
        for split in ['train', 'val']:
            # 모든 json 스캔
            target_label_dir = self.raw_label_base / split / 'pr'
            target_image_dir = self.raw_image_base / split / 'pr'
            
            if not target_label_dir.exists():
                print(f"경고: 경로가 존재하지 않습니다. 건너뜁니다 -> {target_label_dir}")
                continue
                
            json_files = list(target_label_dir.glob('*.json'))
            print(f"\n[{split.upper()}] 매핑 진행 중... 총 {len(json_files)}개 라벨 포착")
            
            for json_file in json_files:
                self.statistics['total_images'] += 1
                
                # 원천 이미지 확장자 대소문자 매칭 유연화
                img_name = json_file.with_suffix('.JPG').name
                src_img = target_image_dir / img_name
                if not src_img.exists():
                    img_name = json_file.with_suffix('.jpg').name
                    src_img = target_image_dir / img_name
                
                if src_img.exists():
                    # 어노테이션 유효성 검사 및 수집
                    res_img_name, yolo_annotations = self.process_annotation(json_file, img_name, split)
                    
                    if res_img_name is not None and yolo_annotations:
                        # 타겟 YOLO 이미지 폴더로 물리 복사 이동
                        dst_img = self.dataset_path / split / 'images' / img_name
                        shutil.copy(src_img, dst_img)
                        
                        # YOLO 레이블 텍스트(.txt) 빌드 및 저장
                        label_name = json_file.with_suffix('.txt').name
                        label_path = self.dataset_path / split / 'labels' / label_name
                        self.save_yolo_annotation(yolo_annotations, label_path, split, img_name, src_img)
                        
                        self.statistics['valid_images'] += 1
                    else:
                        self.statistics['invalid_images'] += 1
                        self.statistics['reasons']['no_annotations'] += 1
                        self.unmapped_files.append({
                            'json_path': str(json_file),
                            'image_path': str(src_img),
                            'reason': 'no_annotations'
                        })
                else:
                    self.statistics['invalid_images'] += 1
                    self.statistics['reasons']['missing_image'] += 1
                    self.unmapped_files.append({
                        'json_path': str(json_file),
                        'image_path': str(src_img),
                        'reason': 'missing_image'
                    })
                    
        # 일괄 변환 후 아티팩트 빌드 자동화
        self.save_metadata_artifacts()

    def save_metadata_artifacts(self):
        """YOLOv11 가동을 위한 메타데이터 파일 아티팩트 일괄 영구 저장"""
        # classes.txt 내보내기
        sorted_classes = sorted([(k-1, v) for k, v in self.classes.items()], key=lambda x: x[0])
        with open(self.metadata_path / 'classes.txt', 'w', encoding='utf-8') as f:
            for _, cls in sorted_classes:
                f.write(f"{cls}\n")
                
        # dataset.yaml 내보내기 (윈도우 역슬래시 충돌 방지를 위해 .as_posix() 세팅)
        yaml_path = self.metadata_path / 'dataset.yaml'
        yaml_content = f"""# SMT Multi-Modal Object Detection Configuration
path: {self.dataset_path.as_posix()}  # dataset root dir
train: train/images  # train images
val: val/images  # val images

# Classes
names:
"""
        for idx, (_, cls) in enumerate(sorted_classes):
            yaml_content += f"  {idx}: {cls}\n"
            
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
            
        # 데이터프레임 구조 연동 백업 CSV 내보내기
        for split in ['train', 'val']:
            if self.annotations_data[split]:
                pd.DataFrame(self.annotations_data[split]).to_csv(
                    self.metadata_path / f"{split}_annotations.csv", index=False, encoding='utf-8-sig'
                )
        
        if self.unmapped_files:
            pd.DataFrame(self.unmapped_files).to_csv(
                self.metadata_path / 'unmapped_files.csv', index=False, encoding='utf-8-sig'
            )
            
        # 분석 리포트 요약 텍스트 출력
        self.save_statistics_report()

    def save_statistics_report(self):
        # 데이터셋 통계 정보를 파일로 요약 및 프로세스 종료 마킹
        stats_file = self.metadata_path / 'dataset_statistics.txt'
        
        for split in ['train', 'val']:
            img_count = len(list((self.dataset_path / split / 'images').glob('*')))
            self.statistics['split_distribution'][split] = img_count
            
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("SMT PR 공정 데이터 가공 리포트 검증 통계\n")
            f.write("="*50 + "\n")
            f.write(f"총 스캔 파일 수: {self.statistics['total_images']}개\n")
            f.write(f"정상 가공 파일: {self.statistics['valid_images']}개\n")
            f.write(f"제외 처리 파일: {self.statistics['invalid_images']}개\n\n")
            
            f.write("데이터셋 분할 정보 (이미지 카운트 기준)\n")
            f.write("-"*30 + "\n")
            for split in ['train', 'val']:
                count = self.statistics['split_distribution'].get(split, 0)
                f.write(f"- {split}: {count}개\n")
            f.write("\n")

            f.write("제외 상세 사유:\n")
            for reason, count in self.statistics['reasons'].items():
                f.write(f"- {reason}: {count}개\n")
            f.write("\n")

            f.write("종합 검출 클래스별 객체 통계:\n")
            f.write("-"*30 + "\n")
            for cls, count in sorted(self.statistics['class_distribution']['total'].items()):
                f.write(f"- {cls}: {count}개\n")
            f.write("\n")

            for split in ['train', 'val']:
                f.write(f"[{split.upper()} 데이터셋 세부 클래스 분포]\n")
                dist = self.statistics['class_distribution'][split]
                for cls, count in sorted(dist.items()):
                    f.write(f" * {cls}: {count}개\n")
                f.write("\n")

            f.write("해상도 통계\n")
            f.write("="*50 + "\n")
            for res, count in sorted(self.resolution_stats.items()):
                f.write(f"- {res}: {count}개\n")
                
        print(f"\n=== 전처리 완료 ===")
        print(f"- 유효 파일 파싱 성공: {self.statistics['valid_images']}개")
        print(f"- 통계 분석 리포트 배포 완료 -> {self.metadata_path}")

if __name__ == "__main__":
    # 로컬 경로 변수 설정 연동
    LOCAL_PROJECT_ROOT = "C:/SMT_multi_modal"
    
    preprocessor = YOLODatasetPreprocessor(project_root=LOCAL_PROJECT_ROOT)
    preprocessor.process_pipeline()