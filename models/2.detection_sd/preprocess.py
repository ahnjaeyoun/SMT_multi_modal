import os
import json
import glob
import shutil
import random
from pathlib import Path
import numpy as np
import pandas as pd

def get_project_root():
    env_root = os.getenv('PROJECT_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]

class YOLODatasetPreprocessor:
    def __init__(self, project_root, random_seed=42):
        self.project_root = Path(project_root).resolve()
        
        # 원천 데이터 레이어 소스 경로
        self.base_data_path = self.project_root / 'dataset' / '1.base_data'
        self.raw_image_base = self.base_data_path / 'images'
        self.raw_label_base = self.base_data_path / 'labels_json'
        
        self.dataset_path = self.project_root / 'dataset' / '5.models' / 'detection_sd'
        self.metadata_path = self.dataset_path / 'metadata'
        
        self.random_seed = random_seed
        self.classes = {}
        
        self.statistics = {
            'total_images': 0, 'total_annotations': 0, 'valid_images': 0, 'invalid_images': 0,
            'class_distribution': {'total': {}, 'train': {}, 'val': {}},
            'split_distribution': {}, 'image_resolution': set(), 'random_seed': random_seed,
            'reasons': {'missing_image': 0, 'no_annotations': 0}
        }
        
        self.unmapped_files = []
        self.annotations_data = {'train': [], 'val': []}
        self.resolution_stats = {}
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.create_directory_structure()
        
    def create_directory_structure(self):
        for split in ['train', 'val']:
            (self.dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
            
    def load_categories(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for category in data.get('categories', []):
                self.classes[category['id']] = category['name']
                
    def process_annotation(self, json_path, target_img_name, split_name):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not self.classes:
            self.load_categories(json_path)
            
        img_info = None
        for img in data.get('images', []):
            if img['file_name'] == target_img_name:
                img_info = img
                break
                
        if img_info is None:
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
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            yolo_annotations.append({
                'category_id': category_id, 'x_center': x_center, 'y_center': y_center,
                'width': width, 'height': height, 'original_bbox': [x, y, x+w, y+h]
            })
        return target_img_name, yolo_annotations
        
    def save_yolo_annotation(self, annotations, output_path, split_name, img_name, src_img_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                category_id = ann['category_id']
                class_name = self.classes[category_id]
                self.statistics['class_distribution'][split_name][class_name] = \
                    self.statistics['class_distribution'][split_name].get(class_name, 0) + 1
                
                x1, y1, x2, y2 = ann['original_bbox']
                image_path = Path(src_img_path)
                try:
                    image_path = image_path.resolve().relative_to(self.project_root)
                except ValueError:
                    image_path = image_path.resolve()
                self.annotations_data[split_name].append({
                    'image_path': image_path.as_posix(), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class_name': class_name
                })
                f.write(f"{category_id-1} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")

    def process_pipeline(self):
        print("=== SMT Solder Deposit(SD) 사전 공정 데이터셋 전처리 가공 시작 ===")
        for split in ['train', 'val']:
            # sd (사전 공정 부품안착 및 납도포) 타겟 스캔
            target_label_dir = self.raw_label_base / split / 'sd'
            target_image_dir = self.raw_image_base / split / 'sd'
            
            if not target_label_dir.exists():
                continue
                
            json_files = list(target_label_dir.glob('*.json'))
            print(f"[{split.upper()}] 매핑 진행 중... 총 {len(json_files)}개 라벨 포착")
            
            for json_file in json_files:
                self.statistics['total_images'] += 1
                img_name = json_file.with_suffix('.JPG').name
                src_img = target_image_dir / img_name
                if not src_img.exists():
                    img_name = json_file.with_suffix('.jpg').name
                    src_img = target_image_dir / img_name
                
                if src_img.exists():
                    res_img_name, yolo_annotations = self.process_annotation(json_file, img_name, split)
                    if res_img_name is not None and yolo_annotations:
                        dst_img = self.dataset_path / split / 'images' / img_name
                        shutil.copy(src_img, dst_img)
                        
                        label_name = json_file.with_suffix('.txt').name
                        label_path = self.dataset_path / split / 'labels' / label_name
                        self.save_yolo_annotation(yolo_annotations, label_path, split, img_name, src_img)
                        self.statistics['valid_images'] += 1
                    else:
                        self.statistics['invalid_images'] += 1
                        self.statistics['reasons']['no_annotations'] += 1
                        self.unmapped_files.append({'json_path': str(json_file), 'image_path': str(src_img), 'reason': 'no_annotations'})
                else:
                    self.statistics['invalid_images'] += 1
                    self.statistics['reasons']['missing_image'] += 1
                    self.unmapped_files.append({'json_path': str(json_file), 'image_path': str(src_img), 'reason': 'missing_image'})
                    
        self.save_metadata_artifacts()

    def save_metadata_artifacts(self):
        sorted_classes = sorted([(k-1, v) for k, v in self.classes.items()], key=lambda x: x[0])
        with open(self.metadata_path / 'classes.txt', 'w', encoding='utf-8') as f:
            for _, cls in sorted_classes:
                f.write(f"{cls}\n")
                
        yaml_path = self.metadata_path / 'dataset.yaml'
        yaml_content = f"""# SMT Multi-Modal Object Detection Configuration (Solder Deposit)
path: ..
train: train/images
val: val/images

names:
"""
        for idx, (_, cls) in enumerate(sorted_classes):
            yaml_content += f"  {idx}: {cls}\n"
            
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
            
        for split in ['train', 'val']:
            if self.annotations_data[split]:
                pd.DataFrame(self.annotations_data[split]).to_csv(self.metadata_path / f"{split}_annotations.csv", index=False, encoding='utf-8-sig')
        if self.unmapped_files:
            pd.DataFrame(self.unmapped_files).to_csv(self.metadata_path / 'unmapped_files.csv', index=False, encoding='utf-8-sig')
            
        self.save_statistics_report()

    def save_statistics_report(self):
        stats_file = self.metadata_path / 'dataset_statistics.txt'
        for split in ['train', 'val']:
            self.statistics['split_distribution'][split] = len(list((self.dataset_path / split / 'images').glob('*')))
            
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("SMT SD 공정 데이터 가공 리포트 검증 통계\n" + "="*50 + "\n")
            f.write(f"총 스캔 파일 수: {self.statistics['total_images']}개\n정상 가공 파일: {self.statistics['valid_images']}개\n제외 처리 파일: {self.statistics['invalid_images']}개\n\n")
            f.write("데이터셋 분할 정보 (이미지 카운트 기준)\n" + "-"*30 + "\n")
            for split in ['train', 'val']:
                f.write(f"- {split}: {self.statistics['split_distribution'].get(split, 0)}개\n")
            f.write("\n제외 상세 사유:\n")
            for reason, count in self.statistics['reasons'].items():
                f.write(f"- {reason}: {count}개\n")
            f.write("\n종합 검출 클래스별 객체 통계:\n" + "-"*30 + "\n")
            for cls, count in sorted(self.statistics['class_distribution']['total'].items()):
                f.write(f"- {cls}: {count}개\n")
            print(f"\n=== 전처리 완료 -> {self.metadata_path} ===")

if __name__ == "__main__":
    preprocessor = YOLODatasetPreprocessor(project_root=get_project_root())
    preprocessor.process_pipeline()
