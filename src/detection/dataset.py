"""  
Dataset classes for object detection.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class YOLODataset(Dataset):
    """
    Dataset for YOLO format annotations.
    
    Directory structure:
        data_dir/
            Equipment1/
                image1.jpg
                image1.txt  # YOLO format: class x_center y_center width height
                image2.jpg
                image2.txt
            Equipment2/
                ...
    """
    
    def __init__(
        self,
        data_dir: Path,
        img_size: int = 640,
        augment: bool = False,
        include_no_annotations: bool = False
    ):
        """
        Args:
            data_dir: Root directory containing equipment folders
            img_size: Target image size (square)
            augment: Apply data augmentation
            include_no_annotations: Include images without annotations
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Find all images with annotations
        self.samples = []
        
        # Check if using YOLO dataset structure (train/images, train/labels)
        images_dir = self.data_dir / 'images'
        labels_dir = self.data_dir / 'labels'
        
        if images_dir.exists() and labels_dir.exists():
            # YOLO structure: data_dir/images/*.jpg, data_dir/labels/*.txt
            for img_path in images_dir.glob('*.jpg'):
                # Extract equipment name from filename (format: Equipment_filename.jpg)
                filename = img_path.stem
                equipment = filename.split('_')[0] if '_' in filename else 'unknown'
                
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    self.samples.append({
                        'image': img_path,
                        'label': label_path,
                        'equipment': equipment
                    })
                elif include_no_annotations:
                    self.samples.append({
                        'image': img_path,
                        'label': None,
                        'equipment': equipment
                    })
        else:
            # Original structure: equipment directories with images and labels
            for equipment_dir in self.data_dir.iterdir():
                if not equipment_dir.is_dir():
                    continue
                
                # Find all images
                for img_path in equipment_dir.glob('*.jpg'):
                    label_path = img_path.with_suffix('.txt')
                    
                    if label_path.exists():
                        self.samples.append({
                            'image': img_path,
                            'label': label_path,
                            'equipment': equipment_dir.name
                        })
                    elif include_no_annotations:
                        self.samples.append({
                            'image': img_path,
                            'label': None,
                            'equipment': equipment_dir.name
                        })
        
        # Define augmentations
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1, clip=True))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1, clip=True))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image']).convert('RGB')
        image = np.array(image)
        
        # Load annotations
        bboxes = []
        class_labels = []
        
        if sample['label'] is not None:
            with open(sample['label'], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # Clip coordinates to [0, 1] range to avoid precision errors
                        x_center = max(0.0, min(1.0, float(parts[1])))
                        y_center = max(0.0, min(1.0, float(parts[2])))
                        width = max(0.0, min(1.0, float(parts[3])))
                        height = max(0.0, min(1.0, float(parts[4])))
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        # Apply transforms
        if len(bboxes) > 0:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            # Clip transformed bboxes to [0, 1] range to avoid precision errors
            clipped_bboxes = []
            for bbox in transformed['bboxes']:
                clipped_bbox = [
                    max(0.0, min(1.0, bbox[0])),  # x_center
                    max(0.0, min(1.0, bbox[1])),  # y_center
                    max(0.0, min(1.0, bbox[2])),  # width
                    max(0.0, min(1.0, bbox[3]))   # height
                ]
                clipped_bboxes.append(clipped_bbox)
            transformed['bboxes'] = clipped_bboxes
        else:
            # No annotations, just transform image
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
        
        return {
            'image': transformed['image'],
            'bboxes': torch.tensor(transformed['bboxes'], dtype=torch.float32) if len(transformed['bboxes']) > 0 else torch.zeros((0, 4)),
            'labels': torch.tensor(transformed['class_labels'], dtype=torch.int64) if len(transformed['class_labels']) > 0 else torch.zeros((0,), dtype=torch.int64),
            'image_path': str(sample['image']),
            'equipment': sample['equipment']
        }


def collate_fn_yolo(batch: List[Dict]) -> Dict:
    """Custom collate function for YOLO dataset."""
    images = torch.stack([item['image'] for item in batch])
    
    # Pad bboxes and labels to same length
    max_boxes = max([item['bboxes'].shape[0] for item in batch])
    
    bboxes_batch = []
    labels_batch = []
    
    for item in batch:
        num_boxes = item['bboxes'].shape[0]
        if num_boxes > 0:
            # Pad if necessary
            if num_boxes < max_boxes:
                pad_boxes = torch.zeros((max_boxes - num_boxes, 4), dtype=torch.float32)
                pad_labels = torch.full((max_boxes - num_boxes,), -1, dtype=torch.int64)
                
                bboxes = torch.cat([item['bboxes'], pad_boxes], dim=0)
                labels = torch.cat([item['labels'], pad_labels], dim=0)
            else:
                bboxes = item['bboxes']
                labels = item['labels']
        else:
            bboxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
            labels = torch.full((max_boxes,), -1, dtype=torch.int64)
        
        bboxes_batch.append(bboxes)
        labels_batch.append(labels)
    
    return {
        'images': images,
        'bboxes': torch.stack(bboxes_batch),
        'labels': torch.stack(labels_batch),
        'image_paths': [item['image_path'] for item in batch],
        'equipment': [item['equipment'] for item in batch]
    }
