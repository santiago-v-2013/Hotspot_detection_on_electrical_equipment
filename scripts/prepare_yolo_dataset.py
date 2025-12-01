#!/usr/bin/env python3
"""
Prepare dataset for YOLO training by creating data.yaml configuration
and splitting into train/val/test sets.
"""

import argparse
import shutil
from pathlib import Path
import yaml
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split dataset into train/val/test sets.
    
    Args:
        source_dir: Source directory with equipment folders
        output_dir: Output directory for split dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    random.seed(seed)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Collect all images with annotations
    all_samples = []
    
    for equipment_dir in source_dir.iterdir():
        if not equipment_dir.is_dir():
            continue
        
        for img_path in equipment_dir.glob('*.jpg'):
            label_path = img_path.with_suffix('.txt')
            
            if label_path.exists():
                all_samples.append({
                    'image': img_path,
                    'label': label_path,
                    'equipment': equipment_dir.name
                })
    
    # Shuffle samples
    random.shuffle(all_samples)
    
    # Split
    total = len(all_samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': all_samples[:train_end],
        'val': all_samples[train_end:val_end],
        'test': all_samples[val_end:]
    }
    
    # Copy files
    for split_name, samples in splits.items():
        logger.info(f"Processing {split_name} split: {len(samples)} samples")
        
        for sample in tqdm(samples, desc=f"  Copying {split_name}"):
            # Generate unique filename with equipment prefix
            filename = f"{sample['equipment']}_{sample['image'].name}"
            
            # Copy image
            dst_img = output_dir / split_name / 'images' / filename
            shutil.copy(sample['image'], dst_img)
            
            # Copy label
            dst_label = output_dir / split_name / 'labels' / Path(filename).with_suffix('.txt').name
            shutil.copy(sample['label'], dst_label)
    
    logger.info("\nDataset split complete:")
    logger.info(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/total*100:.1f}%)")
    logger.info(f"  Val: {len(splits['val'])} samples ({len(splits['val'])/total*100:.1f}%)")
    logger.info(f"  Test: {len(splits['test'])} samples ({len(splits['test'])/total*100:.1f}%)")
    
    return splits


def create_data_yaml(
    output_dir: Path,
    num_classes: int = 1,
    class_names: list = None
):
    """
    Create YOLO data.yaml configuration file.
    
    Args:
        output_dir: Dataset output directory
        num_classes: Number of object classes
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['hotspot']
    
    data_config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logger.info(f"\nYOLO configuration saved to: {yaml_path}")
    logger.info(f"Configuration:")
    logger.info(f"  Dataset path: {data_config['path']}")
    logger.info(f"  Classes: {num_classes}")
    logger.info(f"  Class names: {class_names}")
    
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare hotspot detection dataset for YOLO training'
    )
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=Path('data/processed/hotspot_detection'),
        help='Source directory with annotated images'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/yolo_detection'),
        help='Output directory for split dataset'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1,
        help='Number of object classes'
    )
    parser.add_argument(
        '--class-names',
        nargs='+',
        default=['hotspot'],
        help='List of class names'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splitting'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        logger.error("Train/val/test ratios must sum to 1.0")
        return
    
    if not args.source_dir.exists():
        logger.error(f"Source directory does not exist: {args.source_dir}")
        return
    
    logger.info("="*60)
    logger.info("Preparing YOLO Detection Dataset")
    logger.info("="*60)
    logger.info(f"Source: {args.source_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Split: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    logger.info("")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split dataset
    splits = split_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Create YOLO configuration
    yaml_path = create_data_yaml(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        class_names=args.class_names
    )
    
    logger.info("\n" + "="*60)
    logger.info("Dataset preparation complete!")
    logger.info("="*60)
    logger.info(f"\nYou can now train YOLO with:")
    logger.info(f"  python scripts/train_hotspot_detection.py --model yolov8 --data {yaml_path}")


if __name__ == '__main__':
    main()
