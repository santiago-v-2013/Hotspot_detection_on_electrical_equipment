#!/usr/bin/env python3
"""
Train object detection models for hotspot detection.

Supports:
- YOLOv8 (nano, small, medium, large, xlarge)
- YOLOv11 (nano, small, medium, large, xlarge) - Latest version
"""

import argparse
import torch
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.models import get_model
from src.detection.trainer import DetectionTrainer
from src.logger_cfg.setup import setup_logger


def train_yolo(args):
    """Train YOLO model (YOLOv8 or YOLOv11)."""
    logger = logging.getLogger(__name__)
    
    # Determine model version
    model_type = args.model
    if model_type in ['yolov11', 'yolo11']:
        model_name = f"YOLOv11{args.yolo_size}"
    else:
        model_name = f"YOLOv8{args.yolo_size}"
    
    logger.info("="*60)
    logger.info(f"Training {model_name}")
    logger.info("="*60)
    
    # Check if data.yaml exists
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        logger.error(f"YOLO data configuration not found: {data_yaml}")
        logger.info("\nPlease run: python scripts/prepare_yolo_dataset.py")
        return
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Create model
    model = get_model(
        model_type=model_type,
        model_size=args.yolo_size,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        device=device
    )
    
    # Prepare save directory
    if model_type in ['yolov11', 'yolo11']:
        save_dir = Path(args.save_dir) / f'yolov11{args.yolo_size}'
    else:
        save_dir = Path(args.save_dir) / f'yolov8{args.yolo_size}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Image size: {args.img_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Pretrained: {args.pretrained}")
    logger.info(f"  Save directory: {save_dir}")
    logger.info("")
    
    # Train
    results = DetectionTrainer.train_yolo(
        model=model,
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr0=args.lr,
        save_dir=save_dir,
        patience=args.early_stopping,
        save_period=args.save_period,
        workers=args.num_workers,
        verbose=True
    )
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Best model saved to: {save_dir / 'weights' / 'best.pt'}")
    
    # Clean up GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()
        logger.info("\nGPU memory cleared")


def main():
    parser = argparse.ArgumentParser(
        description='Train object detection models for hotspot detection'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['yolov8', 'yolov11', 'yolo11'],
        help='Model type to train (yolov8, yolov11)'
    )
    
    # YOLO specific
    parser.add_argument(
        '--yolo-size',
        type=str,
        default='s',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO model size'
    )
    parser.add_argument(
        '--data-yaml',
        type=Path,
        default=Path('data/yolo_detection/data.yaml'),
        help='Path to YOLO data.yaml configuration'
    )
    
    # Common training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Image size (square)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1,
        help='Number of object classes'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights'
    )
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=Path('models/hotspot_detection'),
        help='Directory to save models'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        default=Path('logs/hotspot_detection_training.log'),
        help='Path to log file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(
        name=__name__,
        log_file=args.log_file,
        level=logging.INFO
    )
    
    # Print banner
    logger.info("="*60)
    logger.info("Hotspot Detection Training")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60)
    
    # Train model
    train_yolo(args)


if __name__ == '__main__':
    main()
