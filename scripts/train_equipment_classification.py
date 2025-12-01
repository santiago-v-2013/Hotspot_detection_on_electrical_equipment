#!/usr/bin/env python3
"""
Fine-tuning script for equipment classification models.

This script fine-tunes ResNet-50 and Inception-v3 models for classifying
electrical equipment into different categories.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger_cfg.setup import setup_logging
from src.finetuning.dataset import get_data_loaders
from src.finetuning.models import get_model, count_parameters
from src.finetuning.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune models for equipment classification'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/equipment_classification',
        help='Path to the processed data directory'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split fraction (default: 0.2)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Test split fraction (default: 0.1)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet50', 'inception_v3', 'both'],
        default='resnet50',
        help='Model to train (default: resnet50)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use ImageNet pre-trained weights (default: True)'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Do not use pre-trained weights'
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze backbone layers (only train classifier head)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Early stopping patience in epochs (default: 10)'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['step', 'cosine', 'plateau', 'none'],
        default='step',
        help='Learning rate scheduler (default: step)'
    )
    
    # Other arguments
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/equipment_classification',
        help='Directory to save checkpoints (default: models/equipment_classification)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for log files (default: logs)'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_scheduler(optimizer, scheduler_type: str, num_epochs: int):
    """Create learning rate scheduler."""
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1)
    else:
        return None


def train_model(args, model_type: str, logger):
    """Train a single model."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_type.upper()} model")
    logger.info(f"{'='*80}\n")
    
    # Determine input size based on model
    input_size = 299 if model_type == 'inception_v3' else 224
    
    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        input_size=input_size,
        model_type='inception' if model_type == 'inception_v3' else 'resnet',
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    num_classes = len(class_names)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")
    
    # Create model
    logger.info(f"\nCreating {model_type} model...")
    pretrained = args.pretrained and not args.no_pretrained
    model = get_model(
        model_type=model_type,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=args.freeze_backbone
    )
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / model_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=10
    )
    
    # Train model
    logger.info("\nStarting training...\n")
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_best_only=True
    )
    
    # Plot training history
    trainer.plot_history()
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc, test_metrics = trainer.validate()
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Log per-class test metrics
    logger.info("\nTest set per-class metrics:")
    for class_name, metrics in test_metrics['per_class'].items():
        logger.info(
            f"  {class_name}: "
            f"P={metrics['precision']:.3f}, "
            f"R={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"N={metrics['support']}"
        )
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(test_metrics)
    
    # Save test metrics
    import json
    test_metrics_path = checkpoint_dir / 'test_metrics.json'
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"\nSaved test metrics to {test_metrics_path}")
    
    logger.info(f"\n{model_type.upper()} training completed!")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    return trainer, test_metrics


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    log_file = Path(args.log_dir) / 'equipment_classification_training.log'
    logger = setup_logging(log_file=str(log_file))
    
    logger.info("Equipment Classification Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 80)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("CUDA is not available. Using CPU")
    
    # Train models
    results = {}
    
    if args.model == 'both':
        # Train both models
        for model_type in ['resnet50', 'inception_v3']:
            trainer, metrics = train_model(args, model_type, logger)
            results[model_type] = {
                'trainer': trainer,
                'metrics': metrics
            }
    else:
        # Train single model
        trainer, metrics = train_model(args, args.model, logger)
        results[args.model] = {
            'trainer': trainer,
            'metrics': metrics
        }
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    for model_type, result in results.items():
        trainer = result['trainer']
        metrics = result['metrics']
        logger.info(f"\n{model_type.upper()}:")
        logger.info(f"  Best validation accuracy: {trainer.best_val_acc:.4f}")
        logger.info(f"  Test accuracy: {metrics['accuracy']:.4f}")
    
    logger.info("\nAll training completed successfully!")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")


if __name__ == '__main__':
    main()
