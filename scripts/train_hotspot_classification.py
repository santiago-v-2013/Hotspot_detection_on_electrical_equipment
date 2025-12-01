#!/usr/bin/env python3
"""
Fine-tuning script for hotspot classification models.

This script fine-tunes EfficientNet-B0 and DenseNet-121 models for binary
hotspot classification (hotspot vs no_hotspot).
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger_cfg.setup import setup_logging
from src.finetuning.hotspot_dataset import get_data_loaders
from src.finetuning.hotspot_models import get_model, get_model_info
from src.finetuning.trainer import Trainer


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss applies a modulating term to the cross entropy loss in order to
    focus learning on hard misclassified examples. It is especially useful for
    addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               or a list of weights for each class.
        gamma: Focusing parameter for modulating loss (default: 2).
        reduction: Specifies the reduction to apply to the output.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (tensor) predicted logits, shape [batch_size, num_classes]
            targets: (tensor) ground truth labels, shape [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune models for hotspot classification'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/hotspot_classification',
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
        choices=['efficientnet_b0', 'densenet121', 'both'],
        default='efficientnet_b0',
        help='Model to train (default: efficientnet_b0)'
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
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout probability (default: 0.3)'
    )
    
    # Scheduler arguments
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['step', 'cosine', 'plateau', 'none'],
        default='step',
        help='Learning rate scheduler (default: step)'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=10,
        help='Step size for StepLR scheduler (default: 10)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='Gamma for StepLR scheduler (default: 0.1)'
    )
    
    # Early stopping
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=None,
        help='Early stopping patience (None = disabled)'
    )
    
    # Data loading
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    # Output
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/hotspot_classification',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for log files'
    )
    
    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def train_model(
    model_name: str,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    test_loader,
    class_names: list,
    dataset_info: dict,
    logger
):
    """Train a single model."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name.upper()}")
    logger.info(f"{'='*80}")
    
    # Get model info
    model_info = get_model_info(model_name)
    if model_info:
        logger.info(f"Model: {model_info['name']}")
        logger.info(f"Input size: {model_info['input_size']}")
        logger.info(f"Parameters: {model_info['params']}")
        logger.info(f"Description: {model_info['description']}")
    
    # Create model
    pretrained = args.pretrained and not args.no_pretrained
    model = get_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout
    )
    
    # Count parameters
    total_params = count_parameters(model)
    logger.info(f"Trainable parameters: {total_params:,}")
    
    # Move to device
    device = torch.device(args.device)
    model = model.to(device)
    
    # Calculate class weights to handle imbalanced dataset
    class_counts = [
        dataset_info['class_distribution']['no_hotspot'],
        dataset_info['class_distribution']['hotspot']
    ]
    total_samples = sum(class_counts)
    class_weights = torch.FloatTensor([
        total_samples / (len(class_counts) * count) for count in class_counts
    ]).to(device)
    
    logger.info(f"Class distribution: no_hotspot={class_counts[0]}, hotspot={class_counts[1]}")
    logger.info(f"Class weights: no_hotspot={class_weights[0]:.4f}, hotspot={class_weights[1]:.4f}")
    
    # Use Focal Loss with class weights for imbalanced dataset
    # Focal Loss focuses on hard examples and down-weights easy ones
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    logger.info(f"Using Focal Loss with gamma=2.0 and class weights")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
    else:
        scheduler = None
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        class_names=class_names
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    
    # Use the validate method but on test loader
    self_model = trainer.model
    self_model.eval()
    test_running_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = self_model(inputs)
            loss = trainer.criterion(outputs, labels)
            
            test_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = accuracy_score(test_labels, test_preds)
    
    # Calculate per-class metrics
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='weighted', zero_division=0
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    test_metrics = {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': test_loss,
        'predictions': test_preds,
        'labels': test_labels,
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"\nTest Set Results for {model_name}:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1']:.4f}")
    
    # Save test metrics as JSON
    metrics_to_save = {
        'accuracy': float(test_metrics['accuracy']),
        'precision': float(test_metrics['precision']),
        'recall': float(test_metrics['recall']),
        'f1': float(test_metrics['f1']),
        'loss': float(test_metrics['loss']),
        'confusion_matrix': test_metrics['confusion_matrix']
    }
    metrics_file = checkpoint_dir / 'test_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    logger.info(f"\nTest metrics saved to: {metrics_file}")
    logger.info(f"\nTest Metrics JSON:")
    logger.info(json.dumps(metrics_to_save, indent=2))
    
    # Save plots
    logger.info("\nSaving training plots...")
    trainer.plot_history(save_path=str(checkpoint_dir / 'training_history.png'))
    trainer.plot_confusion_matrix(
        test_metrics,
        save_path=str(checkpoint_dir / 'confusion_matrix.png')
    )
    
    logger.info(f"\nModel saved to: {checkpoint_dir}")
    
    return history, test_metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    log_file = Path(args.log_dir) / 'hotspot_classification_training.log'
    logger = setup_logging(log_file=str(log_file))
    
    logger.info("Hotspot Classification Fine-tuning")
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
    models_to_train = []
    if args.model == 'both':
        models_to_train = ['efficientnet_b0', 'densenet121']
    else:
        models_to_train = [args.model]
    
    logger.info(f"\nModels to train: {', '.join(models_to_train)}")
    
    # Get appropriate image size
    image_size = 224  # Both models use 224x224
    
    # Load data (shared across models)
    logger.info("\nLoading dataset...")
    train_loader, val_loader, test_loader, dataset_info = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=not args.no_augmentation
    )
    
    logger.info("\nDataset Information:")
    logger.info(f"  Total samples: {dataset_info['total_samples']}")
    logger.info(f"  Train: {dataset_info['train_samples']}")
    logger.info(f"  Validation: {dataset_info['val_samples']}")
    logger.info(f"  Test: {dataset_info['test_samples']}")
    logger.info(f"  Classes: {dataset_info['classes']}")
    logger.info(f"  Class distribution: {dataset_info['class_distribution']}")
    logger.info(f"\n  Distribution by equipment:")
    for equipment, dist in dataset_info['equipment_distribution'].items():
        logger.info(f"    {equipment}: {dist}")
    
    class_names = dataset_info['classes']
    
    # Train each model
    results = {}
    for model_name in models_to_train:
        try:
            history, test_metrics = train_model(
                model_name=model_name,
                args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_names=class_names,
                dataset_info=dataset_info,
                logger=logger
            )
            results[model_name] = {
                'history': history,
                'test_metrics': test_metrics
            }
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    for model_name, result in results.items():
        test_metrics = result['test_metrics']
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1-Score: {test_metrics['f1']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")


if __name__ == '__main__':
    main()
