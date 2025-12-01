"""
Training utilities for equipment classification models.
"""

import os
import time
import copy
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for fine-tuning classification models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            class_names: List of class names
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler (optional)
            device: Device to use ('cuda' or 'cpu', auto-detected if None)
            checkpoint_dir: Directory to save checkpoints
            log_interval: Batches between logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.log_interval = log_interval
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Setup criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_wts = None
        self.epochs_no_improve = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log progress
            if (batch_idx + 1) % self.log_interval == 0:
                logger.info(
                    f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': epoch_acc,
            'loss': epoch_loss,
            'per_class': {
                self.class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(self.num_classes)
            },
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        return epoch_loss, epoch_acc, metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop training if no improvement for this many epochs
            save_best_only: Only save checkpoints that improve validation accuracy
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validation phase
            val_loss, val_acc, val_metrics = self.validate()
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log per-class metrics
            logger.info("\nPer-class metrics:")
            for class_name, metrics in val_metrics['per_class'].items():
                logger.info(
                    f"  {class_name}: "
                    f"P={metrics['precision']:.3f}, "
                    f"R={metrics['recall']:.3f}, "
                    f"F1={metrics['f1']:.3f}, "
                    f"N={metrics['support']}"
                )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                # ReduceLROnPlateau needs the metric to monitor
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint if improved
            if val_acc > self.best_val_acc:
                logger.info(f"Validation accuracy improved from {self.best_val_acc:.4f} to {val_acc:.4f}")
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                
                # Save checkpoint
                self.save_checkpoint(epoch + 1, val_metrics, is_best=True)
            else:
                self.epochs_no_improve += 1
                logger.info(f"No improvement for {self.epochs_no_improve} epoch(s)")
                
                if not save_best_only:
                    self.save_checkpoint(epoch + 1, val_metrics, is_best=False)
            
            # Early stopping
            if self.epochs_no_improve >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch time: {epoch_time:.2f}s")
        
        # Load best model weights
        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)
            logger.info(f"\nLoaded best model with validation accuracy: {self.best_val_acc:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"\nTotal training time: {total_time / 60:.2f} minutes")
        
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics dictionary
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'class_names': self.class_names,
            'metrics': metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
            
            # Also save metrics to JSON
            metrics_path = self.checkpoint_dir / 'best_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Acc', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        else:
            plt.savefig(self.checkpoint_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrix(self, metrics: Dict, save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            metrics: Metrics dictionary containing confusion matrix
            save_path: Path to save the plot (optional)
        """
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            plt.savefig(self.checkpoint_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        plt.close()
