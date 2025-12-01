"""
Model architectures for hotspot classification.

This module provides EfficientNet-B0 and DenseNet-121 models
for binary hotspot classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 model for hotspot classification.
    
    EfficientNet uses compound scaling to balance network depth, width,
    and resolution for improved performance.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        """
        Initialize EfficientNet-B0 classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone layers initially
            dropout: Dropout probability for the classifier
        """
        super(EfficientNetB0Classifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
            logger.info("Loaded pretrained EfficientNet-B0 weights")
        else:
            self.model = models.efficientnet_b0(weights=None)
            logger.info("Initialized EfficientNet-B0 without pretrained weights")
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        logger.info("Frozen EfficientNet-B0 backbone")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen EfficientNet-B0 backbone")


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 model for hotspot classification.
    
    DenseNet connects each layer to every other layer in a feed-forward fashion,
    encouraging feature reuse and reducing the number of parameters.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        """
        Initialize DenseNet-121 classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone layers initially
            dropout: Dropout probability for the classifier
        """
        super(DenseNet121Classifier, self).__init__()
        
        # Load pretrained DenseNet-121
        if pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
            self.model = models.densenet121(weights=weights)
            logger.info("Loaded pretrained DenseNet-121 weights")
        else:
            self.model = models.densenet121(weights=None)
            logger.info("Initialized DenseNet-121 without pretrained weights")
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        # Replace classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        logger.info("Frozen DenseNet-121 backbone")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen DenseNet-121 backbone")


def get_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3
) -> nn.Module:
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Name of the model ('efficientnet_b0' or 'densenet121')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone initially
        dropout: Dropout probability
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    
    if model_name == 'efficientnet_b0':
        return EfficientNetB0Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    elif model_name == 'densenet121':
        return DenseNet121Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: 'efficientnet_b0', 'densenet121'"
        )


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    info = {
        'efficientnet_b0': {
            'name': 'EfficientNet-B0',
            'input_size': 224,
            'params': '5.3M',
            'description': 'Compound scaling CNN with efficient architecture'
        },
        'densenet121': {
            'name': 'DenseNet-121',
            'input_size': 224,
            'params': '8.0M',
            'description': 'Densely connected CNN with feature reuse'
        }
    }
    
    return info.get(model_name.lower(), {})


def get_hotspot_model(
    model_name: str = 'efficientnet_b0',
    num_classes: int = 2,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda'
) -> nn.Module:
    """
    Load a trained hotspot classification model.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of classes (default: 2 for binary classification)
        pretrained: Whether to use ImageNet pretrained weights as base
        checkpoint_path: Path to saved model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Loaded model ready for inference
    """
    # Create model
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=False
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
                if 'epoch' in checkpoint:
                    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
                if 'best_val_accuracy' in checkpoint:
                    logger.info(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded model state dict from: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
            logger.info("Using model with ImageNet pretrained weights only")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model
