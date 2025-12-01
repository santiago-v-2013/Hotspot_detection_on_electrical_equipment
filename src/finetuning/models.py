"""
Model definitions for equipment classification fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 model fine-tuned for equipment classification.
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super(ResNet50Classifier, self).__init__()
        
        # Load pre-trained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.model = models.resnet50(weights=weights)
            logger.info("Loaded ResNet-50 with ImageNet pre-trained weights")
        else:
            self.model = models.resnet50(weights=None)
            logger.info("Initialized ResNet-50 without pre-trained weights")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Froze ResNet-50 backbone layers")
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"Modified ResNet-50 classifier head for {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze the backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end. If None, unfreezes all.
        """
        if num_layers is None:
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Unfroze all ResNet-50 layers")
        else:
            # Unfreeze specific number of layers from the end
            layers = [self.model.layer4, self.model.layer3, self.model.layer2, self.model.layer1]
            unfrozen_count = 0
            for layer in layers[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = True
                unfrozen_count += 1
            logger.info(f"Unfroze last {unfrozen_count} layer blocks of ResNet-50")


class InceptionV3Classifier(nn.Module):
    """
    Inception-v3 model fine-tuned for equipment classification.
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super(InceptionV3Classifier, self).__init__()
        
        # Load pre-trained Inception-v3
        if pretrained:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            self.model = models.inception_v3(weights=weights)
            logger.info("Loaded Inception-v3 with ImageNet pre-trained weights")
        else:
            self.model = models.inception_v3(weights=None)
            logger.info("Initialized Inception-v3 without pre-trained weights")
        
        # Set aux_logits to False for simpler training
        self.model.aux_logits = False
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Froze Inception-v3 backbone layers")
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"Modified Inception-v3 classifier head for {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def unfreeze_backbone(self, num_blocks: Optional[int] = None):
        """
        Unfreeze the backbone layers for fine-tuning.
        
        Args:
            num_blocks: Number of inception blocks to unfreeze from the end. If None, unfreezes all.
        """
        if num_blocks is None:
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Unfroze all Inception-v3 layers")
        else:
            # Unfreeze specific blocks
            blocks = [
                self.model.Mixed_7c,
                self.model.Mixed_7b,
                self.model.Mixed_7a,
                self.model.Mixed_6e,
                self.model.Mixed_6d,
                self.model.Mixed_6c,
                self.model.Mixed_6b,
                self.model.Mixed_6a
            ]
            unfrozen_count = 0
            for block in blocks[:num_blocks]:
                for param in block.parameters():
                    param.requires_grad = True
                unfrozen_count += 1
            logger.info(f"Unfroze last {unfrozen_count} inception blocks")


def get_model(
    model_type: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to get a model by type.
    
    Args:
        model_type: 'resnet50' or 'inception_v3'
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pre-trained weights
        freeze_backbone: Whether to freeze the backbone layers
    
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'resnet50' or model_type == 'resnet':
        return ResNet50Classifier(num_classes, pretrained, freeze_backbone)
    elif model_type == 'inception_v3' or model_type == 'inception':
        return InceptionV3Classifier(num_classes, pretrained, freeze_backbone)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'resnet50' or 'inception_v3'")


def count_parameters(model: nn.Module) -> tuple:
    """
    Count total and trainable parameters in a model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
