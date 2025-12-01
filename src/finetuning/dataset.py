"""
Dataset and data loading utilities for equipment classification.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class EquipmentDataset(Dataset):
    """
    Dataset for equipment classification.
    Expects folder structure: root_dir/class_name/images.jpg
    """
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Path to the root directory containing class folders
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get class names from folder names
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), class_idx))
            for img_path in class_dir.glob('*.jpeg'):
                self.samples.append((str(img_path), class_idx))
            for img_path in class_dir.glob('*.png'):
                self.samples.append((str(img_path), class_idx))
        
        logger.info(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        for cls_name in self.classes:
            count = sum(1 for _, idx in self.samples if idx == self.class_to_idx[cls_name])
            logger.info(f"  {cls_name}: {count} images")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        Returns weights inversely proportional to class frequencies.
        """
        class_counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Inverse frequency weights
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(self.classes)
        
        return class_weights


def get_transforms(input_size: int = 224, model_type: str = 'resnet') -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms.
    
    Args:
        input_size: Input image size (224 for ResNet, 299 for Inception)
        model_type: 'resnet' or 'inception'
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Normalization values for ImageNet pre-trained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    input_size: int = 224,
    model_type: str = 'resnet',
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to the data directory
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        input_size: Input image size
        model_type: 'resnet' or 'inception'
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Get transforms
    train_transform, val_transform = get_transforms(input_size, model_type)
    
    # Create full dataset with validation transforms to get all samples
    full_dataset = EquipmentDataset(data_dir, transform=None)
    class_names = full_dataset.classes
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = random_split(
        range(total_size),
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create separate datasets with appropriate transforms
    train_dataset = EquipmentDataset(data_dir, transform=train_transform)
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices.indices]
    
    val_dataset = EquipmentDataset(data_dir, transform=val_transform)
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices.indices]
    
    test_dataset = EquipmentDataset(data_dir, transform=val_transform)
    test_dataset.samples = [test_dataset.samples[i] for i in test_indices.indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names
