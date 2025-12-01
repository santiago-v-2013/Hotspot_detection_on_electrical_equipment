"""
Dataset and data loading utilities for hotspot classification.

This module provides dataset classes and data loaders for binary hotspot
classification (hotspot vs no_hotspot).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class HotspotDataset(Dataset):
    """
    Dataset for binary hotspot classification.
    
    Expected directory structure:
    data_dir/
        Equipment_Type_1/
            hotspot/
                image1.jpg
                image2.jpg
            no_hotspot/
                image3.jpg
                image4.jpg
        Equipment_Type_2/
            ...
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        equipment_types: Optional[List[str]] = None
    ):
        """
        Initialize the hotspot dataset.
        
        Args:
            data_dir: Root directory containing equipment subdirectories
            transform: Torchvision transforms to apply to images
            equipment_types: List of equipment types to include (None = all)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['no_hotspot', 'hotspot']  # 0: no_hotspot, 1: hotspot
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all images
        self.samples = []
        self._load_samples(equipment_types)
        
        logger.info(f"Loaded {len(self.samples)} images")
        logger.info(f"Classes: {self.classes}")
        
    def _load_samples(self, equipment_types: Optional[List[str]] = None):
        """Load all image paths and labels."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Get equipment directories
        equipment_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if equipment_types:
            equipment_dirs = [d for d in equipment_dirs if d.name in equipment_types]
        
        if not equipment_dirs:
            raise ValueError(f"No equipment directories found in {self.data_dir}")
        
        # Load images from each equipment type
        for equipment_dir in equipment_dirs:
            for class_name in self.classes:
                class_dir = equipment_dir / class_name
                
                if not class_dir.exists():
                    continue
                
                # Get all image files
                image_files = (
                    list(class_dir.glob('*.jpg')) +
                    list(class_dir.glob('*.png')) +
                    list(class_dir.glob('*.jpeg'))
                )
                
                # Add to samples
                for img_path in image_files:
                    self.samples.append((
                        str(img_path),
                        self.class_to_idx[class_name],
                        equipment_dir.name
                    ))
        
        if not self.samples:
            raise ValueError(f"No images found in {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label, equipment = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {cls: 0 for cls in self.classes}
        for _, label, _ in self.samples:
            distribution[self.classes[label]] += 1
        return distribution
    
    def get_equipment_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get the distribution of classes per equipment type."""
        equipment_dist = {}
        for _, label, equipment in self.samples:
            if equipment not in equipment_dist:
                equipment_dist[equipment] = {cls: 0 for cls in self.classes}
            equipment_dist[equipment][self.classes[label]] += 1
        return equipment_dist


def get_transforms(
    image_size: int = 224,
    augment: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and validation transforms.
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation for training
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Normalization values for ImageNet pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    augment: bool = True,
    equipment_types: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root directory containing the dataset
        batch_size: Batch size for data loaders
        image_size: Target image size
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        augment: Whether to apply data augmentation
        equipment_types: List of equipment types to include (None = all)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict)
    """
    # Set random seed
    torch.manual_seed(seed)
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size, augment)
    
    # Load full dataset
    full_dataset = HotspotDataset(
        data_dir=data_dir,
        transform=None,  # Will apply transforms after split
        equipment_types=equipment_types
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
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
    
    # Get dataset info
    class_distribution = full_dataset.get_class_distribution()
    equipment_distribution = full_dataset.get_equipment_distribution()
    
    info = {
        'num_classes': len(full_dataset.classes),
        'classes': full_dataset.classes,
        'total_samples': total_size,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'class_distribution': class_distribution,
        'equipment_distribution': equipment_distribution
    }
    
    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    logger.info(f"Class distribution: {class_distribution}")
    
    return train_loader, val_loader, test_loader, info
