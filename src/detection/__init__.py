"""
Object detection module for thermal hotspot detection.

Supports:
- YOLOv8 (ultralytics)
- YOLOv11 (ultralytics) - Latest version
"""

from .dataset import YOLODataset
from .models import YOLOv11Detector
from .trainer import DetectionTrainer

__all__ = [
    'YOLODataset',
    'YOLOv11Detector',
    'DetectionTrainer'
]
