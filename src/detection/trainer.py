"""
Training module for object detection models.
"""

from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class DetectionTrainer:
    """
    Unified trainer interface for different detection models.
    """
    
    @staticmethod
    def train_yolo(
        model,
        data_yaml: Path,
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        lr0: float = 0.01,
        save_dir: Path = Path('models/hotspot_detection/yolo'),
        **kwargs
    ) -> Dict:
        """
        Train YOLO model (YOLOv8, YOLOv11, etc.).
        
        Args:
            model: YOLOv11Detector instance
            data_yaml: Path to YOLO data configuration
            epochs: Number of epochs
            batch_size: Batch size
            img_size: Image size
            lr0: Initial learning rate
            save_dir: Save directory
            **kwargs: Additional YOLO training arguments
            
        Returns:
            Training results
        """
        return model.train(
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr0=lr0,
            save_dir=save_dir,
            **kwargs
        )
