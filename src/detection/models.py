"""
Object detection models.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class YOLOv11Detector:
    """
    Wrapper for YOLOv11 detection model.
    """
    
    def __init__(
        self,
        model_size: str = 's',
        num_classes: int = 1,
        pretrained: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            model_size: YOLOv11 size ('n', 's', 'm', 'l', 'x')
            num_classes: Number of object classes
            pretrained: Load pre-trained COCO weights
            device: Device to use
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.device = device
        
        # Create models directory for pretrained weights
        weights_dir = Path('models/pretrained')
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if pretrained:
            # Load pre-trained model - specify path to avoid downloading to project root
            model_path = weights_dir / f'yolo11{model_size}.pt'
            
            if not model_path.exists():
                # Download to models/pretrained/ directory
                logger.info(f"Downloading pre-trained YOLOv11{model_size} model to {weights_dir}")
                # YOLO downloads to current dir, so we need to change to weights_dir temporarily
                import os
                original_dir = os.getcwd()
                try:
                    os.chdir(weights_dir)
                    self.model = YOLO(f'yolo11{model_size}.pt')
                finally:
                    os.chdir(original_dir)
            else:
                logger.info(f"Loading pre-trained YOLOv11{model_size} from {model_path}")
                self.model = YOLO(str(model_path))
        else:
            # Load architecture only
            model_name = f'yolo11{model_size}.yaml'
            logger.info(f"Loading YOLOv11{model_size} architecture")
            self.model = YOLO(model_name)
        
        self.model.to(device)
    
    def train(
        self,
        data_yaml: Path,
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        lr0: float = 0.01,
        save_dir: Path = Path('runs/detect'),
        **kwargs
    ) -> Dict:
        """
        Train the YOLO model.
        
        Args:
            data_yaml: Path to YOLO data configuration file
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            lr0: Initial learning rate
            save_dir: Directory to save results
            **kwargs: Additional YOLO training arguments
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting YOLOv11{self.model_size} training")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
        
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=lr0,
            device=self.device,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True,
            **kwargs
        )
        
        return results
    
    def validate(
        self,
        data_yaml: Optional[Path] = None,
        split: str = 'val',
        **kwargs
    ) -> Dict:
        """
        Validate the model.
        
        Args:
            data_yaml: Path to YOLO data configuration
            split: Dataset split to validate on
            **kwargs: Additional validation arguments
            
        Returns:
            Validation metrics
        """
        if data_yaml:
            results = self.model.val(data=str(data_yaml), split=split, **kwargs)
        else:
            results = self.model.val(split=split, **kwargs)
        
        return results
    
    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs
    ) -> List:
        """
        Run inference.
        
        Args:
            source: Image source (path, directory, or URL)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            **kwargs: Additional prediction arguments
            
        Returns:
            List of detection results
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            **kwargs
        )
        
        return results
    
    def export(
        self,
        format: str = 'onnx',
        **kwargs
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported model
        """
        path = self.model.export(format=format, **kwargs)
        logger.info(f"Model exported to: {path}")
        return path
    
    def save(self, path: Path):
        """Save model weights."""
        self.model.save(str(path))
        logger.info(f"Model saved to: {path}")
    
    def load(self, path: Path):
        """Load model weights."""
        self.model = YOLO(str(path))
        self.model.to(self.device)
        logger.info(f"Model loaded from: {path}")


def get_model(
    model_type: str,
    model_size: str = 's',
    num_classes: int = 1,
    pretrained: bool = True,
    device: str = 'cuda'
):
    """
    Factory function to get detection model.
    
    Args:
        model_type: 'yolov8', 'yolov11', or 'yolo11'
        model_size: For YOLO: 'n', 's', 'm', 'l', 'x'
        num_classes: Number of object classes
        pretrained: Use pre-trained weights
        device: Device to use
        
    Returns:
        Detection model instance
    """
    model_type = model_type.lower()
    
    if model_type in ['yolov8', 'yolov11', 'yolo11']:
        return YOLOv11Detector(
            model_size=model_size,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'yolov8', 'yolov11', or 'yolo11'")
