"""
Preprocessor for equipment type classification task.

This preprocessor applies simple transformations suitable for classifying
the type of electrical equipment (Circuit Breaker, Disconnector, etc.).
"""

import logging
import cv2
import numpy as np
from .base import BasePreprocessor

logger = logging.getLogger(__name__)


class EquipmentClassifierPreprocessor(BasePreprocessor):
    """
    Preprocessor for equipment classification task.
    
    Applies:
    - Grayscale conversion (thermal images are single-channel)
    - Histogram equalization for better contrast
    - Resizing to target size
    - Optional normalization
    """
    
    def __init__(self, input_dir: str, output_dir: str, 
                 target_size: tuple = (224, 224),
                 apply_equalization: bool = True,
                 normalize: bool = False):
        """
        Initialize equipment classifier preprocessor.
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory to save processed images
            target_size: Target image size as (width, height)
            apply_equalization: Whether to apply histogram equalization
            normalize: Whether to normalize to [0, 1] range
        """
        super().__init__(input_dir, output_dir, target_size)
        self.apply_equalization = apply_equalization
        self.normalize = normalize
        logger.info(f"Equipment classifier preprocessor initialized (equalization={apply_equalization})")
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image for equipment classification.
        
        Args:
            image: Input BGR image
            
        Returns:
            Processed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply histogram equalization for better contrast
        if self.apply_equalization:
            gray = cv2.equalizeHist(gray)
        
        # Resize to target size
        resized = self.resize_image(gray)
        
        # Normalize if requested
        if self.normalize:
            resized = self.normalize_image(resized)
        
        return resized
