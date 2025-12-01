"""
Preprocessor for binary hotspot classification task.

This preprocessor applies more sophisticated color-based transformations
to help identify the presence of hotspots in thermal images.
"""

import logging
import cv2
import numpy as np
from .base import BasePreprocessor

logger = logging.getLogger(__name__)


class HotspotClassifierPreprocessor(BasePreprocessor):
    """
    Preprocessor for binary hotspot classification.
    
    Applies color-based transformations to enhance hot regions:
    - Color space conversion (HSV for thermal analysis)
    - Temperature-based color enhancement
    - Multi-channel output preserving thermal information
    """
    
    def __init__(self, input_dir: str, output_dir: str,
                 target_size: tuple = (224, 224),
                 color_mode: str = 'hsv',
                 enhance_hot_regions: bool = True):
        """
        Initialize hotspot classifier preprocessor.
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory to save processed images
            target_size: Target image size as (width, height)
            color_mode: Color space to use ('hsv', 'lab', 'ycrcb', or 'rgb')
            enhance_hot_regions: Whether to enhance hot regions
        """
        super().__init__(input_dir, output_dir, target_size)
        self.color_mode = color_mode.lower()
        self.enhance_hot_regions = enhance_hot_regions
        
        if self.color_mode not in ['hsv', 'lab', 'ycrcb', 'rgb']:
            raise ValueError(f"Invalid color_mode: {color_mode}")
        
        logger.info(f"Hotspot classifier preprocessor initialized (color_mode={color_mode})")
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image for hotspot classification.
        
        Args:
            image: Input BGR image
            
        Returns:
            Processed multi-channel image with enhanced thermal information
        """
        # Resize first to reduce computation
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to desired color space
        if self.color_mode == 'hsv':
            converted = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        elif self.color_mode == 'lab':
            converted = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        elif self.color_mode == 'ycrcb':
            converted = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
        else:  # rgb
            converted = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Enhance hot regions if requested
        if self.enhance_hot_regions:
            converted = self._enhance_hot_regions(resized, converted)
        
        return converted
    
    def _enhance_hot_regions(self, original: np.ndarray, converted: np.ndarray) -> np.ndarray:
        """
        Enhance hot regions in thermal images.
        
        Hot regions typically appear as white/yellow/red in thermal images.
        
        Args:
            original: Original BGR image
            converted: Converted color space image
            
        Returns:
            Enhanced image
        """
        # Create a grayscale version for intensity analysis
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast, making hot spots more visible
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if self.color_mode == 'hsv':
            # Enhance V (value) channel
            converted[:, :, 2] = clahe.apply(converted[:, :, 2])
        elif self.color_mode == 'lab':
            # Enhance L (lightness) channel
            converted[:, :, 0] = clahe.apply(converted[:, :, 0])
        elif self.color_mode == 'ycrcb':
            # Enhance Y (luma) channel
            converted[:, :, 0] = clahe.apply(converted[:, :, 0])
        else:  # rgb
            # Apply to each channel
            for i in range(3):
                converted[:, :, i] = clahe.apply(converted[:, :, i])
        
        return converted
