"""
Preprocessor for hotspot detection with bounding boxes task.

This preprocessor applies advanced transformations optimized for
object detection models that need to localize hotspots precisely.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from .base import BasePreprocessor

logger = logging.getLogger(__name__)


class HotspotDetectorPreprocessor(BasePreprocessor):
    """
    Preprocessor for hotspot object detection.
    
    Applies sophisticated color and spatial transformations:
    - Multi-scale color representation
    - Edge enhancement for precise localization
    - Thermal gradient computation
    - Preserves spatial information for bounding box prediction
    """
    
    def __init__(self, input_dir: str, output_dir: str,
                 target_size: tuple = (416, 416),
                 use_edge_enhancement: bool = True,
                 use_thermal_gradient: bool = True,
                 preserve_aspect_ratio: bool = True):
        """
        Initialize hotspot detector preprocessor.
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory to save processed images
            target_size: Target image size (typically larger for detection)
            use_edge_enhancement: Whether to add edge information
            use_thermal_gradient: Whether to compute thermal gradients
            preserve_aspect_ratio: Whether to preserve aspect ratio (pad if needed)
        """
        super().__init__(input_dir, output_dir, target_size)
        self.use_edge_enhancement = use_edge_enhancement
        self.use_thermal_gradient = use_thermal_gradient
        self.preserve_aspect_ratio = preserve_aspect_ratio
        logger.info(f"Hotspot detector preprocessor initialized (size={target_size})")
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image, optionally preserving aspect ratio with padding.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        if not self.preserve_aspect_ratio:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding to preserve aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
        else:
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image for hotspot detection.
        
        Creates a multi-channel representation with:
        - RGB/HSV color information
        - Edge information (optional)
        - Thermal gradient (optional)
        
        Args:
            image: Input BGR image
            
        Returns:
            Processed multi-channel image
        """
        # Resize with aspect ratio preservation
        resized = self.resize_image(image)
        
        # Convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Start with RGB as base
        channels = [rgb]
        
        # Add edge enhancement if requested
        if self.use_edge_enhancement:
            edges = self._compute_edges(resized)
            channels.append(edges)
        
        # Add thermal gradient if requested
        if self.use_thermal_gradient:
            gradient = self._compute_thermal_gradient(resized)
            channels.append(gradient)
        
        # Stack all channels
        if len(channels) > 1:
            # Ensure all channels have same shape
            processed = np.concatenate(channels, axis=-1)
        else:
            processed = rgb
        
        return processed
    
    def _compute_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Compute edge information using Canny edge detection.
        
        Args:
            image: Input image
            
        Returns:
            Edge map with same spatial dimensions, expanded to 3 channels
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        
        # Expand to 3 channels for concatenation
        edges_3ch = np.stack([edges] * 3, axis=-1)
        
        return edges_3ch
    
    def _compute_thermal_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Compute thermal gradient magnitude.
        
        Thermal gradients help identify boundaries of hot regions.
        
        Args:
            image: Input image
            
        Returns:
            Gradient magnitude with same spatial dimensions, expanded to 3 channels
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        
        # Expand to 3 channels for concatenation
        gradient_3ch = np.stack([gradient_magnitude] * 3, axis=-1)
        
        return gradient_3ch
    
    def save_image(self, image: np.ndarray, output_path: Path):
        """
        Save processed image, handling multi-channel cases.
        
        For images with more than 3 channels, saves only the first 3 channels.
        
        Args:
            image: Image to save
            output_path: Path where to save the image
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If more than 3 channels, save only RGB channels
        if len(image.shape) == 3 and image.shape[2] > 3:
            image_to_save = image[:, :, :3]
            logger.debug(f"Saving only first 3 channels of {image.shape[2]}-channel image")
        else:
            image_to_save = image
        
        # Convert to uint8 if needed
        if image_to_save.dtype == np.float32 or image_to_save.dtype == np.float64:
            image_to_save = (image_to_save * 255).astype(np.uint8)
        
        # Convert RGB back to BGR for OpenCV
        if len(image_to_save.shape) == 3:
            image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(output_path), image_to_save)
        logger.debug(f"Saved image to {output_path}")
