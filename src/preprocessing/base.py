"""
Base preprocessor class for thermal image processing.
"""

import logging
from pathlib import Path
from abc import ABC, abstractmethod
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Abstract base class for image preprocessing.
    
    All preprocessors should inherit from this class and implement
    the process_image method.
    """
    
    def __init__(self, input_dir: str, output_dir: str, target_size: tuple = (224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory to save processed images
            target_size: Target image size as (width, height)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
    
    @abstractmethod
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed image as numpy array
        """
        pass
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def save_image(self, image: np.ndarray, output_path: Path):
        """
        Save processed image to disk.
        
        Args:
            image: Image to save
            output_path: Path where to save the image
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        cv2.imwrite(str(output_path), image)
        logger.debug(f"Saved image to {output_path}")
    
    def process_directory(self, preserve_structure: bool = True):
        """
        Process all images in the input directory.
        
        Args:
            preserve_structure: If True, maintains subdirectory structure
        """
        logger.info(f"Starting preprocessing from {self.input_dir} to {self.output_dir}")
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            if preserve_structure:
                image_files.extend(self.input_dir.rglob(ext))
            else:
                image_files.extend(self.input_dir.glob(ext))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        processed_count = 0
        error_count = 0
        
        for img_path in image_files:
            try:
                # Read image
                image = cv2.imread(str(img_path))
                
                if image is None:
                    logger.warning(f"Could not read image: {img_path}")
                    error_count += 1
                    continue
                
                # Process image
                processed_image = self.process_image(image)
                
                # Determine output path
                if preserve_structure:
                    relative_path = img_path.relative_to(self.input_dir)
                    output_path = self.output_dir / relative_path
                else:
                    output_path = self.output_dir / img_path.name
                
                # Save processed image
                self.save_image(processed_image, output_path)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(image_files)} images")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                error_count += 1
        
        logger.info(f"Preprocessing complete. Processed: {processed_count}, Errors: {error_count}")
