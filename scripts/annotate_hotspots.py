#!/usr/bin/env python3
"""
Automatic hotspot annotation script for thermal images.

This script analyzes thermal images and automatically classifies them as:
- hotspot: if the image contains a significant red/hot region
- no_hotspot: if the image does not contain significant hot regions

The detection is based on:
1. HSV color space analysis to detect red/orange/yellow regions
2. Region size analysis to ensure the hot area is significant
3. Intensity analysis to confirm it's a genuine hotspot
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import shutil
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HotspotAnnotator:
    """Automatic hotspot detection and annotation for thermal images."""
    
    def __init__(
        self,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.95,
        intensity_threshold: int = 150,
        saturation_threshold: int = 50
    ):
        """
        Initialize the hotspot annotator.
        
        Args:
            min_area_ratio: Minimum ratio of hot pixels to total image area (default: 1%)
            max_area_ratio: Maximum ratio to avoid full-image false positives (default: 95%)
            intensity_threshold: Minimum brightness value for hot regions (0-255)
            saturation_threshold: Minimum saturation for color detection (0-255)
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.intensity_threshold = intensity_threshold
        self.saturation_threshold = saturation_threshold
        
    def detect_hotspot(self, image_path: Path, debug: bool = False) -> dict:
        """
        Detect if an image contains a hotspot.
        
        Args:
            image_path: Path to the thermal image
            debug: If True, save debug images showing detection process
            
        Returns:
            Dictionary with detection results:
            {
                'has_hotspot': bool,
                'hot_area_ratio': float,
                'max_intensity': int,
                'num_regions': int,
                'confidence': float
            }
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = img.shape[:2]
        total_pixels = height * width
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for hot regions (red and orange only)
        # Red wraps around in HSV, so we need two ranges
        # Lower red (0-10°)
        lower_red1 = np.array([0, self.saturation_threshold, self.intensity_threshold])
        upper_red1 = np.array([10, 255, 255])
        
        # Upper red (170-180°)
        lower_red2 = np.array([170, self.saturation_threshold, self.intensity_threshold])
        upper_red2 = np.array([180, 255, 255])
        
        # Orange (10-25°)
        lower_orange = np.array([10, self.saturation_threshold, self.intensity_threshold])
        upper_orange = np.array([25, 255, 255])
        
        # Create masks for each color range
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Combine all masks (red + orange)
        hot_mask = cv2.bitwise_or(mask_red1, mask_red2)
        hot_mask = cv2.bitwise_or(hot_mask, mask_orange)
        
        # Apply morphological operations to remove noise and connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hot_mask, connectivity=8)
        
        # Analyze regions (skip background label 0)
        significant_regions = 0
        total_hot_area = 0
        max_region_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            area_ratio = area / total_pixels
            
            # Check if region is significant
            if area_ratio >= self.min_area_ratio:
                significant_regions += 1
                total_hot_area += area
                max_region_area = max(max_region_area, area)
        
        # Calculate metrics
        hot_area_ratio = total_hot_area / total_pixels
        max_intensity = np.max(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        
        # Determine if there's a hotspot
        has_hotspot = (
            significant_regions > 0 and
            self.min_area_ratio <= hot_area_ratio <= self.max_area_ratio and
            max_intensity >= self.intensity_threshold
        )
        
        # Calculate confidence score (0-1)
        confidence = min(1.0, hot_area_ratio / 0.3) if has_hotspot else 1.0 - min(1.0, hot_area_ratio / 0.02)
        
        # Debug visualization
        if debug:
            debug_dir = image_path.parent / 'debug'
            debug_dir.mkdir(exist_ok=True)
            
            # Save mask
            cv2.imwrite(str(debug_dir / f"{image_path.stem}_mask.jpg"), hot_mask)
            
            # Create visualization
            vis = img.copy()
            vis[hot_mask > 0] = [0, 0, 255]  # Highlight hot regions in red
            cv2.imwrite(str(debug_dir / f"{image_path.stem}_visualization.jpg"), vis)
        
        return {
            'has_hotspot': bool(has_hotspot),
            'hot_area_ratio': float(hot_area_ratio),
            'max_intensity': int(max_intensity),
            'num_regions': int(significant_regions),
            'confidence': float(confidence)
        }
    
    def annotate_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        save_annotations: bool = True,
        debug_samples: int = 0
    ) -> dict:
        """
        Annotate all images in a dataset.
        
        Args:
            input_dir: Directory containing thermal images organized by equipment type
            output_dir: Directory to save classified images
            save_annotations: If True, save JSON file with all annotations
            debug_samples: Number of random samples to save with debug visualizations
            
        Returns:
            Dictionary with annotation statistics
        """
        stats = {
            'total_images': 0,
            'hotspot_count': 0,
            'no_hotspot_count': 0,
            'by_equipment': {},
            'annotations': []
        }
        
        # Find all equipment subdirectories
        equipment_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if not equipment_dirs:
            logger.warning(f"No equipment subdirectories found in {input_dir}")
            return stats
        
        logger.info(f"Found {len(equipment_dirs)} equipment types")
        
        # Process each equipment type
        for equipment_dir in equipment_dirs:
            equipment_name = equipment_dir.name
            logger.info(f"Processing: {equipment_name}")
            
            # Get all images
            image_files = list(equipment_dir.glob('*.jpg')) + \
                         list(equipment_dir.glob('*.png')) + \
                         list(equipment_dir.glob('*.jpeg'))
            
            if not image_files:
                logger.warning(f"No images found in {equipment_dir}")
                continue
            
            equipment_stats = {'hotspot': 0, 'no_hotspot': 0}
            
            # Select random samples for debug
            debug_indices = set(np.random.choice(len(image_files), min(debug_samples, len(image_files)), replace=False))
            
            # Process each image
            for idx, image_path in enumerate(tqdm(image_files, desc=equipment_name)):
                try:
                    # Detect hotspot
                    is_debug = idx in debug_indices
                    result = self.detect_hotspot(image_path, debug=is_debug)
                    
                    # Determine class
                    class_name = 'hotspot' if result['has_hotspot'] else 'no_hotspot'
                    equipment_stats[class_name] += 1
                    stats['total_images'] += 1
                    
                    # Create output directory structure
                    output_class_dir = output_dir / equipment_name / class_name
                    output_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy image to classified directory
                    shutil.copy2(image_path, output_class_dir / image_path.name)
                    
                    # Save annotation
                    annotation = {
                        'image_path': str(image_path.relative_to(input_dir)),
                        'equipment_type': equipment_name,
                        'class': class_name,
                        'metrics': result
                    }
                    stats['annotations'].append(annotation)
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
            
            # Update statistics
            stats['by_equipment'][equipment_name] = equipment_stats
            stats['hotspot_count'] += equipment_stats['hotspot']
            stats['no_hotspot_count'] += equipment_stats['no_hotspot']
            
            logger.info(f"{equipment_name}: {equipment_stats['hotspot']} hotspot, "
                       f"{equipment_stats['no_hotspot']} no hotspot")
        
        # Save annotations to JSON
        if save_annotations and stats['annotations']:
            annotations_file = output_dir / 'annotations.json'
            with open(annotations_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Annotations saved to {annotations_file}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Automatic hotspot annotation for thermal images"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/raw'),
        help='Input directory with raw thermal images organized by equipment type'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed/hotspot_classification'),
        help='Output directory for classified images'
    )
    parser.add_argument(
        '--min-area-ratio',
        type=float,
        default=0.02,
        help='Minimum hot area ratio to consider as hotspot (default: 0.02 = 2%%)'
    )
    parser.add_argument(
        '--max-area-ratio',
        type=float,
        default=0.95,
        help='Maximum hot area ratio to avoid false positives (default: 0.95 = 95%%)'
    )
    parser.add_argument(
        '--intensity-threshold',
        type=int,
        default=150,
        help='Minimum brightness for hot regions (0-255, default: 150)'
    )
    parser.add_argument(
        '--saturation-threshold',
        type=int,
        default=50,
        help='Minimum color saturation for detection (0-255, default: 50)'
    )
    parser.add_argument(
        '--debug-samples',
        type=int,
        default=0,
        help='Number of random samples to save with debug visualizations'
    )
    parser.add_argument(
        '--no-save-annotations',
        action='store_true',
        help='Do not save annotations to JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize annotator
    annotator = HotspotAnnotator(
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        intensity_threshold=args.intensity_threshold,
        saturation_threshold=args.saturation_threshold
    )
    
    logger.info("Starting automatic hotspot annotation")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Parameters:")
    logger.info(f"  - Min area ratio: {args.min_area_ratio * 100:.1f}%")
    logger.info(f"  - Max area ratio: {args.max_area_ratio * 100:.1f}%")
    logger.info(f"  - Intensity threshold: {args.intensity_threshold}")
    logger.info(f"  - Saturation threshold: {args.saturation_threshold}")
    logger.info("")
    
    # Annotate dataset
    stats = annotator.annotate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        save_annotations=not args.no_save_annotations,
        debug_samples=args.debug_samples
    )
    
    # Print final statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANNOTATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total images processed: {stats['total_images']}")
    logger.info(f"Images with hotspot: {stats['hotspot_count']} "
               f"({stats['hotspot_count']/stats['total_images']*100:.1f}%)")
    logger.info(f"Images without hotspot: {stats['no_hotspot_count']} "
               f"({stats['no_hotspot_count']/stats['total_images']*100:.1f}%)")
    logger.info("")
    logger.info("By equipment type:")
    for equipment, counts in stats['by_equipment'].items():
        total = counts['hotspot'] + counts['no_hotspot']
        logger.info(f"  {equipment}:")
        logger.info(f"    - Hotspot: {counts['hotspot']} ({counts['hotspot']/total*100:.1f}%)")
        logger.info(f"    - No hotspot: {counts['no_hotspot']} ({counts['no_hotspot']/total*100:.1f}%)")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
