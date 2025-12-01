#!/usr/bin/env python3
"""
Automatic hotspot detection annotation script for thermal images.

This script analyzes thermal images and automatically generates bounding box
annotations for hotspot regions. The annotations are saved in YOLO format
alongside the images, organized by equipment type.

The detection is based on:
1. HSV color space analysis to detect red/orange hot regions
2. Connected component analysis to find individual hotspots
3. Bounding box generation for each detected hotspot
4. Filtering based on size and intensity thresholds
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
import shutil

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HotspotDetectionAnnotator:
    """Automatic hotspot detection with bounding box annotation."""
    
    def __init__(
        self,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.5,
        min_intensity: int = 150,
        min_saturation: int = 50,
        iou_threshold: float = 0.3,
        merge_close_boxes: bool = True
    ):
        """
        Initialize the hotspot detection annotator.
        
        Args:
            min_area_ratio: Minimum ratio of hotspot to image area (default: 0.5%)
            max_area_ratio: Maximum ratio to avoid full-image detections (default: 50%)
            min_intensity: Minimum brightness value for hot regions (0-255)
            min_saturation: Minimum saturation for color detection (0-255)
            iou_threshold: IoU threshold for merging overlapping boxes
            merge_close_boxes: Whether to merge nearby bounding boxes
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_intensity = min_intensity
        self.min_saturation = min_saturation
        self.iou_threshold = iou_threshold
        self.merge_close_boxes = merge_close_boxes
        
    def detect_hotspots(
        self, 
        image_path: Path, 
        debug: bool = False
    ) -> List[Dict]:
        """
        Detect hotspots in an image and return bounding boxes.
        
        Args:
            image_path: Path to the thermal image
            debug: If True, save debug images showing detection process
            
        Returns:
            List of dictionaries with bounding box info:
            [{
                'bbox': [x, y, w, h],  # pixel coordinates
                'bbox_normalized': [x_center, y_center, width, height],  # YOLO format
                'confidence': float,
                'area_ratio': float,
                'max_intensity': int
            }]
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = img.shape[:2]
        total_pixels = height * width
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for hot regions (red and orange)
        # Red wraps around in HSV, so we need two ranges
        # Lower red (0-10°)
        lower_red1 = np.array([0, self.min_saturation, self.min_intensity])
        upper_red1 = np.array([10, 255, 255])
        
        # Upper red (170-180°)
        lower_red2 = np.array([170, self.min_saturation, self.min_intensity])
        upper_red2 = np.array([180, 255, 255])
        
        # Orange (10-25°)
        lower_orange = np.array([10, self.min_saturation, self.min_intensity])
        upper_orange = np.array([25, 255, 255])
        
        # Create masks for each color range
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Combine all masks
        hot_mask = cv2.bitwise_or(mask_red1, mask_red2)
        hot_mask = cv2.bitwise_or(hot_mask, mask_orange)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours (individual hotspot regions)
        contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bounding boxes for each hotspot region
        detections = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
            area_ratio = bbox_area / total_pixels
            
            # Filter by size
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue
            
            # Get region statistics
            region_mask = np.zeros_like(hot_mask)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            region_pixels = cv2.bitwise_and(img, img, mask=region_mask)
            region_v = cv2.bitwise_and(hsv[:,:,2], hsv[:,:,2], mask=region_mask)
            max_intensity = region_v.max()
            
            # Calculate confidence based on area and intensity
            confidence = (area_ratio * 100 + max_intensity / 255) / 2
            
            # Normalize bbox to YOLO format [x_center, y_center, width, height] in range [0, 1]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            
            detections.append({
                'bbox': [x, y, w, h],
                'bbox_normalized': [x_center, y_center, w_norm, h_norm],
                'confidence': confidence,
                'area_ratio': area_ratio,
                'max_intensity': int(max_intensity)
            })
        
        # Merge overlapping boxes if enabled
        if self.merge_close_boxes and len(detections) > 1:
            detections = self._merge_overlapping_boxes(detections)
        
        return detections
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_overlapping_boxes(self, detections: List[Dict]) -> List[Dict]:
        """Merge overlapping bounding boxes based on IoU threshold."""
        if not detections:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = [False] * len(detections)
        
        for i in range(len(detections)):
            if used[i]:
                continue
            
            current = detections[i]
            boxes_to_merge = [current]
            used[i] = True
            
            # Find overlapping boxes
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                
                iou = self._calculate_iou(current['bbox'], detections[j]['bbox'])
                if iou > self.iou_threshold:
                    boxes_to_merge.append(detections[j])
                    used[j] = True
            
            # Merge boxes (take the one with highest confidence)
            if len(boxes_to_merge) == 1:
                merged.append(current)
            else:
                # Merge by averaging (weighted by confidence)
                merged.append(boxes_to_merge[0])  # Keep highest confidence box
        
        return merged


def annotate_dataset(
    data_dir: Path,
    output_dir: Path,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.5,
    min_intensity: int = 150,
    debug: bool = False,
    save_format: str = 'yolo'
) -> dict:
    """
    Annotate entire dataset with hotspot detections.
    
    Organizes output by equipment type, copying images and creating YOLO annotation files.
    
    Args:
        data_dir: Directory containing thermal images (organized by equipment)
        output_dir: Directory to save annotated images and labels
        min_area_ratio: Minimum hotspot area ratio
        max_area_ratio: Maximum hotspot area ratio
        min_intensity: Minimum intensity threshold
        debug: Save debug visualizations
        save_format: Annotation format ('yolo')
    """
    annotator = HotspotDetectionAnnotator(
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_intensity=min_intensity
    )
    
    # Find equipment directories
    equipment_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    # Statistics and annotations
    stats = {
        'total_images': 0,
        'images_with_hotspots': 0,
        'total_hotspots': 0,
        'avg_hotspots_per_image': 0,
        'by_equipment': {},
        'annotations': []
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each equipment type
    for equipment_dir in tqdm(equipment_dirs, desc="Processing equipment"):
        equipment_name = equipment_dir.name
        
        # Create equipment output directory
        equipment_output = output_dir / equipment_name
        equipment_output.mkdir(exist_ok=True)
        
        # Find images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(equipment_dir.glob(ext))
        
        if not image_files:
            continue
        
        equipment_stats = {
            'total': len(image_files),
            'with_hotspots': 0,
            'total_hotspots': 0,
            'images': []
        }
        
        # Process images
        for img_path in tqdm(image_files, desc=f"  {equipment_name}", leave=False):
            try:
                # Detect hotspots
                detections = annotator.detect_hotspots(img_path, debug=False)
                
                num_hotspots = len(detections)
                has_hotspot = num_hotspots > 0
                
                # Copy image to output
                output_img_path = equipment_output / img_path.name
                shutil.copy(img_path, output_img_path)
                
                # Save YOLO annotation if has detections
                if has_hotspot and save_format == 'yolo':
                    label_file = equipment_output / (img_path.stem + '.txt')
                    with open(label_file, 'w') as f:
                        for det in detections:
                            bbox = det['bbox_normalized']
                            # YOLO format: class_id x_center y_center width height
                            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                # Update statistics
                stats['total_images'] += 1
                if has_hotspot:
                    stats['images_with_hotspots'] += 1
                    stats['total_hotspots'] += num_hotspots
                    equipment_stats['with_hotspots'] += 1
                    equipment_stats['total_hotspots'] += num_hotspots
                
                # Store annotation info
                annotation = {
                    'image_path': str(img_path.relative_to(data_dir)),
                    'equipment': equipment_name,
                    'num_hotspots': num_hotspots,
                    'detections': []
                }
                
                for det in detections:
                    annotation['detections'].append({
                        'bbox': det['bbox'],
                        'bbox_normalized': det['bbox_normalized'],
                        'confidence': det['confidence'],
                        'area_ratio': det['area_ratio']
                    })
                
                stats['annotations'].append(annotation)
                equipment_stats['images'].append({
                    'filename': img_path.name,
                    'num_hotspots': num_hotspots
                })
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        stats['by_equipment'][equipment_name] = equipment_stats
    
    # Calculate average
    if stats['images_with_hotspots'] > 0:
        stats['avg_hotspots_per_image'] = stats['total_hotspots'] / stats['images_with_hotspots']
    
    # Save annotations
    annotations_file = output_dir / 'annotations.json'
    with open(annotations_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Annotation Summary")
    logger.info("="*60)
    logger.info(f"Total images processed: {stats['total_images']}")
    logger.info(f"Images with hotspots: {stats['images_with_hotspots']} ({stats['images_with_hotspots']/stats['total_images']*100:.1f}%)")
    logger.info(f"Total hotspots detected: {stats['total_hotspots']}")
    logger.info(f"Average hotspots per image: {stats['avg_hotspots_per_image']:.2f}")
    logger.info(f"\nBy equipment:")
    for equipment, eq_stats in stats['by_equipment'].items():
        logger.info(f"  {equipment}:")
        logger.info(f"    Total: {eq_stats['total']}")
        logger.info(f"    With hotspots: {eq_stats['with_hotspots']}")
        logger.info(f"    Total hotspots: {eq_stats['total_hotspots']}")
    logger.info(f"\nAnnotations saved to: {annotations_file}")
    
    return stats


def create_visualization(output_dir: Path, stats: dict):
    """Create visualization of detection examples."""
    import matplotlib.pyplot as plt
    import random
    
    # Get images with detections
    images_with_detections = [a for a in stats['annotations'] if a['num_hotspots'] > 0]
    
    if not images_with_detections:
        logger.warning("No images with detections found for visualization")
        return
    
    # Select 6 examples
    n_samples = min(6, len(images_with_detections))
    
    # Sort by number of hotspots
    images_with_detections.sort(key=lambda x: x['num_hotspots'], reverse=True)
    
    samples = []
    if n_samples >= 6:
        # 2 with most hotspots
        samples.extend(images_with_detections[:2])
        # 2 from middle
        mid_start = len(images_with_detections) // 3
        samples.extend(images_with_detections[mid_start:mid_start+2])
        # 2 with fewest
        samples.extend(images_with_detections[-2:])
    else:
        samples = images_with_detections[:n_samples]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, annotation in enumerate(samples):
        equipment = annotation['equipment']
        filename = Path(annotation['image_path']).name
        img_path = output_dir / equipment / filename
        
        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        for det in annotation['detections']:
            x, y, w, h = det['bbox']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = f'Hotspot'
            cv2.putText(image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show
        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_title(f'{equipment}\n{annotation["num_hotspots"]} hotspot{"s" if annotation["num_hotspots"] > 1 else ""}',
                          fontsize=10, fontweight='bold')
    
    # Hide unused axes
    for idx in range(len(samples), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Hotspot Detection Examples - Bounding Boxes', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'annotation_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Automatic hotspot detection annotation for thermal images'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing thermal images (organized by equipment)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/hotspot_detection',
        help='Output directory for annotations'
    )
    parser.add_argument(
        '--min-area-ratio',
        type=float,
        default=0.005,
        help='Minimum hotspot area ratio (default: 0.005 = 0.5%%)'
    )
    parser.add_argument(
        '--max-area-ratio',
        type=float,
        default=0.5,
        help='Maximum hotspot area ratio (default: 0.5 = 50%%)'
    )
    parser.add_argument(
        '--min-intensity',
        type=int,
        default=150,
        help='Minimum intensity threshold (0-255, default: 150)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save debug visualizations'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='yolo',
        choices=['yolo'],
        help='Annotation format (default: yolo)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return
    
    logger.info("Starting hotspot detection annotation")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Min area ratio: {args.min_area_ratio * 100:.1f}%")
    logger.info(f"Max area ratio: {args.max_area_ratio * 100:.1f}%")
    logger.info(f"Min intensity: {args.min_intensity}")
    logger.info("")
    
    # Annotate dataset
    stats = annotate_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_intensity=args.min_intensity,
        debug=args.debug,
        save_format=args.format
    )
    
    # Create visualization
    create_visualization(output_dir, stats)
    
    logger.info("\nHotspot detection annotation completed!")


if __name__ == '__main__':
    main()
