#!/usr/bin/env python3
"""
Script to preprocess images for hotspot object detection with bounding boxes.

This script applies advanced preprocessing including edge detection and
thermal gradient computation to help detect and localize hotspots precisely.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import HotspotDetectorPreprocessor, setup_logging


def main():
    """Parse arguments and run hotspot detector preprocessing."""
    # Configure logging system
    setup_logging(log_dir=str(project_root / 'logs'))
    
    # Get logger for this script
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="Preprocess thermal images for hotspot object detection."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw images (default: data/raw)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/hotspot_detection",
        help="Directory to save processed images (default: data/processed/hotspot_detection)"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[416, 416],
        metavar=('WIDTH', 'HEIGHT'),
        help="Target image size as width height (default: 416 416)"
    )
    parser.add_argument(
        "--no_edge_enhancement",
        action="store_true",
        help="Disable edge enhancement"
    )
    parser.add_argument(
        "--no_thermal_gradient",
        action="store_true",
        help="Disable thermal gradient computation"
    )
    parser.add_argument(
        "--no_preserve_aspect",
        action="store_true",
        help="Don't preserve aspect ratio (will stretch images)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Hotspot Object Detection Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target size: {args.target_size[0]}x{args.target_size[1]}")
    logger.info(f"Edge enhancement: {not args.no_edge_enhancement}")
    logger.info(f"Thermal gradient: {not args.no_thermal_gradient}")
    logger.info(f"Preserve aspect ratio: {not args.no_preserve_aspect}")
    logger.info("=" * 60)
    
    try:
        preprocessor = HotspotDetectorPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_size=tuple(args.target_size),
            use_edge_enhancement=not args.no_edge_enhancement,
            use_thermal_gradient=not args.no_thermal_gradient,
            preserve_aspect_ratio=not args.no_preserve_aspect
        )
        
        preprocessor.process_directory(preserve_structure=True)
        
        logger.info("✓ Hotspot detection preprocessing completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
