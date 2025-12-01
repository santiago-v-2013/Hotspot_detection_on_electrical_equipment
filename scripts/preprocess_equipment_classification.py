#!/usr/bin/env python3
"""
Script to preprocess images for equipment type classification.

This script applies simple preprocessing (grayscale, histogram equalization)
suitable for classifying electrical equipment types.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import EquipmentClassifierPreprocessor, setup_logging


def main():
    """Parse arguments and run equipment classifier preprocessing."""
    # Configure logging system
    setup_logging(log_dir=str(project_root / 'logs'))
    
    # Get logger for this script
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="Preprocess thermal images for equipment type classification."
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
        default="data/processed/equipment_classification",
        help="Directory to save processed images (default: data/processed/equipment_classification)"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=('WIDTH', 'HEIGHT'),
        help="Target image size as width height (default: 224 224)"
    )
    parser.add_argument(
        "--no_equalization",
        action="store_true",
        help="Disable histogram equalization"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize images to [0, 1] range"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Equipment Type Classification Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target size: {args.target_size[0]}x{args.target_size[1]}")
    logger.info(f"Histogram equalization: {not args.no_equalization}")
    logger.info(f"Normalization: {args.normalize}")
    logger.info("=" * 60)
    
    try:
        preprocessor = EquipmentClassifierPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_size=tuple(args.target_size),
            apply_equalization=not args.no_equalization,
            normalize=args.normalize
        )
        
        preprocessor.process_directory(preserve_structure=True)
        
        logger.info("✓ Equipment classification preprocessing completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
