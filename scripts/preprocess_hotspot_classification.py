#!/usr/bin/env python3
"""
Script to preprocess images for binary hotspot classification.

This script processes already-annotated images (from automatic or manual annotation)
and applies preprocessing to enhance thermal information for model training.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import HotspotClassifierPreprocessor, setup_logging


def main():
    """Parse arguments and run hotspot classifier preprocessing."""
    # Configure logging system
    setup_logging(log_dir=str(project_root / 'logs'))
    
    # Get logger for this script
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="Preprocess annotated thermal images for binary hotspot classification."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/hotspot_classification",
        help="Directory with annotated images (default: data/processed/hotspot_classification)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/hotspot_classification_enhanced",
        help="Directory to save processed images (default: data/processed/hotspot_classification_enhanced)"
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
        "--color_mode",
        type=str,
        choices=['hsv', 'lab', 'ycrcb', 'rgb'],
        default='hsv',
        help="Color space to use for processing (default: hsv)"
    )
    parser.add_argument(
        "--no_enhancement",
        action="store_true",
        help="Disable hot region enhancement"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Binary Hotspot Classification Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target size: {args.target_size[0]}x{args.target_size[1]}")
    logger.info(f"Color mode: {args.color_mode.upper()}")
    logger.info(f"Hot region enhancement: {not args.no_enhancement}")
    logger.info("=" * 60)
    
    try:
        # Verify input directory exists
        input_path = Path(args.input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {args.input_dir}")
            logger.info("Run './bin/annotate_hotspots.sh' first to generate annotations")
            sys.exit(1)
        
        preprocessor = HotspotClassifierPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_size=tuple(args.target_size),
            color_mode=args.color_mode,
            enhance_hot_regions=not args.no_enhancement
        )
        
        preprocessor.process_directory(preserve_structure=True)
        
        logger.info("✓ Hotspot classification preprocessing completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid parameter: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

