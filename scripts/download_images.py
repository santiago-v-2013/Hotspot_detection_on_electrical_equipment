# scripts/download_images.py
import sys
import argparse
import logging
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import ImageDownloader, setup_logging

def main():
    """Parses arguments and runs the downloader with file-based logging."""
    # Configure logging system
    setup_logging(log_dir=str(project_root / 'logs'))
    
    # Get logger for this script
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="A script to download power equipment images.")
    parser.add_argument("--url_file", type=str, required=True, help="Path to the .txt file containing image URLs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the downloaded images will be saved.")
    args = parser.parse_args()

    try:
        downloader = ImageDownloader(
            url_file=args.url_file, 
            output_dir=args.output_dir
        )
        downloader.run()
    except FileNotFoundError as e:
        logger.error(f"Configuration Error: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()