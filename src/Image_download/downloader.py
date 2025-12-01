# Image_download/downloader.py
import logging
import requests
from tqdm import tqdm
from pathlib import Path
from .utils import parse_url_info, create_directory_if_not_exists

logger = logging.getLogger(__name__)

class ImageDownloader:
    """Downloads and organizes images, reporting progress."""
    def __init__(self, url_file: str, output_dir: str):
        self.url_file = Path(url_file)
        self.output_dir = Path(output_dir)
        
        if not self.url_file.is_file():
            raise FileNotFoundError(f"URL file not found at: {self.url_file}")

    def _read_urls(self) -> list[str]:
        """Reads non-empty lines from the URL file."""
        with open(self.url_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def run(self):
        """Executes the complete download and organization process."""
        logger.info(f"Starting image download from '{self.url_file}'...")
        urls = self._read_urls()
        
        if not urls:
            logger.warning("URL file is empty. Nothing to download.")
            return

        create_directory_if_not_exists(self.output_dir)

        for url in tqdm(urls, desc="Processing images"):
            info = parse_url_info(url)
            
            if not info:
                logger.warning(f"Could not process URL: {url}")
                continue
            
            equipment_type, file_name = info
            target_dir = self.output_dir / equipment_type
            create_directory_if_not_exists(target_dir)
            
            file_path = target_dir / file_name

            if file_path.exists():
                continue

            try:
                response = requests.get(url, stream=True, timeout=15)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {url}: {e}")

        logger.info("Download process finished!")