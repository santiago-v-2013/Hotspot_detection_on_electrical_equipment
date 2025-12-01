# Image_download/utils.py
import logging
import os
import re
from urllib.parse import unquote

logger = logging.getLogger(__name__)

def parse_url_info(url: str) -> tuple[str, str] | None:
    """Extracts the equipment type and filename from the URL."""
    try:
        path_match = re.search(r'/Infrared Power Equipment Dataset/([^/]+)/([^/]+)$', url)
        filename_match = re.search(r'fileName=([^&]+)', url)

        if path_match:
            equipment_type = path_match.group(1).replace(" ", "_")
            if filename_match:
                filename = unquote(filename_match.group(1))
            else:
                filename = unquote(path_match.group(2))
            return equipment_type, filename
        return None
    except Exception:
        return None

def create_directory_if_not_exists(path: str):
    """Creates a directory if it does not already exist, logging the action."""
    if not os.path.exists(path):
        logger.info(f"Creating directory: {path}")
        os.makedirs(path)