"""
Thermal Image application for Electrical Panel Hot Spot Detection
"""

__version__ = "1.0.0"
__author__ = "Hot Spot Detection Team"

from .Image_download.downloader import ImageDownloader
from .Image_download.utils import parse_url_info, create_directory_if_not_exists
from .logger_cfg import setup_logging
from .preprocessing import (
    EquipmentClassifierPreprocessor,
    HotspotClassifierPreprocessor,
    HotspotDetectorPreprocessor,
    BasePreprocessor,
)

__all__ = [
    "ImageDownloader",
    "parse_url_info",
    "create_directory_if_not_exists",
    "setup_logging",
    "BasePreprocessor",
    "EquipmentClassifierPreprocessor",
    "HotspotClassifierPreprocessor",
    "HotspotDetectorPreprocessor",
]
