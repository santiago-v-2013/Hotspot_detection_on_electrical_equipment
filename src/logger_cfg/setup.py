"""
Logger configuration module for the vision project.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = "logs", log_file: str = None, log_level: int = logging.INFO):
    """
    Configures the logging system for the application.
    
    Sets up both file and console handlers with appropriate formatting.
    After calling this function, use logging.getLogger(__name__) in your scripts.
    
    Args:
        log_dir: Directory where log files will be stored (used if log_file is not provided)
        log_file: Full path to the log file (optional, overrides log_dir)
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        Logger instance
    """
    # Determine log file path
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_path / 'app.log'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(levelname)s - %(message)s'
    )
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(
        log_file_path,
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simpler output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Silence noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return root_logger


def setup_logger(name: str = None, log_file: Path = None, level: int = logging.INFO):
    """
    Setup logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Setup logging system
    setup_logging(log_file=log_file, log_level=level)
    
    # Return named logger
    if name:
        return logging.getLogger(name)
    else:
        return logging.getLogger()
