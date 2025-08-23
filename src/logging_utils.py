"""
Logging configuration for AuraNet
Sets up a unified logging system that writes to both console and file
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir='logs', log_file=None, level=logging.INFO):
    """
    Set up logging to file and console.
    
    Args:
        log_dir: Directory to store log files
        log_file: Specific log filename (if None, generates a timestamped name)
        level: Logging level
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"auranet_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Log initial message
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger

def get_logger(name):
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name, typically __name__ of the module
        
    Returns:
        logger: Logger instance
    """
    return logging.getLogger(name)
