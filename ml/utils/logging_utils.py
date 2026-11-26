"""Logging configuration helper for standard structured logging across crop-prognosis."""

import logging
import sys
from typing import Optional

def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Create or retrieve a configured Python standard logger.
    
    Args:
        name: Name of the logger (typically __name__).
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional file path to store log output alongside stderr.
        
    Returns:
        Configured logging.Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        
    return logger
