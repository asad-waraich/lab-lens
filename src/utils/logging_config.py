"""
Centralized Logging Configuration for Lab Lens
Provides consistent logging setup across all modules
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class LabLensLogger:
    """Centralized logging configuration for the Lab Lens project"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def setup_logging(
        cls,
        log_level: str = "INFO",
        log_dir: str = "logs",
        log_to_file: bool = True,
        log_to_console: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        module_name: Optional[str] = None
    ) -> logging.Logger:
        """
        Set up logging configuration for the Lab Lens project
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            log_to_file: Whether to log to files
            log_to_console: Whether to log to console
            max_file_size: Maximum size of log files before rotation
            backup_count: Number of backup files to keep
            module_name: Name of the module requesting the logger
            
        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls._initialize_logging_config()
            cls._initialized = True
        
        # Create logs directory if it doesn't exist
        if log_to_file:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Get or create logger
        logger_name = module_name or "lab_lens"
        if logger_name not in cls._loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, log_level.upper()))
            
            # Clear existing handlers to avoid duplicates
            logger.handlers.clear()
            
            # Create formatters
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            simple_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            
            # Console handler
            if log_to_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(getattr(logging, log_level.upper()))
                console_handler.setFormatter(simple_formatter)
                logger.addHandler(console_handler)
            
            # File handlers
            if log_to_file:
                # Main log file
                main_log_file = os.path.join(log_dir, f"lab_lens_{datetime.now().strftime('%Y%m%d')}.log")
                file_handler = logging.handlers.RotatingFileHandler(
                    main_log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count
                )
                file_handler.setLevel(logging.DEBUG)  # File gets all levels
                file_handler.setFormatter(detailed_formatter)
                logger.addHandler(file_handler)
                
                # Error log file
                error_log_file = os.path.join(log_dir, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
                error_handler = logging.handlers.RotatingFileHandler(
                    error_log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(detailed_formatter)
                logger.addHandler(error_handler)
            
            cls._loggers[logger_name] = logger
        
        return cls._loggers[logger_name]
    
    @classmethod
    def _initialize_logging_config(cls):
        """Initialize logging configuration"""
        # Configure root logger to prevent duplicate messages
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        
        # Remove default handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    @classmethod
    def get_logger(cls, module_name: str) -> logging.Logger:
        """Get an existing logger or create a new one with default settings"""
        if module_name not in cls._loggers:
            return cls.setup_logging(module_name=module_name)
        return cls._loggers[module_name]
    
    @classmethod
    def log_performance_metrics(cls, logger: logging.Logger, operation: str, 
                              duration: float, **kwargs):
        """Log performance metrics for operations"""
        metrics = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        logger.info(f"PERFORMANCE: {json.dumps(metrics)}")
    
    @classmethod
    def log_data_metrics(cls, logger: logging.Logger, operation: str, 
                        records_processed: int, **kwargs):
        """Log data processing metrics"""
        metrics = {
            "operation": operation,
            "records_processed": records_processed,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        logger.info(f"DATA_METRICS: {json.dumps(metrics)}")


def get_logger(module_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Convenience function to get a logger for a module
    
    Args:
        module_name: Name of the module (usually __name__)
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    return LabLensLogger.setup_logging(log_level=log_level, module_name=module_name)


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls with parameters and execution time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Function {func.__name__} completed successfully in {duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_data_operation(logger: logging.Logger, operation_name: str):
    """Decorator to log data operations with metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"Starting {operation_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                # Try to extract record count from result if it's a DataFrame
                if hasattr(result, '__len__'):
                    record_count = len(result)
                    LabLensLogger.log_data_metrics(logger, operation_name, record_count, duration=duration)
                else:
                    LabLensLogger.log_performance_metrics(logger, operation_name, duration)
                
                logger.info(f"Completed {operation_name} successfully")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {operation_name} after {duration:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator
