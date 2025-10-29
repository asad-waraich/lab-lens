"""
Error Handling Utilities for Lab Lens
Provides comprehensive error handling and custom exceptions
"""

from pathlib import Path
import logging
import traceback
import sys
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
import pandas as pd
import numpy as np


class LabLensError(Exception):
    """Base exception class for Lab Lens project"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = pd.Timestamp.now().isoformat()


class DataValidationError(LabLensError):
    """Raised when data validation fails"""
    pass


class DataProcessingError(LabLensError):
    """Raised when data processing operations fail"""
    pass


class BiasDetectionError(LabLensError):
    """Raised when bias detection operations fail"""
    pass


class ConfigurationError(LabLensError):
    """Raised when configuration is invalid or missing"""
    pass


class ExternalServiceError(LabLensError):
    """Raised when external service calls fail"""
    pass


class FileOperationError(LabLensError):
    """Raised when file operations fail"""
    pass


class ModelTrainingError(LabLensError):
    """Raised when model training fails"""
    pass


class ErrorHandler:
    """Centralized error handling utilities"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_file_error(self, operation: str, file_path: str, 
                         original_error: Exception) -> FileOperationError:
        """Handle file operation errors with detailed logging"""
        error_msg = f"File operation '{operation}' failed for '{file_path}': {str(original_error)}"
        
        self.logger.error(error_msg, exc_info=True)
        
        return FileOperationError(
            message=error_msg,
            error_code="FILE_OPERATION_FAILED",
            details={
                "operation": operation,
                "file_path": file_path,
                "original_error": str(original_error),
                "error_type": type(original_error).__name__
            }
        )
    
    def handle_data_error(self, operation: str, data_info: Dict[str, Any], 
                         original_error: Exception) -> DataProcessingError:
        """Handle data processing errors with detailed logging"""
        error_msg = f"Data processing '{operation}' failed: {str(original_error)}"
        
        self.logger.error(error_msg, exc_info=True)
        
        return DataProcessingError(
            message=error_msg,
            error_code="DATA_PROCESSING_FAILED",
            details={
                "operation": operation,
                "data_info": data_info,
                "original_error": str(original_error),
                "error_type": type(original_error).__name__
            }
        )
    
    def handle_validation_error(self, validation_type: str, 
                              validation_details: Dict[str, Any]) -> DataValidationError:
        """Handle data validation errors"""
        error_msg = f"Data validation failed for '{validation_type}'"
        
        self.logger.error(error_msg)
        
        return DataValidationError(
            message=error_msg,
            error_code="VALIDATION_FAILED",
            details={
                "validation_type": validation_type,
                "validation_details": validation_details
            }
        )
    
    def handle_external_service_error(self, service: str, operation: str, 
                                    original_error: Exception) -> ExternalServiceError:
        """Handle external service errors"""
        error_msg = f"External service '{service}' operation '{operation}' failed: {str(original_error)}"
        
        self.logger.error(error_msg, exc_info=True)
        
        return ExternalServiceError(
            message=error_msg,
            error_code="EXTERNAL_SERVICE_FAILED",
            details={
                "service": service,
                "operation": operation,
                "original_error": str(original_error),
                "error_type": type(original_error).__name__
            }
        )


def safe_execute(operation_name: str, logger: logging.Logger, 
                error_handler: ErrorHandler, default_return: Any = None,
                reraise: bool = True):
    """
    Decorator to safely execute operations with comprehensive error handling
    
    Args:
        operation_name: Name of the operation for logging
        logger: Logger instance
        error_handler: ErrorHandler instance
        default_return: Value to return if operation fails and reraise=False
        reraise: Whether to reraise the exception after logging
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                logger.info(f"Completed {operation_name} successfully")
                return result
            
            except FileNotFoundError as e:
                error = error_handler.handle_file_error(operation_name, str(e), e)
                if reraise:
                    raise error
                return default_return
            
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                data_info = {"file_type": "csv", "error": str(e)}
                error = error_handler.handle_data_error(operation_name, data_info, e)
                if reraise:
                    raise error
                return default_return
            
            except (ValueError, TypeError) as e:
                data_info = {"error_type": "data_type", "error": str(e)}
                error = error_handler.handle_data_error(operation_name, data_info, e)
                if reraise:
                    raise error
                return default_return
            
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {str(e)}", exc_info=True)
                if reraise:
                    raise LabLensError(
                        message=f"Unexpected error in {operation_name}: {str(e)}",
                        error_code="UNEXPECTED_ERROR",
                        details={"operation": operation_name, "original_error": str(e)}
                    )
                return default_return
        
        return wrapper
    return decorator


def validate_dataframe(df: pd.DataFrame, required_columns: list, 
                      logger: logging.Logger) -> bool:
    """
    Validate DataFrame structure and content
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        logger: Logger instance
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    error_handler = ErrorHandler(logger)
    
    # Check if DataFrame is empty
    if df.empty:
        raise error_handler.handle_validation_error(
            "empty_dataframe",
            {"message": "DataFrame is empty"}
        )
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise error_handler.handle_validation_error(
            "missing_columns",
            {
                "missing_columns": missing_columns,
                "required_columns": required_columns,
                "available_columns": df.columns.tolist()
            }
        )
    
    # Check for completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        logger.warning(f"Found {empty_rows} completely empty rows")
    
    logger.info(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def validate_file_path(file_path: Union[str, Path], logger: logging.Logger, 
                      must_exist: bool = True) -> bool:
    """
    Validate file path and existence
    
    Args:
        file_path: Path to validate
        logger: Logger instance
        must_exist: Whether file must exist
        
    Returns:
        True if validation passes
        
    Raises:
        FileOperationError: If validation fails
    """
    error_handler = ErrorHandler(logger)
    
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise error_handler.handle_file_error(
                "file_validation",
                str(file_path),
                FileNotFoundError(f"File does not exist: {file_path}")
            )
        
        if must_exist and not path.is_file():
            raise error_handler.handle_file_error(
                "file_validation",
                str(file_path),
                ValueError(f"Path is not a file: {file_path}")
            )
        
        logger.debug(f"File path validation passed: {file_path}")
        return True
        
    except Exception as e:
        if isinstance(e, LabLensError):
            raise
        raise error_handler.handle_file_error("file_validation", str(file_path), e)


def handle_numpy_warnings(logger: logging.Logger):
    """Configure numpy to log warnings instead of printing to stderr"""
    import warnings
    
    def numpy_warning_handler(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"NumPy Warning: {message}")
    
    warnings.showwarning = numpy_warning_handler


def create_error_report(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a comprehensive error report
    
    Args:
        error: Exception instance
        context: Additional context information
        
    Returns:
        Dictionary containing error details
    """
    report = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": pd.Timestamp.now().isoformat(),
        "traceback": traceback.format_exc(),
        "context": context or {}
    }
    
    if isinstance(error, LabLensError):
        report.update({
            "error_code": error.error_code,
            "details": error.details
        })
    
    return report


def log_error_summary(logger: logging.Logger, error: Exception, 
                     operation: str, context: Dict[str, Any] = None):
    """
    Log a comprehensive error summary
    
    Args:
        logger: Logger instance
        error: Exception instance
        operation: Operation that failed
        context: Additional context
    """
    error_report = create_error_report(error, context)
    
    logger.error(f"Error in {operation}: {error_report['error_message']}")
    logger.debug(f"Full error report: {error_report}")
    
    # Log critical errors with full traceback
    if isinstance(error, (DataValidationError, ConfigurationError)):
        logger.critical(f"Critical error in {operation}: {traceback.format_exc()}")


# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in specific operations"""
    
    def __init__(self, operation_name: str, logger: logging.Logger, 
                 error_handler: ErrorHandler, suppress_errors: bool = False):
        self.operation_name = operation_name
        self.logger = logger
        self.error_handler = error_handler
        self.suppress_errors = suppress_errors
        self.start_time = None
    
    def __enter__(self):
        self.start_time = pd.Timestamp.now()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (pd.Timestamp.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            log_error_summary(self.logger, exc_val, self.operation_name)
            self.logger.error(f"Operation {self.operation_name} failed after {duration:.3f}s")
            
            if not self.suppress_errors:
                return False  # Reraise the exception
        else:
            self.logger.info(f"Operation {self.operation_name} completed successfully in {duration:.3f}s")
        
        return self.suppress_errors
