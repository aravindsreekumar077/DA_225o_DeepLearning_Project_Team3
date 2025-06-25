"""Exceptions module for SLMRunner.

This module contains all custom exceptions used by the SLM (Structured Language Model) runner.
Each exception includes an error code, message, and optional context for better error handling
and debugging.
"""

from enum import Enum
from typing import Any, Dict, Optional

class ErrorCode(Enum):
    """Error codes for SLMRunner exceptions."""
    # Model initialization errors (1xxx)
    MODEL_NOT_FOUND = 1001
    INVALID_MODEL_CONFIG = 1002
    GPU_ERROR = 1003
    MEMORY_ERROR = 1004
    THREAD_ERROR = 1005
    
    # Generation errors (2xxx)
    INVALID_PROMPT = 2001
    CONTEXT_LENGTH_EXCEEDED = 2002
    GENERATION_TIMEOUT = 2003
    INVALID_PARAMETERS = 2004
    
    # General errors (9xxx)
    UNKNOWN_ERROR = 9999

class SLMRunnerError(Exception):
    """Base exception for SLMRunner errors.
    
    Attributes:
        message (str): Human-readable error message
        code (ErrorCode): Specific error code
        context (dict, optional): Additional context about the error
    """
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return formatted error message with code and context."""
        error_msg = f"[{self.code.name}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            error_msg += f"\nContext: {context_str}"
        return error_msg

class ModelInitializationError(SLMRunnerError):
    """Raised when model initialization fails.
    
    This can happen due to:
    - Model file not found
    - Invalid model configuration
    - GPU initialization errors
    - Insufficient memory
    - Thread allocation issues
    """
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.INVALID_MODEL_CONFIG,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, context)

class GenerationError(SLMRunnerError):
    """Raised when text generation fails.
    
    This can happen due to:
    - Invalid prompt format
    - Context length exceeded
    - Generation timeout
    - Invalid generation parameters
    """
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.INVALID_PARAMETERS,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, context)
