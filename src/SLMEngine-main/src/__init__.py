"""
SLMEngine - Structured Language Model Engine
A toolkit for running and managing LLM interactions with structured prompts and tools.
"""

from .models import (
    SLMConfig,
    Tool,
    ToolParameter,
    GenerationConfig,
    ModelConfig,
    HardwareConfig
)
from .runner.slm_runner import SLMRunner
from .prompt_handling.prompt_handler import PromptHandler

__all__ = [
    'SLMConfig',
    'SLMRunner',
    'PromptHandler',
    'Tool',
    'ToolParameter',
    'GenerationConfig',
    'ModelConfig',
    'HardwareConfig'
]

__version__ = '0.1.0'