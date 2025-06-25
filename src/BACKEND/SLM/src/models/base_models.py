from pydantic import BaseModel, Field # type: ignore
from typing import Dict, Any, Optional, Union
from pathlib import Path
from enum import Enum

class ModelSource(Enum):
    """Enum for model source types."""
    LOCAL = "local"
    PRETRAINED = "pretrained"

class PretrainedModelConfig(BaseModel):
    """Configuration for pretrained models."""
    repo_id: str
    filename: Optional[str] = None

class ToolParameter(BaseModel):
    """Pydantic model for tool parameters."""
    description: str
    type: str
    default: Optional[Union[str, int, float, bool]] = None

class Tool(BaseModel):
    """Pydantic model for tools."""
    name: str
    description: str
    parameters: Dict[str, ToolParameter]

class GenerationConfig(BaseModel):
    """Pydantic model for text generation settings."""
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    max_tokens: int = Field(default=256, ge=0)
    repeat_penalty: float = Field(default=1.1, ge=0.0)
    stream: bool = Field(default=True, description="Enable streaming output")

class ModelConfig(BaseModel):
    """Pydantic model for model settings."""
    source: ModelSource = ModelSource.LOCAL
    model_path: Optional[Path] = None
    model_type: str = "llama"
    context_size: int = Field(default=2048, ge=0)
    pretrained: Optional[PretrainedModelConfig] = None
    use_prompt: bool = Field(default=True, description="Enable prompt handler usage")
    verbose: bool = Field(default=False, description="Enable verbose logging")

    @property
    def is_local(self) -> bool:
        """Check if this is a local model configuration."""
        return self.source == ModelSource.LOCAL
    
    @property
    def is_pretrained(self) -> bool:
        """Check if this is a pretrained model configuration."""
        return self.source == ModelSource.PRETRAINED

    def validate_configuration(self) -> None:
        """
        Validate the model configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        if self.is_local and not self.model_path:
            raise ValueError("model_path is required for local models")
        if self.is_pretrained and not self.pretrained:
            raise ValueError("pretrained configuration is required for pretrained models")
        if self.is_pretrained and not self.pretrained.repo_id:
            raise ValueError("repo_id is required in pretrained configuration")

class HardwareConfig(BaseModel):
    """Pydantic model for hardware settings."""
    n_gpu_layers: int = Field(default=0, ge=0)
    main_gpu: int = Field(default=0, ge=0)
    n_threads: Optional[int] = None

class SystemRequirements(BaseModel):
    """System requirements for running models."""
    min_memory_gb: float = Field(default=8.0, ge=0.0)
    cpu_usage_percent: float = Field(default=0.8, ge=0.0, le=1.0)
    gpu_vram_per_layer: float = Field(default=2.0, ge=0.0)
    default_threads: int = Field(default=4, ge=1)

class SLMConfig(BaseModel):
    """Main configuration model for SLM."""
    model: ModelConfig
    hardware: HardwareConfig
    generation: GenerationConfig
    system_requirements: SystemRequirements = Field(default_factory=SystemRequirements)
    prompt_template: str = "{instruction}\n\n{input}\n\nResponse:"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True  # To allow Path objects
