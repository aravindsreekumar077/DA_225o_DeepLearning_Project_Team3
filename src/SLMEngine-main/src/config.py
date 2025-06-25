"""Default configurations for SLMEngine."""

from pathlib import Path
from .models.base_models import (
    SLMConfig,
    ModelConfig,
    HardwareConfig,
    GenerationConfig,
    SystemRequirements,
    ModelSource,
    PretrainedModelConfig
)

# Default paths
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Default model settings
DEFAULT_MODEL_CONFIG = ModelConfig(
    source=ModelSource.LOCAL,
    model_path=MODELS_DIR / "model.bin",  # Change this to your model path
    model_type="llama",
    context_size=2048,
    verbose=False,
)

# Hardware settings
DEFAULT_HARDWARE_CONFIG = HardwareConfig(
    n_gpu_layers=4,  # Set to higher number to use GPU
    main_gpu=0,
    n_threads=None  # Will be auto-detected
)

# Generation settings
DEFAULT_GENERATION_CONFIG = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_tokens=256,
    repeat_penalty=1.1
)

# System requirements
DEFAULT_SYSTEM_REQUIREMENTS = SystemRequirements(
    min_memory_gb=8.0,
    cpu_usage_percent=0.8,
    gpu_vram_per_layer=2.0,
    default_threads=4
)

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """{instruction}

{input}

Response:"""

def get_default_config() -> SLMConfig:
    """Get the default configuration."""
    return SLMConfig(
        model=DEFAULT_MODEL_CONFIG,
        hardware=DEFAULT_HARDWARE_CONFIG,
        generation=DEFAULT_GENERATION_CONFIG,
        system_requirements=DEFAULT_SYSTEM_REQUIREMENTS,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    )

# Example configurations for different use cases
def get_gpu_config(model_path: Path = DEFAULT_MODEL_CONFIG.model_path) -> SLMConfig:
    """Get configuration optimized for GPU usage."""
    config = get_default_config()
    config.hardware.n_gpu_layers = 32  # Use more GPU layers
    config.model.model_path = model_path
    return config

def get_cpu_config(model_path: Path = DEFAULT_MODEL_CONFIG.model_path) -> SLMConfig:
    """Get configuration optimized for CPU usage."""
    config = get_default_config()
    config.hardware.n_gpu_layers = 0
    config.hardware.n_threads = 8  # Use fixed number of threads
    config.model.model_path = model_path
    return config

def get_pretrained_config(
    repo_id: str,
    filename: str = None
) -> SLMConfig:
    """Get configuration for using a pretrained model."""
    config = get_default_config()
    config.model.source = ModelSource.PRETRAINED
    config.model.pretrained = PretrainedModelConfig(
        repo_id=repo_id,
        filename=filename
    )
    return config
