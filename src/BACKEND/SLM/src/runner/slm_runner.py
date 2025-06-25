from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from llama_cpp import Llama

from ..models.base_models import SLMConfig
from .exceptions import ModelInitializationError, GenerationError, ErrorCode
from ..utils.sanity_checker import SanityChecker
from ..prompt_handling import PromptHandler

model_path_phi4="src/SLM/models/Phi-4-mini-instruct-Q5_K_M.gguf"

class SLMRunner:
    """Main class for running Structured Language Models using llama.cpp."""
    
    def __init__(self, config: Union[SLMConfig, Dict[str, Any]]):
        """
        Initialize the SLM runner.
        
        Args:
            config: Either a SLMConfig instance or a dictionary of configuration parameters
        """
        # Convert dict config to SLMConfig if necessary
        self.config = (
            config if isinstance(config, SLMConfig) 
            else SLMConfig.parse_obj(config)
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize prompt handler if enabled
        self.prompt_handler = None
        if self.config.model.use_prompt:
            current_dir = Path(__file__).parent.parent
            self.prompt_handler = PromptHandler(
                tools_path=current_dir / "prompt_handling/tools.json"
            )
        
        # Run sanity checks
        self._run_sanity_checks()
        
        # Initialize model
        self._initialize_model()
    
    def _run_sanity_checks(self) -> None:
        """
        Run system sanity checks before model initialization.
        
        Raises:
            FileNotFoundError: If the model file does not exist
            ValueError: If configuration is invalid
        """
        # Skip checks for pretrained models as they'll be downloaded
        if self.config.model.is_local:
            if not self.config.model.model_path:
                raise ValueError("model_path is required for local models")
                
            checks = SanityChecker.run_all_checks(self.config.model.model_path)
            
            if not checks["model_exists"]:
                raise FileNotFoundError(f"Model not found at {self.config.model.model_path}")
            
            if not checks["enough_memory"]:
                self.logger.warning(
                    f"Low system memory ({checks['available_memory_gb']:.2f}GB). "
                    "This might affect performance."
                )
            
            # Update config based on checks
            if self.config.hardware.n_threads is None:
                self.config.hardware.n_threads = checks["optimal_threads"]
            
            if checks["gpu_available"] and self.config.hardware.n_gpu_layers == 0:
                self.logger.info(
                    f"GPU {checks['gpu_name']} available. Consider setting n_gpu_layers "
                    f"(recommended: {checks['recommended_gpu_layers']})."
                )
    
    def _initialize_model(self):
        """
        Initialize the llama.cpp model with current configuration.
        
        Raises:
            FileNotFoundError: If the model file cannot be found
            ModelInitializationError: If model initialization fails
            ValueError: If configuration is invalid
        """
        try:
            # Validate configuration
            self.config.model.validate_configuration()
            
            # Initialize based on source
            if self.config.model.is_pretrained:
                print(f"Loading pretrained model from {self.config.model.pretrained.repo_id}")
                self.logger.info(f"Loading pretrained model from {self.config.model.pretrained.repo_id}")
                # self.model = Llama.from_pretrained(
                #     repo_id=self.config.model.pretrained.repo_id,
                #     filename=self.config.model.pretrained.filename,
                #     n_ctx=self.config.model.context_size,
                #     n_threads=self.config.hardware.n_threads,
                #     n_gpu_layers=self.config.hardware.n_gpu_layers,
                #     verbose = self.config.model.verbose,
                #     **self.config.model_kwargs
                # )
                self.model = Llama(
                    model_path=model_path_phi4,
                    n_ctx=self.config.model.context_size,
                    n_threads=self.config.hardware.n_threads,
                    n_gpu_layers=self.config.hardware.n_gpu_layers,
                    verbose = self.config.model.verbose,
                    **self.config.model_kwargs
                )
            else:
                print(f"Loading local model from {self.config.model.model_path}")
                self.logger.info(f"Loading model from local path: {self.config.model.model_path}")
                self.model = Llama(
                    model_path=str(self.config.model.model_path),
                    n_ctx=self.config.model.context_size,
                    n_threads=self.config.hardware.n_threads,
                    n_gpu_layers=self.config.hardware.n_gpu_layers,
                    **self.config.model_kwargs
                )
            self.logger.info("Model initialized successfully")
        except FileNotFoundError as e:
            self.logger.error(f"Model file not found: {str(e)}")
            raise ModelInitializationError(
                message=f"Model file not found: {str(e)}",
                code=ErrorCode.MODEL_NOT_FOUND,
                context={"path": str(self.config.model.model_path)}
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise ModelInitializationError(
                message=f"Model initialization failed: {str(e)}",
                code=ErrorCode.INVALID_MODEL_CONFIG,
                context={
                    "model_type": self.config.model.model_type,
                    "n_gpu_layers": self.config.hardware.n_gpu_layers,
                    "n_threads": self.config.hardware.n_threads
                }
            )

    def generate(self, 
                user_query: str,
                system_behavior: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate text using the model with structured prompts and tools.
        
        Args:
            user_query (str): The user's query or instruction
            system_behavior (str, optional): Override default system behavior
            **kwargs: Override default generation parameters
            
        Returns:
            Dict[str, Any]: Generation results including generated text and metadata
        """
        # Construct prompt based on configuration
        if self.config.model.use_prompt and self.prompt_handler:
            if system_behavior:
                self.prompt_handler.system_behavior = system_behavior
            prompt = self.prompt_handler.construct_prompt(user_query)
        else:
            # Use simple prompt template if prompt handler is disabled
            prompt = self.config.prompt_template.format(
                instruction=user_query,
                input=""
            )
        
        # Merge generation parameters from config with any overrides
        generation_config = self.config.generation.dict()
        params = {
            k: v for k, v in generation_config.items()
            if v is not None  # Skip None values
        }
        params.update(kwargs)  # Override with any user-provided parameters
        
        try:
            # Generate response
            # out = self.model(
            #     prompt,
            #     **params,
            # )
            # response = ""
            # for chunk in out:
            #     response += chunk["choices"][0]["text"]
            #     print(chunk["choices"][0]["text"], end="", flush=True)

            response = self.model(
                prompt,
                **params
            )

            return response
            # return {
            #     "generated_text": response,
            #     "usage": {
            #         "prompt_tokens": len(self.model.tokenizer().encode(prompt)),
            #         "generated_tokens": len(self.model.tokenizer().encode(response)),
            #     },
            #     "status": "success"
            # }
        
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise GenerationError(
                message=error_msg,
                code=ErrorCode.INVALID_PARAMETERS if "parameter" in str(e).lower() 
                     else ErrorCode.UNKNOWN_ERROR,
                context={
                    "query_length": len(user_query),
                    "params": params,
                    "original_error": str(e)
                }
            )
    
    def __del__(self):
        """
        Cleanup when the instance is deleted.
        The Llama instance and its resources will be automatically cleaned up.
        """
        if hasattr(self, 'model'):
            try:
                del self.model
            except Exception as e:
                # Log but don't raise as this is destructor
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Error during cleanup: {str(e)}")