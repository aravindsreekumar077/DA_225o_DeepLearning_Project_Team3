import psutil
import torch
from pathlib import Path
from typing import Optional, Tuple

class SanityChecker:
    """Utility class for performing system checks and optimizations."""
    
    @staticmethod
    def check_model_exists(model_path: Path) -> bool:
        """
        Check if the model file exists at the specified path.
        
        Args:
            model_path (Path): Path to the model file
            
        Returns:
            bool: True if model exists, False otherwise
        """
        return model_path.exists() and model_path.is_file()
    
    @staticmethod
    def get_optimal_threads() -> int:
        """
        Determine the optimal number of threads based on system CPU.
        
        Returns:
            int: Recommended number of threads
        """
        # Get physical CPU count (excluding hyperthreading)
        physical_cores = psutil.cpu_count(logical=False)
        
        # If can't determine physical cores, use logical cores
        if physical_cores is None:
            physical_cores = psutil.cpu_count(logical=True)
            if physical_cores is None:
                return 4  # fallback to reasonable default
        
        # Use 80% of available cores to leave some headroom for system
        return max(1, int(physical_cores * 0.8))
    
    @staticmethod
    def check_gpu_availability() -> Tuple[bool, Optional[str], int]:
        """
        Check GPU availability and get recommended number of layers to offload.
        
        Returns:
            Tuple[bool, Optional[str], int]: 
                - GPU available
                - GPU name if available
                - Recommended number of layers to offload
        """
        if not torch.cuda.is_available():
            return False, None, 0
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        # Conservative estimate: 1 layer per 2GB of VRAM
        recommended_layers = max(1, int(gpu_memory / 2))
        
        return True, gpu_name, recommended_layers
    
    @staticmethod
    def check_system_memory() -> Tuple[float, bool]:
        """
        Check available system memory.
        
        Returns:
            Tuple[float, bool]:
                - Available memory in GB
                - Whether there's enough memory (>8GB)
        """
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Consider system has enough memory if >8GB available
        return available_gb, available_gb > 8
    
    @classmethod
    def run_all_checks(cls, model_path: Path) -> dict:
        """
        Run all sanity checks and return results.
        
        Args:
            model_path (Path): Path to the model file
            
        Returns:
            dict: Dictionary containing all check results
        """
        model_exists = cls.check_model_exists(model_path)
        optimal_threads = cls.get_optimal_threads()
        gpu_available, gpu_name, recommended_layers = cls.check_gpu_availability()
        available_memory, enough_memory = cls.check_system_memory()
        
        return {
            "model_exists": model_exists,
            "optimal_threads": optimal_threads,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "recommended_gpu_layers": recommended_layers,
            "available_memory_gb": round(available_memory, 2),
            "enough_memory": enough_memory
        }