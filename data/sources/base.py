"""
Data source abstraction for FloatSOM - SOLID compliant data access layer
Implements Strategy pattern for different data sources
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Iterator, Optional, Any
import numpy as np
import cupy as cp


class DataSource(ABC):
    """
    Abstract base class for data sources
    Follows Single Responsibility: Only responsible for data access
    Follows Dependency Inversion: High-level modules depend on this abstraction
    """
    
    @staticmethod
    def _validate_chunk_size(chunk_size: int) -> int:
        """Validate chunk_size and return a normalized positive integer."""
        if isinstance(chunk_size, (bool, np.bool_)):
            raise TypeError("chunk_size must be an integer")
        if not isinstance(chunk_size, (int, np.integer)):
            raise TypeError("chunk_size must be an integer")
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        return chunk_size

    @staticmethod
    def _validate_max_samples(max_samples: Optional[int]) -> Optional[int]:
        """Validate max_samples and return a normalized value."""
        if max_samples is None:
            return None
        if isinstance(max_samples, (bool, np.bool_)):
            raise TypeError("max_samples must be an integer or None")
        if not isinstance(max_samples, (int, np.integer)):
            raise TypeError("max_samples must be an integer or None")
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        return max_samples

    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """
        Get the shape of the dataset without loading it
        
        Returns:
            Tuple of (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def get_initialization_sample(
        self,
        max_samples: Optional[int] = None,
        *,
        use_gpu: bool = True,
    ) -> Union[np.ndarray, cp.ndarray]:
        """
        Get a sample of data for initialization (e.g., PCA)
        
        Args:
            max_samples: Maximum number of samples to return
            use_gpu: Whether to return a CuPy array (True) or NumPy array (False)
            
        Returns:
            Sample of data for initialization
        """
        pass
    
    @abstractmethod
    def iterate_chunks(
        self,
        chunk_size: int,
        *,
        use_gpu: bool = True,
    ) -> Iterator[Union[np.ndarray, cp.ndarray]]:
        """
        Iterate over data in chunks
        
        Args:
            chunk_size: Size of each chunk
            use_gpu: Whether to yield CuPy arrays (True) or NumPy arrays (False)
            
        Yields:
            Data chunks
        """
        pass
    
    @abstractmethod
    def get_reference(self) -> Any:
        """
        Get a reference to the data that can be passed to selectors/processors
        For arrays: returns the array itself
        For files: returns the file path
        
        Returns:
            Data reference (array or path)
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Check if this data source supports streaming
        
        Returns:
            True if streaming is supported
        """
        pass
    
    @abstractmethod
    def get_memory_requirement(self) -> int:
        """
        Estimate memory requirement in bytes
        
        Returns:
            Estimated memory requirement in bytes
        """
        pass
    
    def cleanup(self) -> None:
        """
        Optional cleanup method for resource management
        """
        pass
