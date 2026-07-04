"""
File-based data source implementation
Handles loading data from disk
"""

from typing import Union, Tuple, Iterator, Optional
import numpy as np
import cupy as cp
from pathlib import Path
import logging

from .base import DataSource
from ..dataset_format import detect_dataset_format
from ..zarr_utils import open_array_read
from ...floatsom_params import RayConfig

logger = logging.getLogger(__name__)


class FileDataSource(DataSource):
    """
    Data source for file-based data
    Single Responsibility: Manages access to file-based data
    Open/Closed: Can be extended for new file formats without modification
    """
    
    def __init__(self, 
                 file_path: str,
                 ray_config: Optional[RayConfig] = None):
        """
        Initialize with file path and optional Ray configuration
        
        Args:
            file_path: Path to data file
            ray_config: Optional Ray configuration
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Store ray_config
        self.ray_config = ray_config
        
        # Cache shape (lazy initialization)
        self._shape = None
        self._data = None
        self._store = None  # Used for FastArrayStore backing resources
        
    def _ensure_data_loaded(self):
        """Lazy loading of data"""
        if self._shape is None:
            fmt = detect_dataset_format(self.file_path)

            if fmt == "zarr":
                try:
                    z = open_array_read(str(self.file_path)).array
                except Exception as exc:
                    raise ValueError(f"Failed to load Zarr store: {self.file_path}") from exc

                shape = getattr(z, "shape", None)
                if not shape:
                    raise ValueError(f"Zarr store has no shape: {self.file_path}")
                if len(shape) != 2:
                    raise ValueError(
                        f"Zarr store must be 2D for FloatSOM training, got shape {shape}: "
                        f"{self.file_path}"
                    )
                self._shape = shape
                self._data = z
                self._format = "zarr"
                return

            if fmt == "fast_array":
                from ..fast_array_store import FastArrayStore

                store = FastArrayStore(str(self.file_path), mode="r")
                self._store = store
                data = store.mmap_array
                if getattr(data, "ndim", 1) == 1:
                    data = data.reshape(-1, 1)
                self._data = data
                self._shape = data.shape
                self._format = "fast_array"
                return

            suffix = self.file_path.suffix.lower()
            if suffix == ".npy":
                data = np.load(str(self.file_path), allow_pickle=False)
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                self._shape = data.shape
                self._data = data
                self._format = "npy"
                return

            if suffix == ".npz":
                with np.load(str(self.file_path), allow_pickle=False) as npz:
                    if not npz.files:
                        raise ValueError(f".npz archive contains no arrays: {self.file_path}")
                    if len(npz.files) != 1:
                        raise ValueError(
                            f".npz archive {self.file_path} contains multiple arrays "
                            f"({', '.join(npz.files)}). Use an explicit key-aware loader."
                        )
                    data = np.asarray(npz[npz.files[0]])
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                self._shape = data.shape
                self._data = data
                self._format = "npz"
                return

            if suffix in [".csv", ".txt"]:
                data = np.loadtxt(str(self.file_path), delimiter=",")
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                self._shape = data.shape
                self._data = data
                self._format = "csv"
                return

            if suffix == ".parquet":
                try:
                    import pandas as pd  # type: ignore
                except ImportError as exc:
                    raise ImportError(
                        "pandas is required to load parquet files (install pandas or convert your data)."
                    ) from exc
                df = pd.read_parquet(str(self.file_path))
                data = df.to_numpy()
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                self._shape = data.shape
                self._data = data
                self._format = "parquet"
                return

            raise ValueError(f"Unsupported dataset format: {self.file_path}")
    
    def get_shape(self) -> Tuple[int, int]:
        """Get dataset shape from file metadata"""
        self._ensure_data_loaded()
        return self._shape
    
    def get_initialization_sample(
        self,
        max_samples: Optional[int] = None,
        *,
        use_gpu: bool = True,
    ) -> Union[np.ndarray, cp.ndarray]:
        """
        Load sample for initialization, optimized for GPU memory
        
        Args:
            max_samples: Maximum samples to load
            use_gpu: Whether to return a CuPy array (True) or NumPy array (False)
        Returns:
            Sample data for initialization
        """
        self._ensure_data_loaded()
        n_samples, n_features = self._shape

        validated_max_samples = self._validate_max_samples(max_samples)
        if validated_max_samples is None:
            if use_gpu:
                # Calculate maximum safe samples for GPU
                device = cp.cuda.Device()
                free_mem = device.mem_info[0]

                # Calculate memory needed per sample
                bytes_per_sample = n_features * 4  # float32

                # Use safety factor for PCA computation overhead
                safety_factor = 0.3
                max_bytes_for_data = free_mem * safety_factor
                validated_max_samples = min(
                    int(max_bytes_for_data / bytes_per_sample),
                    n_samples
                )
                validated_max_samples = max(1, validated_max_samples)

                logger.info(
                    f"Loading {validated_max_samples:,} samples for initialization "
                    f"({validated_max_samples * bytes_per_sample / (1024**3):.2f} GB)"
                )
            else:
                validated_max_samples = n_samples
        else:
            validated_max_samples = min(validated_max_samples, n_samples)
        
        sample_data = self._data[:validated_max_samples]
        
        # Ensure numpy array
        if not isinstance(sample_data, np.ndarray):
            sample_data = np.asarray(sample_data)
        
        if use_gpu:
            return cp.asarray(sample_data)
        return np.asarray(sample_data)
    
    def iterate_chunks(
        self,
        chunk_size: int,
        *,
        use_gpu: bool = True,
    ) -> Iterator[Union[np.ndarray, cp.ndarray]]:
        """
        Stream chunks from file
        
        Args:
            chunk_size: Number of samples per chunk
            use_gpu: Whether to yield CuPy arrays (True) or NumPy arrays (False)
        Yields:
            Data chunks from file
        """
        self._ensure_data_loaded()
        chunk_size = self._validate_chunk_size(chunk_size)
        n_samples = self._shape[0]
        
        # Calculate number of chunks
        num_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)
            
            chunk_data = self._data[start_idx:end_idx]
            
            # Ensure numpy array
            if not isinstance(chunk_data, np.ndarray):
                chunk_data = np.asarray(chunk_data)

            if use_gpu:
                yield cp.asarray(chunk_data)
                cp.get_default_memory_pool().free_all_blocks()
            else:
                yield np.asarray(chunk_data)
    
    def get_reference(self) -> str:
        """
        Return file path as reference for processors
        
        Returns:
            File path string
        """
        return str(self.file_path)
    
    def supports_streaming(self) -> bool:
        """
        File sources support streaming
        
        Returns:
            True
        """
        return True
    
    def get_memory_requirement(self) -> int:
        """
        Estimate memory requirement (just for one chunk when streaming)
        
        Returns:
            Memory requirement in bytes
        """
        self._ensure_data_loaded()
        
        # Get configured chunk size
        if self.ray_config and hasattr(self.ray_config, 'chunk_size'):
            chunk_size = self.ray_config.chunk_size
        else:
            chunk_size = 10000  # Default chunk size
        
        n_features = self._shape[1]
        bytes_per_sample = n_features * 4  # float32
        return chunk_size * bytes_per_sample
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self._store is not None and hasattr(self._store, "close"):
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None
        self._data = None
        if hasattr(self, '_shape'):
            self._shape = None
