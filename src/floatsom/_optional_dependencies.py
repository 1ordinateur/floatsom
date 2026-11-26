"""Load optional dependencies and give an install hint when one is missing."""

import importlib

# The install source for the hints: the package is not on PyPI.
GIT_SOURCE = "git+https://github.com/1ordinateur/floatsom.git"

CUPY_INSTALL_HINT = (
    "FloatSOM needs CuPy to train or use a SOM on the GPU, but CuPy is not installed. "
    "Run nvidia-smi and read 'CUDA Version'. For 12.x, install with: "
    f'pip install "floatsom[cuda12] @ {GIT_SOURCE}". '
    "For 13.x, use floatsom[cuda13]. "
    "For AMD GPUs, install CuPy for your ROCm version, then install floatsom with no extra. "
    "Run 'python -m floatsom' to check the GPU setup."
)

RAY_INSTALL_HINT = (
    "Multi-GPU processing (ProcessingConfig.ray_config) needs Ray, but Ray is not installed. "
    f'Install Ray, NCCL and PyTorch with: pip install "floatsom[ray-cuda12] @ {GIT_SOURCE}" '
    "(use floatsom[ray-cuda13] for CUDA 13.x). "
    "If a cluster module gives CuPy, NCCL and PyTorch, use floatsom[ray]. "
    "For single-GPU processing, set ray_config=None."
)

NCCL_INSTALL_HINT = (
    "Ray found no NCCL backend for multi-GPU processing. "
    "Ray needs the NCCL library, and Ray 2.50 and later also need PyTorch, also when "
    "FloatSOM does not use PyTorch. "
    f'Install CuPy, Ray, NCCL and PyTorch with: pip install "floatsom[ray-cuda12] @ {GIT_SOURCE}" '
    "(use floatsom[ray-cuda13] for CUDA 13.x). "
    "With a cluster module, install NCCL and PyTorch, or use Ray earlier than 2.50."
)


def _require(module_name: str, hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise ImportError(hint) from error


def require_cupy():
    """Return the ``cupy`` module, or raise ImportError with an install hint."""
    return _require("cupy", CUPY_INSTALL_HINT)


def require_ray():
    """Return the ``ray`` module, or raise ImportError with an install hint."""
    return _require("ray", RAY_INSTALL_HINT)
