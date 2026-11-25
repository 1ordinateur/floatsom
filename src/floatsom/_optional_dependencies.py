"""Load optional dependencies and give an install hint when one is missing."""

import importlib

CUPY_INSTALL_HINT = (
    "FloatSOM needs CuPy to train or use a SOM on the GPU, but CuPy is not installed. "
    "Install the extra that matches your CUDA version: "
    "pip install 'floatsom[cuda12]' or pip install 'floatsom[cuda13]'. "
    "For AMD GPUs, install CuPy for your ROCm version and then 'floatsom[rocm]'. "
    "Run 'python -m floatsom' to check the GPU setup."
)

RAY_INSTALL_HINT = (
    "Multi-GPU processing (ProcessingConfig.ray_config) needs Ray, but Ray is not installed. "
    "Install it with: pip install 'floatsom[ray]'. "
    "For single-GPU processing, set ray_config=None."
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
