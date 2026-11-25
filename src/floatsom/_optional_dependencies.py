"""Load optional dependencies and give an install hint when one is missing."""

import importlib

# The install source for the hints: the package is not on PyPI.
GIT_SOURCE = "git+https://github.com/1ordinateur/floatsom.git"

CUPY_INSTALL_HINT = (
    "FloatSOM needs CuPy to train or use a SOM on the GPU, but CuPy is not installed. "
    "Run nvidia-smi and read 'CUDA Version'. For 12.x, install with: "
    f'pip install "floatsom[cuda12] @ {GIT_SOURCE}". '
    "For 13.x, use floatsom[cuda13]. "
    "For AMD GPUs, install CuPy for your ROCm version and then floatsom[rocm]. "
    "Run 'python -m floatsom' to check the GPU setup."
)

RAY_INSTALL_HINT = (
    "Multi-GPU processing (ProcessingConfig.ray_config) needs Ray, but Ray is not installed. "
    f'Install it with: pip install "floatsom[ray] @ {GIT_SOURCE}". '
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
