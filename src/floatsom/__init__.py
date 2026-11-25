"""FloatSOM: GPU-accelerated self-organizing maps.

Start here::

    import floatsom
    som = floatsom.train_som(data)          # data: (n_samples, n_features)
    clusters = som.predict(data)
    som.save("model.floatsom")

Names load on first use. The config classes work with no CuPy. The training
names need CuPy, and give an install hint when it is missing.
"""

# Ray reads the GPU visibility variables at ``import ray``. On an AMD ROCm host,
# change them to the form that Ray and HIP accept before a floatsom module imports Ray.
from .gpu_visibility import normalize_rocm_visible_devices

normalize_rocm_visible_devices()

import importlib

from ._optional_dependencies import require_cupy

# Public name -> (module, needs CuPy).
_EXPORTS = {
    "FloatSOMParams": ("floatsom.floatsom_params", False),
    "SamplingConfig": ("floatsom.floatsom_params", False),
    "ProcessingConfig": ("floatsom.floatsom_params", False),
    "TopologyConfig": ("floatsom.floatsom_params", False),
    "RayConfig": ("floatsom.floatsom_params", False),
    "PROFILE_NAMES": ("floatsom.defaults", False),
    "load_profile": ("floatsom.defaults", False),
    "check_environment": ("floatsom.environment", False),
    "suggest_grid_size": ("floatsom.api", True),
    "train_som": ("floatsom.api", True),
    "create_floatsom": ("floatsom.base.floatsom_factories", True),
    "FloatSOM": ("floatsom.base.floatsom", True),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'floatsom' has no attribute {name!r}")
    module_name, needs_cupy = _EXPORTS[name]
    if needs_cupy:
        require_cupy()
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
