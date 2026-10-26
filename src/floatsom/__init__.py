# Ray reads the GPU visibility variables at ``import ray``. On an AMD ROCm host,
# change them to the form that Ray and HIP accept before a floatsom module imports Ray.
from .gpu_visibility import normalize_rocm_visible_devices

normalize_rocm_visible_devices()
