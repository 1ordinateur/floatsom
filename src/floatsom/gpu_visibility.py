"""GPU visibility variables on AMD (ROCm) hosts.

Slurm on an AMD host sets ROCR_VISIBLE_DEVICES. Ray stops at ``import ray``
when that variable is set, or when CUDA_VISIBLE_DEVICES differs from
HIP_VISIBLE_DEVICES. ``import floatsom`` calls ``normalize_rocm_visible_devices``
before any floatsom module imports Ray. This module uses the standard library only.
"""
import os
import re
from typing import MutableMapping, Optional

PLATFORM_ENV = "FLOATSOM_GPU_PLATFORM"
ROCM_DEVICE_NODE = "/dev/kfd"
NVIDIA_DEVICE_NODE = "/dev/nvidiactl"
# HIP reads an empty list as "all devices". This value selects no device.
NO_DEVICE = "NoDevFiles"
_DEVICE_INDEX = re.compile(r"0|[1-9][0-9]*")


def is_rocm_host() -> bool:
    """Return True on an AMD ROCm host. FLOATSOM_GPU_PLATFORM ("rocm" or "cuda") overrides the check."""
    platform = os.environ.get(PLATFORM_ENV, "").strip().lower()
    if platform in ("rocm", "cuda"):
        return platform == "rocm"
    if platform:
        raise ValueError(f"{PLATFORM_ENV} must be 'rocm' or 'cuda', not {os.environ[PLATFORM_ENV]!r}")
    return os.path.exists(ROCM_DEVICE_NODE) and not os.path.exists(NVIDIA_DEVICE_NODE)


def _device_indices(value: str) -> Optional[list[int]]:
    """Return the indices in a device list, or None if a token is not an index."""
    tokens = [token.strip() for token in value.split(",")] if value.strip() else []
    if not all(_DEVICE_INDEX.fullmatch(token) for token in tokens):
        return None
    return [int(token) for token in tokens]


def normalize_rocm_visible_devices(env: Optional[MutableMapping[str, str]] = None) -> None:
    """Keep the GPU selection of the HIP runtime in HIP_VISIBLE_DEVICES only.

    The rules follow the HIP runtime:
    - A non-empty HIP_VISIBLE_DEVICES, else CUDA_VISIBLE_DEVICES, selects
      positions in the ROCR_VISIBLE_DEVICES list.
    - A relative list that is equal to the ROCR list (Slurm sets both) selects
      the ROCR list.
    - The selection stops at the first position that is out of range.
    A token that is not an index (for example a UUID) stops the change.
    """
    env = os.environ if env is None else env
    rocr = env.get("ROCR_VISIBLE_DEVICES")
    cuda = env.get("CUDA_VISIBLE_DEVICES")
    if (rocr is None and cuda is None) or not is_rocm_host():
        return
    relative = env.get("HIP_VISIBLE_DEVICES") or cuda or None
    base = None if rocr is None else _device_indices(rocr)
    chosen = None if relative is None else _device_indices(relative)
    if (rocr is not None and base is None) or (relative is not None and chosen is None):
        return
    if chosen is None or chosen == base:
        selection = base
    elif base is None:
        selection = chosen
    else:
        selection = []
        for position in chosen:
            if position >= len(base):
                break
            selection.append(base[position])
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if selection is not None:
        env["HIP_VISIBLE_DEVICES"] = ",".join(map(str, dict.fromkeys(selection))) or NO_DEVICE
