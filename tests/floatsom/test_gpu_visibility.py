"""GPU visibility variables on AMD (ROCm) hosts before the first ``import ray``.

Slurm on an AMD host sets ROCR_VISIBLE_DEVICES. Ray 2.47 raises an error at
``import ray`` when that variable is set, and it asserts that
HIP_VISIBLE_DEVICES equals CUDA_VISIBLE_DEVICES when the CUDA variable is set.
``import floatsom`` changes the variables to the form that Ray and HIP accept.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from floatsom import gpu_visibility

SRC_ROOT = Path(gpu_visibility.__file__).resolve().parents[1]


@pytest.mark.parametrize(("kfd", "nvidia", "expected"), [
    (True, False, True),
    (True, True, False),
    (False, False, False),
    (False, True, False),
])
def test_is_rocm_host_reads_device_nodes(monkeypatch, tmp_path, kfd, nvidia, expected):
    monkeypatch.delenv(gpu_visibility.PLATFORM_ENV, raising=False)
    for flag, name in ((kfd, "ROCM_DEVICE_NODE"), (nvidia, "NVIDIA_DEVICE_NODE")):
        node = tmp_path / name
        if flag:
            node.touch()
        monkeypatch.setattr(gpu_visibility, name, str(node))
    assert gpu_visibility.is_rocm_host() is expected


@pytest.mark.parametrize(("value", "expected"), [("rocm", True), ("cuda", False), (" ROCm ", True)])
def test_is_rocm_host_override(monkeypatch, value, expected):
    monkeypatch.setenv(gpu_visibility.PLATFORM_ENV, value)
    assert gpu_visibility.is_rocm_host() is expected


def test_is_rocm_host_rejects_unknown_override(monkeypatch):
    monkeypatch.setenv(gpu_visibility.PLATFORM_ENV, "metal")
    with pytest.raises(ValueError):
        gpu_visibility.is_rocm_host()


@pytest.mark.parametrize(("before", "after"), [
    ({"ROCR_VISIBLE_DEVICES": "2,3"}, {"HIP_VISIBLE_DEVICES": "2,3"}),
    # Slurm sets the same list in both variables.
    ({"ROCR_VISIBLE_DEVICES": "2,3", "CUDA_VISIBLE_DEVICES": "2,3"}, {"HIP_VISIBLE_DEVICES": "2,3"}),
    ({"ROCR_VISIBLE_DEVICES": "2,3", "HIP_VISIBLE_DEVICES": "1"}, {"HIP_VISIBLE_DEVICES": "3"}),
    ({"ROCR_VISIBLE_DEVICES": "4,5,6", "CUDA_VISIBLE_DEVICES": "0,2"}, {"HIP_VISIBLE_DEVICES": "4,6"}),
    ({"ROCR_VISIBLE_DEVICES": "4,5,6", "HIP_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "0"},
     {"HIP_VISIBLE_DEVICES": "6"}),
    ({"ROCR_VISIBLE_DEVICES": "2,3", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "1"},
     {"HIP_VISIBLE_DEVICES": "3"}),
    ({"CUDA_VISIBLE_DEVICES": "0,2"}, {"HIP_VISIBLE_DEVICES": "0,2"}),
    ({"ROCR_VISIBLE_DEVICES": ""}, {"HIP_VISIBLE_DEVICES": "NoDevFiles"}),
    ({"ROCR_VISIBLE_DEVICES": "2,3", "HIP_VISIBLE_DEVICES": "5"}, {"HIP_VISIBLE_DEVICES": "NoDevFiles"}),
    ({"ROCR_VISIBLE_DEVICES": "2,3,4", "HIP_VISIBLE_DEVICES": "1,7,0"}, {"HIP_VISIBLE_DEVICES": "3"}),
    ({"ROCR_VISIBLE_DEVICES": "2,3", "HIP_VISIBLE_DEVICES": "1,1,0"}, {"HIP_VISIBLE_DEVICES": "3,2"}),
])
def test_normalize_rocm_visible_devices(monkeypatch, before, after):
    monkeypatch.setenv(gpu_visibility.PLATFORM_ENV, "rocm")
    env = dict(before, OTHER="kept")
    gpu_visibility.normalize_rocm_visible_devices(env)
    assert env == dict(after, OTHER="kept")


@pytest.mark.parametrize("before", [
    {"ROCR_VISIBLE_DEVICES": "GPU-0123456789abcdef"},
    {"CUDA_VISIBLE_DEVICES": "GPU-0123456789abcdef"},
    {"ROCR_VISIBLE_DEVICES": "-1"},
    {"HIP_VISIBLE_DEVICES": "1"},
    {},
])
def test_normalize_rocm_visible_devices_leaves_other_values(monkeypatch, before):
    monkeypatch.setenv(gpu_visibility.PLATFORM_ENV, "rocm")
    env = dict(before)
    gpu_visibility.normalize_rocm_visible_devices(env)
    assert env == before


def test_normalize_rocm_visible_devices_does_nothing_on_cuda_host(monkeypatch):
    monkeypatch.setenv(gpu_visibility.PLATFORM_ENV, "cuda")
    env = {"ROCR_VISIBLE_DEVICES": "2,3", "CUDA_VISIBLE_DEVICES": "1"}
    gpu_visibility.normalize_rocm_visible_devices(env)
    assert env == {"ROCR_VISIBLE_DEVICES": "2,3", "CUDA_VISIBLE_DEVICES": "1"}


def test_normalize_rocm_visible_devices_second_call_does_nothing(monkeypatch):
    # PIBLO converts the variables first. The floatsom call then changes nothing.
    monkeypatch.setenv(gpu_visibility.PLATFORM_ENV, "rocm")
    env = {"ROCR_VISIBLE_DEVICES": "4,5,6", "CUDA_VISIBLE_DEVICES": "0,2"}
    gpu_visibility.normalize_rocm_visible_devices(env)
    first = dict(env)
    gpu_visibility.normalize_rocm_visible_devices(env)
    assert env == first == {"HIP_VISIBLE_DEVICES": "4,6"}


FAKE_RAY = textwrap.dedent(
    """
    import os
    # The same import-time checks as the Ray 2.47 AMD GPU manager.
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        assert os.environ.get("HIP_VISIBLE_DEVICES") == os.environ["CUDA_VISIBLE_DEVICES"]
    """
)


def test_import_floatsom_prepares_the_environment_for_ray(tmp_path):
    fake = tmp_path / "fake" / "ray"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(FAKE_RAY)
    env = {key: value for key, value in os.environ.items() if not key.endswith("_VISIBLE_DEVICES")}
    env.update({
        gpu_visibility.PLATFORM_ENV: "rocm",
        "ROCR_VISIBLE_DEVICES": "2,3",
        "CUDA_VISIBLE_DEVICES": "2,3",
        "PYTHONPATH": os.pathsep.join([str(fake.parent), str(SRC_ROOT)]),
    })
    script = (
        "import json, os, floatsom, ray; "
        "print(json.dumps({k: v for k, v in os.environ.items() if k.endswith('_VISIBLE_DEVICES')}))"
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"HIP_VISIBLE_DEVICES": "2,3"}
