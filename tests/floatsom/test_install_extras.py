"""Install extras, driver install hints and PyPI package metadata.

The pyproject tests read ``pyproject.toml``. The driver tests put a fake
``nvidia-smi`` on PATH: the NVIDIA driver is an external dependency, and the
floatsom code that reads its output runs unchanged.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
EXTRAS = PROJECT["optional-dependencies"]
BLOCK_CUPY_AND_RAY = "import sys\n" + "".join(
    f"sys.modules[{name!r}] = None\n" for name in ("cupy", "cupyx", "ray")
)


def requirements(extra):
    return {Requirement(item).name: Requirement(item) for item in EXTRAS[extra]}


@pytest.mark.parametrize("cuda", ["12", "13"])
def test_gpu_extra_installs_cupy_14_with_the_cuda_libraries(cuda):
    cupy = requirements(f"cuda{cuda}")[f"cupy-cuda{cuda}x"]
    assert cupy.extras == {"ctk"}
    assert cupy.specifier.contains("14.0.0")
    assert cupy.specifier.contains("14.9.0")
    assert not cupy.specifier.contains("13.6.0")
    assert not cupy.specifier.contains("15.0.0")


@pytest.mark.parametrize("cuda", ["12", "13"])
def test_gpu_extra_installs_no_nccl_and_no_ray(cuda):
    assert set(requirements(f"cuda{cuda}")) == {f"cupy-cuda{cuda}x"}


@pytest.mark.parametrize("cuda", ["12", "13"])
def test_ray_gpu_extra_adds_ray_and_nccl(cuda):
    extra = requirements(f"ray-cuda{cuda}")
    assert extra["floatsom"].extras == {f"cuda{cuda}", "ray"}
    assert f"nvidia-nccl-cu{cuda}" in extra


def test_ray_extra_installs_only_ray():
    assert set(requirements("ray")) == {"ray"}


def test_ray_extra_excludes_ray_that_needs_torch_for_nccl():
    # Ray 2.56 and later find the NCCL backend only when PyTorch imports.
    # FloatSOM uses CuPy for NCCL, so these versions fail with no PyTorch.
    ray = requirements("ray")["ray"]
    assert ray.specifier.contains("2.47.1")
    assert ray.specifier.contains("2.55.1")
    assert not ray.specifier.contains("2.56.0")


def test_networkx_is_a_core_dependency():
    core = {Requirement(item).name: Requirement(item) for item in PROJECT["dependencies"]}
    assert "networkx" in core
    assert core["networkx"].specifier.contains("3.4")
    assert not core["networkx"].specifier.contains("3.3")


def test_no_empty_or_rocm_extra():
    assert "rocm" not in EXTRAS
    assert all(EXTRAS.values())


def test_install_hints_name_the_new_extras():
    from floatsom import _optional_dependencies as hints

    assert "floatsom[rocm]" not in hints.CUPY_INSTALL_HINT
    assert "floatsom[ray-cuda12]" in hints.RAY_INSTALL_HINT
    assert "floatsom[ray-cuda13]" in hints.RAY_INSTALL_HINT


def test_package_metadata_is_ready_for_pypi():
    assert PROJECT["license"] == "MIT"
    assert PROJECT["license-files"] == ["LICENSE"]
    assert "MIT License" in (REPO_ROOT / "LICENSE").read_text()
    assert "Environment :: GPU :: NVIDIA CUDA" in PROJECT["classifiers"]
    assert "Programming Language :: Python :: 3" in PROJECT["classifiers"]
    assert PROJECT["description"] != "FloatSOM core library"


def fake_nvidia_smi(directory, cuda_version, compute_caps=("8.0",)):
    """Write an nvidia-smi script that acts as the real tool.

    With no argument it prints the header. With
    ``--query-gpu=compute_cap --format=csv,noheader`` it prints one compute
    capability for each GPU.
    """
    script = directory / "nvidia-smi"
    caps = "".join(f"echo '{cap}'\n" for cap in compute_caps)
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--query-gpu=compute_cap" ] && [ "$2" = "--format=csv,noheader" ]; then\n'
        f"{caps}"
        "exit 0\n"
        "fi\n"
        "echo '+-----------------------------------------------------------------------------+'\n"
        f"echo '| NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: {cuda_version}  |'\n"
    )
    script.chmod(0o755)


def environment_report(path_dir):
    """Run check_environment with no CuPy and no Ray, and PATH set to path_dir."""
    code = BLOCK_CUPY_AND_RAY + textwrap.dedent(
        """
        import json
        import floatsom
        print(json.dumps(floatsom.check_environment(print_report=False)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": str(path_dir)},
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "cuda_version, extra, other_extra",
    [
        ("12.4", "cuda12", "cuda13"),
        ("13.0", "cuda13", "cuda12"),
        ("13.2", "cuda13", "cuda12"),
        # A newer driver runs the CUDA 13 libraries.
        ("14.0", "cuda13", "cuda12"),
    ],
)
def test_check_environment_gives_the_install_command_for_the_driver(
    tmp_path, cuda_version, extra, other_extra
):
    fake_nvidia_smi(tmp_path, cuda_version)
    report = environment_report(tmp_path)
    assert report["driver_cuda"] == cuda_version
    commands = [problem for problem in report["problems"] if "pip install" in problem]
    assert commands, report["problems"]
    assert all(f"floatsom[{extra}]" in problem for problem in commands)
    assert not any(f"floatsom[{other_extra}]" in problem for problem in commands)
    assert any(f"CUDA {cuda_version}" in problem for problem in report["problems"])


@pytest.mark.parametrize("compute_caps", [("7.0",), ("7.0", "8.0"), ("6.0",)])
def test_check_environment_gives_cuda12_for_a_gpu_that_cuda13_does_not_support(
    tmp_path, compute_caps
):
    # CUDA 13 has no support for compute capability below 7.5 (for example
    # V100 7.0 or P100 6.0), also with a driver for CUDA 13.
    fake_nvidia_smi(tmp_path, "13.0", compute_caps)
    report = environment_report(tmp_path)
    commands = [problem for problem in report["problems"] if "pip install" in problem]
    assert commands, report["problems"]
    assert all("floatsom[cuda12]" in problem for problem in commands)
    assert not any("floatsom[cuda13]" in problem for problem in commands)
    assert any(min(compute_caps) in problem for problem in commands)


def test_check_environment_says_that_an_old_driver_needs_an_update(tmp_path):
    fake_nvidia_smi(tmp_path, "11.8")
    report = environment_report(tmp_path)
    assert report["driver_cuda"] == "11.8"
    assert any("11.8" in problem and "update" in problem.lower() for problem in report["problems"])
    assert not any("floatsom[cuda" in problem for problem in report["problems"])


def test_check_environment_does_not_wait_long_for_a_hung_nvidia_smi(tmp_path):
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!/bin/sh\nexec {shutil.which('sleep')} 120\n")
    script.chmod(0o755)
    start = time.monotonic()
    report = environment_report(tmp_path)
    assert time.monotonic() - start < 25
    assert report["driver_cuda"] is None


def test_check_environment_without_nvidia_smi_gives_the_general_hint(tmp_path):
    report = environment_report(tmp_path)
    assert report["driver_cuda"] is None
    assert any(
        "floatsom[cuda12]" in problem and "floatsom[cuda13]" in problem
        for problem in report["problems"]
    )
