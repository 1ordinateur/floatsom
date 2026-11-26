"""Check that the machine can run FloatSOM, and say how to fix each problem."""

import importlib.metadata
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional

from ._optional_dependencies import GIT_SOURCE, require_cupy

_RAY_NOTE = (
    f'needed only for multi-GPU training: pip install "floatsom[ray-cuda12] @ {GIT_SOURCE}"; '
    "ray-cuda13 for CUDA 13.x; floatsom[ray] with a cluster module"
)

# CUDA 13 has no support for GPUs with a compute capability below 7.5
# (Maxwell, Pascal and Volta, for example P100 and V100).
_CUDA13_MIN_COMPUTE_CAPABILITY = 7.5

_TEST_KERNEL = r'extern "C" __global__ void floatsom_check(float* a) { a[threadIdx.x] += 1.0f; }'


def check_environment(print_report: bool = True) -> Dict[str, Any]:
    """Return (and by default print) the CuPy, CUDA, GPU and Ray status.

    ``report["problems"]`` lists each problem that stops GPU training, with
    the step to fix it. An empty list means that FloatSOM can train here.
    """
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "floatsom": _installed_version("floatsom"),
        "driver_cuda": driver_cuda_version(),
        "compute_capabilities": [],
        "cupy": None,
        "cuda_runtime": None,
        "gpus": [],
        "ray": None,
        "problems": [],
    }
    if report["driver_cuda"] is not None:
        report["compute_capabilities"] = gpu_compute_capabilities()
    _check_cupy(report)
    try:
        import ray

        report["ray"] = ray.__version__
    except ImportError:
        pass
    if print_report:
        print(format_report(report))
    return report


def _check_cupy(report: Dict[str, Any]) -> None:
    try:
        cp = require_cupy()
    except ImportError as error:
        report["problems"].append(_cupy_install_problem(report, str(error)))
        return
    report["cupy"] = cp.__version__
    try:
        report["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
        for index in range(int(cp.cuda.runtime.getDeviceCount())):
            properties = cp.cuda.runtime.getDeviceProperties(index)
            name = properties["name"]
            report["gpus"].append({
                "index": index,
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "memory_mib": int(properties["totalGlobalMem"] // 2**20),
                "compute_capability": int(properties["major"]) + int(properties["minor"]) / 10,
            })
    except Exception as error:  # CUDA errors differ between drivers and platforms.
        report["problems"].append(
            f"CuPy {cp.__version__} is installed, but it cannot use a GPU ({error}). "
            "Make sure that the GPU driver works (for example, run nvidia-smi) and that "
            "the CuPy extra (cuda12 or cuda13) matches the CUDA version of the driver."
        )
        return
    if not report["gpus"]:
        report["problems"].append(
            "CuPy found no GPU. On a cluster, run inside a GPU job. "
            "Check CUDA_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES on AMD)."
        )
        return
    _check_kernel(cp, report)


def _check_kernel(cp, report: Dict[str, Any]) -> None:
    """Check that each GPU can run FloatSOM kernels (GPU training compiles them with NVRTC)."""
    lowest = min(gpu["compute_capability"] for gpu in report["gpus"])
    is_hip = bool(getattr(cp.cuda.runtime, "is_hip", False))
    if not is_hip and report["cuda_runtime"] >= 13000 and lowest < _CUDA13_MIN_COMPUTE_CAPABILITY:
        report["problems"].append(
            f"CuPy {report['cupy']} is for CUDA 13, but CUDA 13 has no support for a GPU with "
            f"compute capability {lowest:.1f} (it needs {_CUDA13_MIN_COMPUTE_CAPABILITY} or higher). "
            "Make a new virtual environment, then install with: "
            f'pip install "floatsom[cuda12] @ {GIT_SOURCE}" '
            "(floatsom[ray-cuda12] for several GPUs)."
        )
        return
    for gpu in report["gpus"]:
        error = _kernel_error(cp, gpu["index"])
        if error is not None:
            report["problems"].append(
                f"CuPy {report['cupy']} cannot compile or run a kernel on GPU {gpu['index']} "
                f"({error}). Make sure that the CuPy extra (cuda12 or cuda13) agrees with the "
                "driver and the GPU. Run nvidia-smi to see the driver CUDA version."
            )
            return


def _kernel_error(cp, index: int) -> Optional[str]:
    """Run a small kernel on GPU ``index``. Return None, or the error as text."""
    try:
        with cp.cuda.Device(index):
            kernel = cp.RawKernel(_TEST_KERNEL, "floatsom_check")
            values = cp.zeros(4, dtype=cp.float32)
            kernel((1,), (4,), (values,))
            cp.cuda.Device(index).synchronize()
            if float(values.sum()) == 4.0:
                return None
        return "the kernel gave a wrong result"
    except Exception as exc:  # NVRTC and CUDA errors differ between versions.
        return str(exc)


def _run_nvidia_smi(arguments: List[str]) -> Optional[str]:
    """Return the output of nvidia-smi, or None when it fails or hangs."""
    try:
        process = subprocess.Popen(
            ["nvidia-smi", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    try:
        output, _ = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        # A hung driver can keep nvidia-smi alive after kill(), so wait only briefly.
        process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        return None
    return output if process.returncode == 0 else None


def driver_cuda_version() -> Optional[str]:
    """Return the "CUDA Version" that nvidia-smi shows, or None with no NVIDIA driver."""
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", _run_nvidia_smi([]) or "")
    return match.group(1) if match else None


def gpu_compute_capabilities() -> List[str]:
    """Return the compute capability of each GPU (for example "7.0"), from nvidia-smi."""
    output = _run_nvidia_smi(["--query-gpu=compute_cap", "--format=csv,noheader"]) or ""
    return re.findall(r"^\s*(\d+\.\d+)\s*$", output, flags=re.MULTILINE)


def _cupy_install_problem(report: Dict[str, Any], general_hint: str) -> str:
    """Return the fix for a missing CuPy, for the driver and the GPUs of this machine."""
    driver_cuda = report["driver_cuda"]
    if driver_cuda is None:
        return general_hint
    major = int(driver_cuda.split(".")[0])
    if major < 12:
        return (
            f"CuPy is not installed, and the GPU driver supports only CUDA {driver_cuda}. "
            "FloatSOM needs a driver for CUDA 12 or later. Update the GPU driver, "
            "then run 'python -m floatsom' again."
        )
    capabilities = report["compute_capabilities"]
    old_gpu = min(capabilities, key=float) if capabilities else None
    if major == 12:
        reason = f"The GPU driver supports CUDA {driver_cuda}."
        extra = "cuda12"
    elif old_gpu is not None and float(old_gpu) < _CUDA13_MIN_COMPUTE_CAPABILITY:
        reason = (
            f"The GPU driver supports CUDA {driver_cuda}, but CUDA 13 has no support for "
            f"a GPU with compute capability {old_gpu}."
        )
        extra = "cuda12"
    else:
        # A driver for a newer CUDA (14 or later) runs the CUDA 13 libraries.
        reason = f"The GPU driver supports CUDA {driver_cuda}."
        extra = "cuda13"
    return (
        f"CuPy is not installed. {reason} Install with: "
        f'pip install "floatsom[{extra}] @ {GIT_SOURCE}". '
        "Then run 'python -m floatsom' again."
    )


def format_report(report: Dict[str, Any]) -> str:
    """Return the report as text for a terminal."""
    gpus = ", ".join(
        f"[{gpu['index']}] {gpu['name']} ({gpu['memory_mib']:,} MiB)" for gpu in report["gpus"]
    )
    lines = [
        "FloatSOM environment",
        f"  Python    {report['python']}",
        f"  floatsom  {report['floatsom'] or 'not installed as a package'}",
        f"  Driver    CUDA {report['driver_cuda'] or '- (no CUDA Version from nvidia-smi)'}",
        f"  CuPy      {report['cupy'] or 'not installed'}",
        f"  CUDA      {report['cuda_runtime'] or '-'}",
        f"  GPUs      {gpus or 'none'}",
        f"  Ray       {report['ray'] or 'not installed'} ({_RAY_NOTE})",
    ]
    if report["problems"]:
        lines.append("Problems:")
        lines.extend(f"  - {problem}" for problem in report["problems"])
        lines.append("Result: not ready for GPU training.")
    else:
        lines.append("Result: ready for GPU training.")
    return "\n".join(lines)


def _installed_version(distribution: str):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
