"""Check that the machine can run FloatSOM, and say how to fix each problem."""

import importlib.metadata
import platform
from typing import Any, Dict

from ._optional_dependencies import GIT_SOURCE, require_cupy

_RAY_NOTE = f'needed only for multi-GPU training: pip install "floatsom[ray] @ {GIT_SOURCE}"'


def check_environment(print_report: bool = True) -> Dict[str, Any]:
    """Return (and by default print) the CuPy, CUDA, GPU and Ray status.

    ``report["problems"]`` lists each problem that stops GPU training, with
    the step to fix it. An empty list means that FloatSOM can train here.
    """
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "floatsom": _installed_version("floatsom"),
        "cupy": None,
        "cuda_runtime": None,
        "gpus": [],
        "ray": None,
        "problems": [],
    }
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
        report["problems"].append(str(error))
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


def format_report(report: Dict[str, Any]) -> str:
    """Return the report as text for a terminal."""
    gpus = ", ".join(
        f"[{gpu['index']}] {gpu['name']} ({gpu['memory_mib']:,} MiB)" for gpu in report["gpus"]
    )
    lines = [
        "FloatSOM environment",
        f"  Python    {report['python']}",
        f"  floatsom  {report['floatsom'] or 'not installed as a package'}",
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
