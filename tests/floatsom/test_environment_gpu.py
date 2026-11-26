"""check_environment on a GPU host: "ready" must mean that CuPy kernels run.

This file does not use GPU_AVAILABLE from conftest.py. That check runs a
kernel, so it skips the hosts where a kernel fails, for example CuPy for
CUDA 13 on a V100 (CUDA 13 has no support for compute capability below 7.5).
"""

import pytest

cp = pytest.importorskip("cupy")

try:
    GPU_COUNT = cp.cuda.runtime.getDeviceCount()
except Exception:  # No driver or no GPU.
    GPU_COUNT = 0

pytestmark = pytest.mark.skipif(GPU_COUNT == 0, reason="GPU required")


def kernel_runs():
    """Return True when NVRTC compiles a kernel and the GPU runs it."""
    try:
        kernel = cp.RawKernel(
            r'extern "C" __global__ void add_one(float* a) { a[threadIdx.x] += 1.0f; }',
            "add_one",
        )
        values = cp.zeros(4, dtype=cp.float32)
        kernel((1,), (4,), (values,))
        cp.cuda.Device().synchronize()
        return float(values.sum()) == 4.0
    except Exception:
        return False


def lowest_compute_capability():
    return min(
        int(cp.cuda.runtime.getDeviceProperties(index)["major"]) * 10
        + int(cp.cuda.runtime.getDeviceProperties(index)["minor"])
        for index in range(GPU_COUNT)
    )


def test_ready_report_means_that_kernels_run():
    from floatsom import check_environment

    report = check_environment(print_report=False)
    assert (not report["problems"]) == kernel_runs(), report["problems"]


def test_cuda13_cupy_on_an_old_gpu_names_the_cuda12_extra():
    if cp.cuda.runtime.runtimeGetVersion() < 13000 or lowest_compute_capability() >= 75:
        pytest.skip("needs CuPy for CUDA 13 and a GPU with compute capability below 7.5")
    from floatsom import check_environment

    problems = check_environment(print_report=False)["problems"]
    assert any("floatsom[cuda12]" in problem for problem in problems), problems
