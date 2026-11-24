"""The Ray inference worker uses the distance metric that the caller gives."""

import numpy as np
import pytest

from ..conftest import GPU_AVAILABLE, RAY_AVAILABLE

pytestmark = pytest.mark.skipif(
    not (GPU_AVAILABLE and RAY_AVAILABLE), reason="Requires CUDA and Ray"
)

if GPU_AVAILABLE and RAY_AVAILABLE:
    import cupy as cp

    from floatsom.processing.ray_ops.workers.ray_inference_worker import RayInferenceWorker

    from .test_ray_selected_index_processing import _resolve_worker_impl_class

# Sample [9, 0]: the Euclidean BMU is node 1 [10, 1]; the cosine BMU is node 0 [1, 0].
WEIGHTS = np.array([[1.0, 0.0], [10.0, 1.0], [-5.0, -5.0], [-6.0, 5.0]], dtype=np.float32)


def one_chunk_worker():
    """A worker object with one in-memory data chunk in place of the file loader."""
    impl = _resolve_worker_impl_class(RayInferenceWorker)
    worker = impl.__new__(impl)
    worker.worker_id = 0
    worker.data_loader = object()
    worker.data_path = None
    worker.device = cp.cuda.Device(0)
    worker.reset_for_iteration = lambda: None
    worker.get_num_chunks = lambda: 1
    worker.loader_chunk_size = 1
    worker.data_start_idx = 0
    worker.data_end_idx = 1
    worker.last_chunk_info = {"local_size": 1}
    worker.load_chunk = lambda index: cp.asarray([[9.0, 0.0]], dtype=cp.float32)
    return worker


def assign(tmp_path, **kwargs):
    output_path = tmp_path / "assignments.npy"
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.int32, shape=(1,))
    output[:] = -1
    del output
    one_chunk_worker().assign_bmus_to_output(
        weights=WEIGHTS, output_path=str(output_path), node_chunk_size=4, **kwargs
    )
    return int(np.load(output_path)[0])


def test_ray_inference_worker_uses_given_metric(tmp_path):
    assert assign(tmp_path, metric="cosine") == 0


def test_ray_inference_worker_default_metric_stays_euclidean(tmp_path):
    assert assign(tmp_path) == 1
