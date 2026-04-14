"""
Ray worker for final BMU assignment over file-backed FastArrayStore shards.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import cupy as cp
import numpy as np
import ray

from .ray_pipeline_base_worker import RayPipelineBaseWorker
from ...utils import find_bmus

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class RayInferenceWorker(RayPipelineBaseWorker):
    """
    Final-assignment worker that reuses the native RayPipelineBaseWorker loader path.
    """

    def assign_bmus_to_output(
        self,
        *,
        weights: np.ndarray,
        output_path: str,
        node_chunk_size: int = 900,
    ) -> Dict[str, Any]:
        if self.data_loader is None and self.data_path:
            self._initialize_data_loader()
        if self.data_loader is None:
            raise RuntimeError(
                f"Worker {self.worker_id}: data loader not initialized before final inference"
            )

        output = np.lib.format.open_memmap(str(output_path), mode="r+")
        rows_processed = 0
        chunk_reports = 0

        with self.device:
            weights_gpu = cp.asarray(np.asarray(weights, dtype=np.float32))
            self.reset_for_iteration()
            total_chunks = int(self.get_num_chunks())
            global_start_idx = int(getattr(self, "data_start_idx", 0) or 0)
            global_end_idx = int(getattr(self, "data_end_idx", global_start_idx) or global_start_idx)

            try:
                for chunk_index in range(total_chunks):
                    local_samples = self.load_chunk(int(chunk_index))
                    chunk_info = dict(getattr(self, "last_chunk_info", {}) or {})
                    local_size = int(
                        chunk_info.get(
                            "local_size",
                            int(local_samples.shape[0]) if local_samples.ndim > 0 else 0,
                        )
                    )
                    if local_size <= 0:
                        continue

                    chunk_start = global_start_idx + (int(chunk_index) * int(self.loader_chunk_size))
                    chunk_stop = min(chunk_start + local_size, global_end_idx)
                    if chunk_stop <= chunk_start:
                        continue

                    bmus = find_bmus(
                        batch=local_samples,
                        weights=weights_gpu,
                        verbose=False,
                        return_distances=False,
                        node_chunk_size=int(node_chunk_size),
                    )
                    output[chunk_start:chunk_stop] = cp.asnumpy(bmus).astype(np.int32, copy=False)
                    rows_processed += int(chunk_stop - chunk_start)
                    chunk_reports += 1
                    del local_samples, bmus

                output.flush()
            finally:
                del output
                del weights_gpu
                cp.cuda.get_current_stream().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

        return {
            "worker_id": int(self.worker_id),
            "start_idx": global_start_idx,
            "end_idx": global_end_idx,
            "rows_processed": int(rows_processed),
            "chunks_processed": int(chunk_reports),
        }
