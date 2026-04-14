"""
Ray worker manager for final BMU assignment without collective setup.
"""

from __future__ import annotations

from typing import Any, Dict

from .ray_pipeline_base import RayWorkerManager
from .workers.ray_inference_worker import RayInferenceWorker


class RayInferenceWorkerManager(RayWorkerManager):
    """
    Lightweight Ray worker manager for file-backed inference-only workloads.
    """

    def _initialize_collective_group(self):
        return None

    def initialize_inference_workers(self, *, chunk_size: int) -> None:
        if self.workers:
            return
        self.chunk_size = int(chunk_size)
        self.target_chunk_size = int(chunk_size)
        self.loader_chunk_size = int(chunk_size)
        self.sampling_fraction = 1.0
        self.initialize_workers(RayInferenceWorker, int(chunk_size))

    def prepare_fast_array_inference(self, *, source_path: str) -> Dict[str, Any]:
        self.initialize_inference_workers(chunk_size=int(self.chunk_size))

        wipe_enabled = True
        if isinstance(self.ray_config, dict):
            wipe_enabled = bool(self.ray_config.get("wipe_local_storage_on_start", True))
        else:
            wipe_enabled = bool(getattr(self.ray_config, "wipe_local_storage_on_start", True))

        if wipe_enabled:
            wipe_futures = [worker.wipe_local_storage.remote() for worker in self.workers]
            if wipe_futures:
                self.wait_for_worker_futures(
                    wipe_futures,
                    phase="inference_wipe_local_storage",
                )

        configure_futures = [
            worker.configure_chunk_sampling.remote(
                int(self.loader_chunk_size),
                1.0,
                "full",
                whole_chunk_random=False,
            )
            for worker in self.workers
        ]
        if configure_futures:
            self.wait_for_worker_futures(
                configure_futures,
                phase="inference_configure_chunk_sampling",
            )

        staging_details = self._distribute_fast_array_once(
            {
                "type": "file",
                "path": str(source_path),
                "format": "fast_array",
            }
        )
        self.ensure_distribution_ready()
        self.data_distributed = True
        return staging_details
