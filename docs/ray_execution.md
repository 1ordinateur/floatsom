# Ray execution

Set `ProcessingConfig(method="batch", chunk_size=..., ray_config=RayConfig(...))`
to select `RayBatchProcessor`. Omitting `ray_config` selects local GPU batch
processing. `RayConfig` is defined in `floatsom.floatsom_params`; its `chunk_size`
must also be specified.

The Ray cluster must expose GPU resources. For multiple nodes, start the cluster
using the scheduler and networking setup for your HPC system, then connect the
training process to it. Do not rely on a local Ray instance to discover other
allocated nodes automatically.

- `storage_path` is the shared staging location accessible to the workers.
- `local_storage_path` is the worker-local staging/cache location. Provision enough
  space for each worker's shards; it is distinct from the shared dataset store.
- Choose GPU count and chunk sizes to fit the allocation and available VRAM.
  Chunk sizes, buffering and staging requirements depend on feature count and
  the selected processing mode.
- NCCL is used for GPU collective operations. Install the CUDA extra matching
  the cluster's toolkit, and ensure its network and GPU transport requirements
  are satisfied on every node.

Full-batch parity, random selection and color processing have separate test
coverage. Results for one processing mode should not be assumed to establish
parity for every configuration. The publication's execution-path concordance
analysis reports the evaluated full-sampling settings.

Run the distributed tests with a GPU-enabled environment:

```bash
python -m pytest tests/floatsom/test_ray_processor_parity.py tests/floatsom/test_gpu_scaling.py
```

Single- versus multi-GPU comparisons require at least two usable GPUs. The test
fixtures release a Ray instance they start themselves, while preserving an
instance initialized by the caller.
