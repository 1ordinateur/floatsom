# Ray execution

Use Ray to train on several GPUs or several nodes. One GPU does not need Ray;
use `train_som` for one GPU.

## Install

Use the `ray-cuda12` or `ray-cuda13` extra. It installs CuPy, the CUDA
libraries, Ray, NCCL and PyTorch:

```bash
pip install "floatsom[ray-cuda12] @ git+https://github.com/1ordinateur/floatsom.git"
```

If a cluster module gives CuPy, NCCL and PyTorch, use the `ray` extra. It
installs only Ray.

FloatSOM does not use PyTorch. Ray 2.50 and later find the NCCL backend only
when PyTorch imports, so the `ray-cuda12` and `ray-cuda13` extras install it.
PyTorch also selects the NCCL version and the CUDA library versions. With
`ray-cuda12`, pip selects PyTorch 2.10 (the newest PyTorch for CUDA 12 on
PyPI). With `ray-cuda13`, pip selects PyTorch 2.11 or later (CUDA 13).

## Train on several GPUs

Give a `RayConfig` to `ProcessingConfig`. FloatSOM then uses Ray for the
batch processing:

```python
from floatsom import (
    FloatSOMParams, ProcessingConfig, RayConfig, TopologyConfig, create_floatsom,
)

ray_config = RayConfig(
    chunk_size=1_000_000,
    storage_path="/shared/floatsom_staging",       # a folder that all the workers can read
    local_storage_path="/local/floatsom_cache",    # a fast folder on each node
    num_gpus=4,                                    # omit to use all visible GPUs
)
params = FloatSOMParams(
    input_dim=40,
    total_iterations=50,
    seed=42,
    processing_config=ProcessingConfig(
        method="batch", chunk_size=1_000_000, ray_config=ray_config,
    ),
    topology_config=TopologyConfig(topology_type="hexagonal", grid_size=20),
)
som = create_floatsom(params)
som.train("cells.zarr")
som.save("cells.floatsom")
```

After training, use `som` as the [user guide](user_guide.md#use-the-results)
shows. A loaded model uses one GPU.

## Settings

- `RayConfig` needs `chunk_size`, `storage_path` and `local_storage_path`.
  Give `ProcessingConfig.chunk_size` too.
- `storage_path` is the shared staging folder that the workers can read.
- `local_storage_path` is the staging and cache folder on each worker. Give
  it enough space for the shards of each worker. It is not the shared data
  store.
- Select the GPU count and the chunk sizes to fit the allocation and the
  GPU memory. The chunk sizes, the buffers and the staging space depend on
  the feature count and the processing mode.
- `ray_address` connects to a Ray cluster that is already running.

## Clusters and several nodes

The Ray cluster must give GPU resources. For several nodes, start the
cluster with the scheduler and network setup of your HPC system. Then
connect the training process to it with `ray_address`. A local Ray instance
does not find the other nodes of your allocation by itself.

NCCL does the GPU collective operations. The `ray-cuda12` and `ray-cuda13`
extras install NCCL. Use the extra that agrees with "CUDA Version" in
`nvidia-smi`. Make sure that each node has the network and GPU transport
that NCCL needs.

For the distributed tests, see the [developer notes](developer_notes.md#tests).
