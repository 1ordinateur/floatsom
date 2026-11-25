# FloatSOM

GPU-accelerated self-organizing maps (SOMs) with flexible sampling, topologies
and local or distributed processing.

## Installation

FloatSOM needs Python 3.10 or later and a GPU. Select the extra that matches
your GPU:

| GPU | Install command |
| --- | --- |
| NVIDIA, CUDA 12 | `pip install '.[cuda12]'` |
| NVIDIA, CUDA 13 | `pip install '.[cuda13]'` |
| AMD (ROCm) | Install CuPy for your ROCm version, then `pip install '.[rocm]'` |

For multi-GPU or multi-node training with Ray, also add the `ray` extra, for
example `pip install '.[cuda12,ray]'`. Single-GPU training does not need Ray.

```bash
git clone https://github.com/1ordinateur/floatsom.git
cd floatsom
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[cuda12]'
```

### Check the setup

```bash
python -m floatsom
```

This command shows the CuPy, CUDA, GPU and Ray status. If a problem stops GPU
training, the command tells you how to fix it and exits with code 1. In
Python, `floatsom.check_environment()` returns the same report as a dict.

## Quick start

```python
import floatsom

som = floatsom.train_som(data)          # data: (n_samples, n_features)
clusters = som.predict(data)            # BMU node index of each sample (CuPy array)
som.save("model.floatsom")

som = floatsom.FloatSOM.load("model.floatsom")
clusters = som.predict(new_data)
```

`train_som` accepts a NumPy array, a CuPy array, a pandas DataFrame with
numeric columns, or a nested list. It checks the data before it uses the GPU,
and it gives a clear error for these problems:

- data that is not 2-D,
- columns that are not numeric (the error names them),
- rows with NaN or infinite values (the error counts them).

Run the complete example:

```bash
python examples/quickstart.py
```

### Options of `train_som`

| Option | Default | Values |
| --- | --- | --- |
| `grid_size` | `suggest_grid_size(n_samples)`: about 5·√n nodes, side from 2 to 32 | integer, 2 or more; the map has `grid_size ** 2` nodes |
| `topology` | `"hexagonal"` | `"hexagonal"`, `"grid"`, `"mst"`, `"rng"` |
| `iterations` | `100` | positive integer |
| `distance_metric` | `"euclidean"` | `"euclidean"`, `"cosine"`, `"manhattan"` |
| `profile` | `"library"` | a name in `floatsom.PROFILE_NAMES`; see [profile selection](docs/defaults.md) |
| `seed` | `None` | integer, for results that you can repeat |
| `verbose` | `False` | `True` logs the training progress |

`train_som` uses full sampling and full-batch processing on one GPU. It
selects `chunk_size` from the GPU memory.

### Use the trained SOM

| Method | Result |
| --- | --- |
| `som.predict(data)` | BMU index of each sample |
| `som.map_vectors(data)` | BMU index and BMU distance of each sample |
| `som.infer(data_or_path, output_path=None)` | BMU indices in chunks, for large arrays or files; writes Zarr if `output_path` is set |
| `som.get_weights()` | Node weights as a NumPy array, shape `(n_nodes, n_features)` |
| `som.save(path)` / `FloatSOM.load(path)` | One `.npz` file with the weights, the settings and the grid reform; no pickle. A loaded SOM uses one GPU, also if it trained with Ray. |

Prediction uses the distance metric of the training. The data must have the
same columns as the training data, in the same order.

## Full configuration

For subsampling, other processing methods, data in files or more GPUs, build
the parameters and call `create_floatsom`:

```python
from floatsom import (
    FloatSOMParams, ProcessingConfig, SamplingConfig, TopologyConfig, create_floatsom,
)

params = FloatSOMParams(
    input_dim=data.shape[1],
    total_iterations=20,
    seed=42,
    sampling_config=SamplingConfig(method="random", target_proportion=0.1),
    processing_config=ProcessingConfig(method="batch", chunk_size=100_000),
    topology_config=TopologyConfig(topology_type="mst", num_nodes=400),
)
som = create_floatsom(params, data=data)
som.train(data)
```

Each setting has a default, so give only the settings that you change. The
exception is `RayConfig`, which needs `chunk_size`, `storage_path` and
`local_storage_path`. The main choices are:

| Setting | Values |
| --- | --- |
| `SamplingConfig.method` | `"full"`, `"random"`, `"hdsssom"` |
| `ProcessingConfig.method` | `"batch"`, `"colors"`, `"minisom"` |
| `ProcessingConfig.batch_mode` | `"full_batch"`, `"minibatch"` |
| `ProcessingConfig.distance_metric` | `"euclidean"`, `"cosine"`, `"manhattan"` |
| `TopologyConfig.topology_type` | `"grid"`, `"hexagonal"`, `"mst"`, `"rng"` |
| `TopologyConfig.topology_variant` | `"planar"`, `"toroidal"` (grid and hexagonal only) |
| `FloatSOMParams.initialization_method` | `"random"`, `"pca"`, `"pca_sampling"`, `"pca_sampling_snake"`, `"pca_density"` |

### Settings in JSON

`FloatSOMParams.to_dict()` gives the settings as a nested, JSON-safe dict.
`FloatSOMParams.from_dict()` builds the parameters from such a dict. Omitted
settings get their defaults. An unknown setting gives an error with the
nearest valid name.

```python
import json
from pathlib import Path
from floatsom import FloatSOMParams

params = FloatSOMParams.from_dict(json.loads(Path("som.json").read_text()))
Path("som_copy.json").write_text(json.dumps(params.to_dict(), indent=2))
```

```json
{
  "input_dim": 40,
  "total_iterations": 50,
  "topology_config": {"topology_type": "hexagonal", "grid_size": 20},
  "processing_config": {"chunk_size": 200000, "distance_metric": "cosine"}
}
```

See [profile selection](docs/defaults.md) and [Ray execution](docs/ray_execution.md)
for more configuration options.

## Troubleshooting

| Message or problem | Fix |
| --- | --- |
| `ImportError: FloatSOM needs CuPy ...` | Install the `cuda12`, `cuda13` or `rocm` extra. Then run `python -m floatsom`. |
| `CuPy ... cannot use a GPU` | Make sure that the driver works (`nvidia-smi`). Make sure that the CuPy extra matches the CUDA version of the driver. |
| `CuPy found no GPU` | On a cluster, run inside a GPU job. Check `CUDA_VISIBLE_DEVICES` (or `ROCR_VISIBLE_DEVICES` on AMD). |
| CUDA out of memory | Set a smaller `ProcessingConfig.chunk_size`. `train_som` selects it from the GPU memory. |
| `... got data with N features, but the SOM was trained with M features` | Give the same columns as the training data, in the same order. |
| `... rows have NaN or infinite values` | Remove or impute these rows before training. |
| `Multi-GPU processing ... needs Ray` | Install the `ray` extra, or set `ray_config=None` for one GPU. |
| `reform_grid=True` is slow on AMD | RAPIDS has no AMD build, so FloatSOM uses a CPU fallback (scikit-learn PCA and the networkx ForceAtlas2 layout). The time and memory increase as the square of the node count: about 40 s for 1024 nodes, and about 1 h and 8 GB of RAM for 10,000 nodes. |

## Citation

Please cite **FloatSOM: GPU Accelerated, Distributed, Topology-Flexible
Self-Organizing Maps**, by Tony Xu, Sarah Klamt, Katharine Turner, Anne Brüstle,
Felix Marsh-Wakefield and Givanna Putri. The manuscript and benchmarks are in the
[publication repository](https://github.com/1ordinateur/floatsom-publication).

## License

This project is licensed under the MIT License.
