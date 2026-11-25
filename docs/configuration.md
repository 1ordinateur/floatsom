# Configuration

`train_som` is sufficient for most work. Use the full configuration only when
you need a setting that `train_som` does not have, for example subsampling,
data in files, or more GPUs.

## Build and train

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
    topology_config=TopologyConfig(topology_type="hexagonal", grid_size=20),
)
som = create_floatsom(params)
stats = som.train(data)    # data: an array, or a path to a file (see the user guide)
```

`train` changes `som` and returns a dict with the training statistics. After
training, use `som` as the [user guide](user_guide.md#use-the-results) shows.
With `ProcessingConfig.method="minisom"`, `create_floatsom` gives a MiniSOM
adapter. It has `train`, `predict` and `get_weights`, but no `map_vectors`,
`infer`, `save` or column-name check.

Each setting has a default, so give only the settings that you change. The
exception is `RayConfig`, which needs `chunk_size`, `storage_path` and
`local_storage_path`; see [Ray execution](ray_execution.md).

## Main choices

Keep the default unless you have a reason to change it.

### Sampling

| `SamplingConfig.method` | Use when |
| --- | --- |
| `"full"` (default) | Most data. Each iteration uses all the rows. |
| `"random"` | Very large data. Each iteration uses a random part of the rows; set the part with `target_proportion`. |
| `"hdsssom"` | Advanced. Dynamic subset selection (Wetmore et al., 2005), which selects the rows with large errors. |

### Processing

| Setting | Use when |
| --- | --- |
| `ProcessingConfig.method="batch"` (default) | Most work: batch SOM training on the GPU. |
| `ProcessingConfig.method="colors"` | Advanced. Colour-set processing, an alternative update method. |
| `ProcessingConfig.method="minisom"` | Comparison with MiniSOM. Needs the `minisom` extra. |
| `ProcessingConfig.batch_mode="full_batch"` (default) | One update with all the rows of each iteration. |
| `ProcessingConfig.batch_mode="minibatch"` | An update after each chunk of `chunk_size` rows. |
| `ProcessingConfig.chunk_size` | CUDA out-of-memory errors: make it smaller. `train_som` selects it from the GPU memory. |
| `ProcessingConfig.distance_metric` | `"euclidean"` (default), `"cosine"`, `"manhattan"` or `"norm_p"`. For `"norm_p"`, set p in `distance_metric_params`, for example `{"p": 4}`. p must be a finite number of 1 or more; the default is 3. |

### Topology

| `TopologyConfig.topology_type` | Use when |
| --- | --- |
| `"grid"` | A square map, where each node has four neighbours. |
| `"hexagonal"` (default in `train_som`) | A map where each node has six neighbours. This is the usual SOM map. |
| `"mst"` | A graph topology (minimum spanning tree) that the node weights make during training. |
| `"rng"` | A graph topology (relative neighbourhood graph) that the node weights make during training. |

Set the size with `grid_size` (the side of the map). For `"mst"` and `"rng"`,
you can set `num_nodes` in its place.
`TopologyConfig.topology_variant="toroidal"` joins the opposite edges of a
grid or hexagonal map. The default is `"planar"`.

### Initialization

| `FloatSOMParams.initialization_method` | Use when |
| --- | --- |
| not set (default) | The tuned profile selects the method. |
| `"random"` | Random start weights. |
| `"pca"` | Start weights spread on the first principal components of the data. |
| `"pca_sampling"`, `"pca_sampling_snake"`, `"pca_density"` | Advanced variants of the PCA start. |

### Profiles

`FloatSOMParams.defaults_profile` selects a set of tuned default values. See
[tuned profiles](defaults.md).

## Settings in JSON

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
