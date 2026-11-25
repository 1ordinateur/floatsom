# User guide

This guide shows the usual work with FloatSOM: prepare the data, train a
SOM, use the results, and save the model. Install FloatSOM first, as the
[README](../README.md#install) shows.

## Prepare the data

FloatSOM needs a table of numbers: one row for each sample (for example a
cell) and one column for each feature (for example a marker).

- Select the numeric columns that you want to use. A DataFrame column that is
  not numeric gives an error that names the column.
- Remove or impute rows with NaN or infinite values. These rows give an error
  that counts them.
- Transform and scale the data before training, as for any SOM. For
  cytometry data, an arcsinh transform is usual. When the features have
  very different ranges, scale each feature, for example to zero mean and
  unit variance. FloatSOM does not do this for you.

```python
import numpy as np
import pandas as pd

cells = pd.read_csv("cells.csv")
markers = ["CD3", "CD4", "CD8", "CD19"]
features = np.arcsinh(cells[markers] / 5)    # cofactor 5; use the value for your data
```

FloatSOM does not read files in `train_som`. Read a file with pandas or
NumPy first, for example `pd.read_csv(path)` or `np.load(path)`.

## Train a SOM

```python
import floatsom

som = floatsom.train_som(features, seed=42)
```

`train_som` uses one GPU, full sampling and full-batch training. It selects
the chunk size from the GPU memory. Its options are:

| Option | Default | Values and advice |
| --- | --- | --- |
| `grid_size` | `suggest_grid_size(n_samples)` | Side of the map; the map has `grid_size ** 2` nodes. The default is about 5·√n nodes, with a side from 2 to 32. For a map with 100 nodes, as in FlowSOM, set `grid_size=10`. |
| `topology` | `"hexagonal"` | `"hexagonal"`, `"grid"`, `"mst"`, `"rng"`. See [configuration](configuration.md#topology). |
| `iterations` | `100` | Positive integer. More iterations take more time. |
| `distance_metric` | `"euclidean"` | `"euclidean"`, `"cosine"`, `"manhattan"`, `"norm_p"` (Lp distance, p = 3) |
| `profile` | `"library"` | A set of tuned default values; see [tuned profiles](defaults.md). |
| `seed` | `None` | An integer gives results that you can repeat. |
| `verbose` | `False` | `True` shows the training progress. |

For settings that `train_som` does not have (subsampling, data in files, more
GPUs), see [configuration](configuration.md).

## Use the results

```python
cells["node"] = som.predict(features)          # SOM node index of each row
nodes, distances = som.map_vectors(features)   # node index and distance to the node
weights = som.get_weights()                    # node weights, shape (n_nodes, n_features)
```

- `predict` and `map_vectors` give NumPy arrays for NumPy, pandas or list
  input, and CuPy arrays for CuPy input.
- The node index goes from 0 to `n_nodes - 1`.
- Prediction uses the distance metric of the training.

### Column names

When you train with a DataFrame, the model keeps its column names in
`som.feature_names`. `predict`, `map_vectors` and `infer` then check the
columns of each DataFrame:

- Columns in a different order give an error. Select the columns in the
  training order with `data[som.feature_names]`.
- A missing or different column gives an error that names it.
- A NumPy or CuPy array, or a file, has no column names that FloatSOM
  checks. FloatSOM checks only the number of columns, so give the columns in
  the training order.

## Metaclusters

A SOM has many nodes, for example 100 to 1024. To get a small number of
clusters, cluster the node weights, then give each row the cluster of its
node. This example uses scikit-learn:

```python
from sklearn.cluster import AgglomerativeClustering

nodes = som.predict(features)
node_metacluster = AgglomerativeClustering(n_clusters=20).fit_predict(som.get_weights())
cells["metacluster"] = node_metacluster[nodes]
```

You can use a different clustering method or a different number of clusters.
FloatSOM does not save the metaclusters. Save `node_metacluster` yourself,
for example with `np.save`.

## Save and load

```python
som.save("cells.floatsom")
som = floatsom.FloatSOM.load("cells.floatsom")
```

The model file is one `.npz` file with the weights, the settings, the column
names and the grid reform. It does not use pickle. A loaded SOM uses one GPU,
also if it trained with Ray.

## Large data

`infer` labels the rows in chunks, so the GPU does not need to hold all the
data. It accepts the same inputs as `predict`, and also a file path:

```python
labels = som.infer("cells.zarr")                             # NumPy array of node indices
som.infer("cells.zarr", output_path="labels.zarr")          # writes the labels to Zarr
```

To read the labels back from Zarr:

```python
import zarr

labels = zarr.open_group("labels.zarr", mode="r")["assignments"][:]
```

The file formats that `infer` and `FloatSOM.train` read are:

| Format | Path | Memory |
| --- | --- | --- |
| Zarr (v3) | a `.zarr` folder with one 2-D array | reads chunks from disk |
| FastArrayStore | a folder with `array_metadata.json` | reads chunks from disk |
| NumPy | `.npy`, or `.npz` with one array | loads all the data into RAM |
| Parquet | `.parquet` with numeric columns only; needs `pip install pyarrow` | loads all the data into RAM |
| CSV | `.csv` or `.txt`, numbers only, comma between values, no header row | loads all the data into RAM |

For data that is larger than the RAM, use Zarr. To change a CSV file to
Zarr:

```python
import numpy as np
import pandas as pd
import zarr

cells = pd.read_csv("cells.csv", usecols=markers)
zarr.save_array("cells.zarr", cells[markers].to_numpy(np.float32))
```

## Progress and repeatable results

- `verbose=True` in `train_som`, or `som.infer(..., verbose=True)`, shows the
  progress of that call. If your program does not configure Python logging,
  FloatSOM adds a log handler for that call only.
- `train_som(..., seed=42)` makes the random parts (initialization and
  sampling) the same in each run. Use the same data, settings and GPU type
  to get repeatable results.
