# FloatSOM

FloatSOM trains self-organizing maps (SOMs) on a GPU. Give it a table of
numbers, for example cytometry markers, and it puts each row (cell) on a node
of a map. Similar rows go to the same node or to nodes near each other.

## Install

FloatSOM needs Python 3.10 or later and a GPU.

1. Find the CUDA version of your GPU driver. Run `nvidia-smi` and read
   "CUDA Version" in the top line. 12.x needs the `cuda12` extra. 13.x
   needs the `cuda13` extra.
2. Install FloatSOM with that extra:

   ```bash
   pip install "floatsom[cuda12] @ git+https://github.com/1ordinateur/floatsom.git"
   ```

3. Check the setup:

   ```bash
   python -m floatsom
   ```

   This command shows the CuPy, CUDA, GPU and Ray status. If a problem stops
   GPU training, it tells you how to fix the problem.

Other cases:

- Several GPUs or nodes (Ray): use the extras `[cuda12,ray]`. One GPU does
  not need Ray.
- AMD GPU (ROCm): install CuPy for your ROCm version, then use the `[rocm]`
  extra.
- From a clone: `git clone https://github.com/1ordinateur/floatsom.git`, then
  `pip install "./floatsom[cuda12]"`.

## Quick start

```python
import pandas as pd
import floatsom

cells = pd.read_csv("cells.csv")
markers = ["CD3", "CD4", "CD8", "CD19"]        # numeric columns, already transformed

som = floatsom.train_som(cells[markers], seed=42)
cells["node"] = som.predict(cells[markers])    # SOM node of each cell
som.save("cells.floatsom")
```

Use the model again on new data:

```python
som = floatsom.FloatSOM.load("cells.floatsom")
new_cells["node"] = som.predict(new_cells[som.feature_names])
```

Some important points:

- `train_som` accepts a pandas DataFrame, a NumPy array, a CuPy array or a
  nested list. The model keeps the column names of a DataFrame, and
  `predict` checks them.
- `predict` gives a NumPy array for NumPy, pandas or list input, and a CuPy
  array for CuPy input.
- `predict` gives the SOM node of each row, not a final cluster. To get a
  small number of clusters (metaclusters), see the
  [user guide](docs/user_guide.md#metaclusters).

## Examples

| Example | What it shows |
| --- | --- |
| [`examples/quickstart.py`](examples/quickstart.py) | Train, predict, save and load, with a NumPy array |
| [`examples/dataframe_labels.py`](examples/dataframe_labels.py) | CSV file in, node and metacluster columns out |
| [`examples/large_file.py`](examples/large_file.py) | Label a large Zarr file in chunks, and write the labels to Zarr |

Run an example with `python examples/quickstart.py`.

## Documentation

| Page | Contents |
| --- | --- |
| [User guide](docs/user_guide.md) | Data preparation, `train_som` options, results, metaclusters, save and load, large files |
| [Configuration](docs/configuration.md) | All settings, with advice on when to change them; settings in JSON |
| [Troubleshooting](docs/troubleshooting.md) | Error messages and their fixes |
| [Tuned profiles](docs/defaults.md) | The sets of tuned default values |
| [Ray execution](docs/ray_execution.md) | Training on several GPUs or nodes |

## Citation

Please cite **FloatSOM: GPU Accelerated, Distributed, Topology-Flexible
Self-Organizing Maps**, by Tony Xu, Sarah Klamt, Katharine Turner, Anne Brüstle,
Felix Marsh-Wakefield and Givanna Putri. The manuscript and benchmarks are in the
[publication repository](https://github.com/1ordinateur/floatsom-publication).

## License

This project is licensed under the MIT License.
