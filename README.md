# FloatSOM

FloatSOM trains self-organizing maps (SOMs) on a GPU. Give it a table of
numbers, for example cytometry markers, and it puts each row (cell) on a node
of a map. Similar rows go to the same node or to nodes near each other.

## Install

FloatSOM needs Python 3.10 or later, Linux, and an NVIDIA GPU with its
driver. You do not need a CUDA Toolkit or a CUDA module: pip installs the
CUDA libraries.

1. Find the CUDA version of your GPU driver. Run `nvidia-smi` and read
   "CUDA Version" in the top line. 12.x needs the `cuda12` extra. 13.x
   needs the `cuda13` extra, but a GPU older than Turing (compute capability
   below 7.5, for example V100 or P100) needs `cuda12`. A version older than
   12 needs a driver update. `python -m floatsom` tells you the extra.
2. Install FloatSOM with that extra:

   ```bash
   pip install "floatsom[cuda12] @ git+https://github.com/1ordinateur/floatsom.git"
   ```

3. Check the setup:

   ```bash
   python -m floatsom
   ```

   This command shows the driver, CuPy, CUDA, GPU and Ray status. If a
   problem stops GPU training, it tells you how to fix the problem. If CuPy
   is not installed, it gives the install command for your driver.

Other cases:

| Case | Extra | Installs |
| --- | --- | --- |
| One GPU | `cuda12` or `cuda13` | CuPy and the CUDA libraries |
| Several GPUs or nodes (Ray) | `ray-cuda12` or `ray-cuda13` | CuPy, the CUDA libraries, Ray, NCCL and PyTorch (approx. 4 GiB) |
| A cluster module gives CuPy (for example RAPIDS) | none, or `ray` for several GPUs | FloatSOM only, or FloatSOM and Ray |
| AMD GPU (ROCm) | none: install CuPy for your ROCm version first | FloatSOM only |

One GPU does not need Ray. FloatSOM does not use PyTorch, but Ray 2.50 and
later need it for NCCL, so the Ray extras install it. From a clone, run
`git clone https://github.com/1ordinateur/floatsom.git`, then
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
  [user guide](https://github.com/1ordinateur/floatsom/blob/main/docs/user_guide.md#metaclusters).

## Examples

| Example | What it shows |
| --- | --- |
| [`examples/quickstart.py`](https://github.com/1ordinateur/floatsom/blob/main/examples/quickstart.py) | Train, predict, save and load, with a NumPy array |
| [`examples/dataframe_labels.py`](https://github.com/1ordinateur/floatsom/blob/main/examples/dataframe_labels.py) | CSV file in, node and metacluster columns out |
| [`examples/large_file.py`](https://github.com/1ordinateur/floatsom/blob/main/examples/large_file.py) | Label a large Zarr file in chunks, and write the labels to Zarr |

Run an example with `python examples/quickstart.py`.

## Documentation

| Page | Contents |
| --- | --- |
| [User guide](https://github.com/1ordinateur/floatsom/blob/main/docs/user_guide.md) | Data preparation, `train_som` options, results, metaclusters, save and load, large files |
| [Configuration](https://github.com/1ordinateur/floatsom/blob/main/docs/configuration.md) | All settings, with advice on when to change them; settings in JSON |
| [Troubleshooting](https://github.com/1ordinateur/floatsom/blob/main/docs/troubleshooting.md) | Error messages and their fixes |
| [Tuned profiles](https://github.com/1ordinateur/floatsom/blob/main/docs/defaults.md) | The sets of tuned default values |
| [Ray execution](https://github.com/1ordinateur/floatsom/blob/main/docs/ray_execution.md) | Training on several GPUs or nodes |

## Citation

Please cite **FloatSOM: GPU Accelerated, Distributed, Topology-Flexible
Self-Organizing Maps**, by Tony Xu, Sarah Klamt, Katharine Turner, Anne Brüstle,
Felix Marsh-Wakefield and Givanna Putri. The manuscript and benchmarks are in the
[publication repository](https://github.com/1ordinateur/floatsom-publication).

## License

This project is licensed under the MIT License.
