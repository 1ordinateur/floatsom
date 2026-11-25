# Troubleshooting

Run `python -m floatsom` first. It checks CuPy, CUDA, the GPU and Ray, and it
tells you how to fix a problem that it finds.

## Install and GPU

| Message or problem | Fix |
| --- | --- |
| `ImportError: FloatSOM needs CuPy ...` | Install FloatSOM with the `cuda12`, `cuda13` or `rocm` extra, as the [README](../README.md#install) shows. Then run `python -m floatsom`. |
| `CuPy ... cannot use a GPU` | Make sure that the driver works (`nvidia-smi`). Make sure that the extra agrees with "CUDA Version" in `nvidia-smi`: 12.x needs `cuda12`, 13.x needs `cuda13`. |
| `CuPy found no GPU` | On a cluster, run inside a GPU job. Check `CUDA_VISIBLE_DEVICES` (or `ROCR_VISIBLE_DEVICES` on AMD). |
| `Multi-GPU processing ... needs Ray` | Install the `ray` extra, or set `ray_config=None` for one GPU. |
| CUDA out of memory | Set a smaller `ProcessingConfig.chunk_size`. `train_som` selects it from the GPU memory. For labels, use `som.infer(data, chunk_size=...)`. |

## Data

| Message or problem | Fix |
| --- | --- |
| `... needs an array or a DataFrame, not a file path` | Read the file first, for example `pd.read_csv(path)`. For a large file, use `som.infer(path)`; see [large data](user_guide.md#large-data). |
| `... needs numeric columns. Remove or encode these columns: ...` | Select only the numeric columns, for example `data[markers]`. |
| `... rows have NaN or infinite values` | Remove or impute these rows before training. |
| `... got data with N features, but the SOM was trained with M features` | Give the same columns as the training data. |
| `... columns in a different order` | Select the columns with `data[som.feature_names]`. |
| `... the SOM was trained with columns ...` and a missing column | Give the data all the training columns, with the same names. |
| `TypeError: Implicit conversion to a NumPy array is not allowed` | You gave CuPy input, so the result is a CuPy array. Use `cupy.asnumpy(result)`, or give NumPy or pandas input. |

## Results and speed

| Message or problem | Fix |
| --- | --- |
| `verbose=True` shows nothing | Configure Python logging at level INFO, for example `logging.basicConfig(level=logging.INFO)`. FloatSOM adds a handler only when your program has none. |
| Too many clusters | `predict` gives SOM nodes. Make metaclusters from the nodes; see [metaclusters](user_guide.md#metaclusters). |
| `reform_grid=True` is slow on AMD | RAPIDS has no AMD build, so FloatSOM uses a CPU fallback (scikit-learn PCA and the networkx ForceAtlas2 layout). The time and memory increase as the square of the node count: about 40 s for 1024 nodes, and about 1 h and 8 GB of RAM for 10,000 nodes. |
