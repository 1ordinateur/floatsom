# Troubleshooting

Run `python -m floatsom` first. It checks CuPy, CUDA, the GPU and Ray, and it
tells you how to fix a problem that it finds.

## Install and GPU

| Message or problem | Fix |
| --- | --- |
| `ImportError: FloatSOM needs CuPy ...` | Run `python -m floatsom`. It gives the install command for your driver. Or install FloatSOM with the `cuda12` or `cuda13` extra, as the [README](../README.md#install) shows. On AMD, install CuPy for ROCm first. |
| `... the GPU driver supports only CUDA 11.x` | Update the GPU driver. FloatSOM needs a driver for CUDA 12 or later. |
| `CuPy ... cannot use a GPU` | Make sure that the driver works (`nvidia-smi`). Make sure that the extra agrees with "CUDA Version" in `nvidia-smi`: 12.x needs `cuda12`, 13.x needs `cuda13`. |
| `CuPy found no GPU` | On a cluster, run inside a GPU job. Check `CUDA_VISIBLE_DEVICES` (or `ROCR_VISIBLE_DEVICES` on AMD). |
| `Multi-GPU processing ... needs Ray` | Install the `ray-cuda12` or `ray-cuda13` extra (Ray, NCCL and PyTorch), or set `ray_config=None` for one GPU. With a cluster module that gives CuPy, NCCL and PyTorch, install the `ray` extra. |
| `Ray found no NCCL backend ...` | Ray 2.50 and later need PyTorch for NCCL. Install the `ray-cuda12` or `ray-cuda13` extra, or install PyTorch. |
| `CuPy ... for CUDA 13 cannot run a kernel on this GPU` | CUDA 13 has no support for GPUs below compute capability 7.5 (for example V100). Remove `cupy-cuda13x` and install the `cuda12` extra. |
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
| `verbose=True` shows nothing | A log handler in your program has a level above INFO. Set the level of that handler to INFO. |
| Too many clusters | `predict` gives SOM nodes. Make metaclusters from the nodes; see [metaclusters](user_guide.md#metaclusters). |
| `reform_grid=True` is slow on AMD | RAPIDS has no AMD build, so FloatSOM uses a CPU fallback (scikit-learn PCA and the networkx ForceAtlas2 layout). The time and memory increase as the square of the node count: about 40 s for 1024 nodes, and about 1 h and 8 GB of RAM for 10,000 nodes. |
