# FloatSOM

GPU-accelerated self-organizing maps with full, random and HDSSSOM sampling;
hexagonal, rectangular, minimum-spanning-tree and relative-neighborhood-graph
topologies; and local or distributed Ray processing.

The [publication repository](https://github.com/1ordinateur/floatsom-publication)
contains the manuscript, benchmark code, supporting analysis tables and supplementary
results. This repository contains the standalone library and its tests.

## Installation

Use Python 3.10 or later in a dedicated environment. Training requires an NVIDIA
GPU, a compatible driver and CUDA toolkit. Choose one CUDA extra:

```bash
git clone https://github.com/1ordinateur/floatsom.git
cd floatsom
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[cuda12]'
```

For a CUDA 13 environment, use `'.[cuda13]'` instead. Do not install both extras
in one environment. CUDA 12 is the appropriate path for the V100 hardware used
in the paper: [CUDA 13 removed Volta support](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).
See [CuPy's installation guide](https://docs.cupy.dev/en/stable/install.html) for
toolkit and driver setup. The extras include NCCL on Linux for multi-GPU execution.
Installing without a CUDA extra is sufficient for CPU-only utilities, not SOM training.

Check CUDA before training:

```bash
python -c 'import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(cp.arange(10).sum().item())'
python examples/quickstart.py
```

The [quickstart](examples/quickstart.py) trains a small hexagonal SOM and returns
best-matching-unit assignments. For other configurations, use `FloatSOMParams`,
`SamplingConfig`, `ProcessingConfig`, `TopologyConfig` and `RayConfig` from
`floatsom.floatsom_params`, then `create_floatsom` from
`floatsom.base.floatsom_factories`. Training uses `som.train(data)` and inference
uses `som.predict(data)`.

`ProcessingConfig(method='batch', chunk_size=..., ray_config=RayConfig(...))`
selects distributed batch processing.
Provide Ray GPU resources and storage paths accessible to the intended workers;
start an existing cluster before connecting from a multi-node job. See
[Ray execution](docs/ray_execution.md) for path and memory constraints. Zarr I/O
requires Zarr 3 and supports explicit chunks and shards.

The optional MiniSom adapter is installed with `'.[cuda12,minisom]'`. Optional
post-training grid reformation additionally uses the RAPIDS `cudf`, `cugraph`
and `cuml` libraries; they are not required for normal SOM training and should
be installed for the chosen CUDA environment separately.

## Source layout and defaults

- `src/floatsom/`: runtime library, adapters, topology, sampling and storage.
- `src/floatsom/defaults/profiles/`: named tuned profiles with their original values.
- `tests/`: runtime and profile regression tests.
- `examples/`: quickstart and optional manual diagnostics.
- `docs/`: installation details, Ray guidance and [profile selection](docs/defaults.md).

The library is the sole home of FloatSOM runtime code. The publication repository
contains benchmarks, results and manuscript assets and installs this package.
Existing imports such as `floatsom.base.floatsom_factories` remain unchanged.
Use `FloatSOMParams(defaults_profile="publication", ...)` to select the paper's
profile; omitting the option preserves the existing standalone-library defaults.

Convert a Zarr array using `python -m floatsom.data.convert_zarr_to_fast --help`.
The optional `examples/diagnostics/check_async_loading.py` performs manual loading
and memory/performance checks; it is excluded from routine pytest collection.

## Development and verification

```bash
python -m pip install -e '.[cuda12,dev]'
python -m pytest tests
python -m build
python tools/check_distribution.py dist/*.whl
```

The full suite requires CUDA; Ray and multi-GPU cases additionally need the
corresponding resources. A CPU-only environment can install `'.[dev]'` and run:

```bash
python -m pytest tests/floatsom/test_resource_limits.py tests/floatsom/test_local_storage_wipe.py tests/floatsom/test_zarr_utils.py tests/floatsom/test_inference_intent_contracts.py
```

CI builds and checks the wheel and runs this CPU subset. It does not validate
GPU numerical behavior. The publication benchmarks use the separate
`floatsom_benchmarks` namespace and can be installed in the same environment.

## Citation

Please cite **FloatSOM: GPU Accelerated, Distributed, Topology-Flexible
Self-Organizing Maps**, by Tony Xu, Sarah Klamt, Katherine Turner, Anne Brüstle,
Felix Marsh-Wakefield and Givanna Putri. The accepted TMLR manuscript is available
in the [publication repository](https://github.com/1ordinateur/floatsom-publication).

A repository-level software license has not yet been selected.
