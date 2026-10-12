# FloatSOM

GPU-accelerated self-organizing maps with flexible sampling, topologies and
local or distributed processing.

## Installation

Requires Python 3.10 or later and an NVIDIA GPU with a compatible CUDA setup.

```bash
git clone https://github.com/1ordinateur/floatsom.git
cd floatsom
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[cuda12]'
```

For CUDA 13, install `'.[cuda13]'` instead. Choose the extra that matches your
CUDA version.

## Usage

Run the example:

```bash
python examples/quickstart.py
```

Or train a SOM in Python:

```python
import numpy as np
from floatsom.base.floatsom_factories import create_floatsom
from floatsom.floatsom_params import (
    FloatSOMParams, ProcessingConfig, SamplingConfig, TopologyConfig,
)

data = np.random.default_rng(42).normal(size=(2000, 8)).astype(np.float32)
params = FloatSOMParams(
    input_dim=data.shape[1],
    total_iterations=20,
    seed=42,
    sampling_config=SamplingConfig(method="full"),
    processing_config=ProcessingConfig(
        method="batch", batch_mode="full_batch", chunk_size=1000,
    ),
    topology_config=TopologyConfig(topology_type="hexagonal", grid_size=10),
)
som = create_floatsom(params, data=data)
som.train(data)
assignments = som.predict(data)
```

See [profile selection](docs/defaults.md) and [Ray execution](docs/ray_execution.md)
for more configuration options.

## Citation

Please cite **FloatSOM: GPU Accelerated, Distributed, Topology-Flexible
Self-Organizing Maps**, by Tony Xu, Sarah Klamt, Katharine Turner, Anne Brüstle,
Felix Marsh-Wakefield and Givanna Putri. The manuscript and benchmarks are in the
[publication repository](https://github.com/1ordinateur/floatsom-publication).

## License

This project is licensed under the MIT License.
