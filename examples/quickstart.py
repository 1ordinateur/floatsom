"""Small single-GPU example: python examples/quickstart.py."""
import numpy as np
from floatsom.base.floatsom_factories import create_floatsom
from floatsom.floatsom_params import FloatSOMParams, ProcessingConfig, SamplingConfig, TopologyConfig


def main():
    data = np.random.default_rng(42).normal(size=(2000, 8)).astype(np.float32)
    params = FloatSOMParams(
        input_dim=data.shape[1], total_iterations=20, seed=42,
        sampling_config=SamplingConfig(method="full"),
        processing_config=ProcessingConfig(method="batch", batch_mode="full_batch", chunk_size=1000),
        topology_config=TopologyConfig(topology_type="hexagonal", grid_size=10),
    )
    som = create_floatsom(params, data=data)
    som.train(data)
    assignments = som.predict(data)
    print(f"Assigned {len(assignments)} observations to {params.total_nodes} SOM nodes.")


if __name__ == "__main__":
    main()
