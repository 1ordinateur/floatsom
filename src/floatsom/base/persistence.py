"""Save a trained FloatSOM to one .npz file and load it again, with no pickle.

The file holds the weights and a JSON metadata string: the settings
(``FloatSOMParams.to_dict``) and, for a SOM with a grid reform, the grid
assignment. ``load_som`` rebuilds the topology from the settings and the
weights, so an MST or RNG graph is the same graph as after training. A loaded
SOM uses single-GPU processing (``ray_config`` is None).
"""

import json
from pathlib import Path

import cupy as cp
import numpy as np

FORMAT_NAME = "floatsom-model"
FORMAT_VERSION = 1


def save_som(som, path) -> Path:
    """Write ``som`` to ``path`` (the exact name, no suffix added) and return the path."""
    if not som.is_trained or som.weights is None:
        raise ValueError("Only a trained SOM can be saved. Call train() first.")
    path = Path(path)
    weights = np.asarray(som.get_weights(as_numpy=True), dtype=np.float32)
    params = som.params.to_dict()
    # Training can get input_dim from the data, so params.input_dim can be None.
    params["input_dim"] = int(weights.shape[1])
    metadata = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "params": params,
        "grid_reform": _grid_reform_state(som),
        "feature_names": som.feature_names,
    }
    with path.open("wb") as handle:
        np.savez(handle, weights=weights, metadata=np.array(json.dumps(metadata)))
    return path


def load_som(path):
    """Return the trained FloatSOM in a file that ``save_som`` wrote."""
    from ..floatsom_params import FloatSOMParams
    from .floatsom_factories import create_floatsom

    path = Path(path)
    not_a_model = f"{path} is not a FloatSOM model file. Make model files with FloatSOM.save()."
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files or "weights" not in archive.files:
            raise ValueError(not_a_model)
        metadata = json.loads(str(archive["metadata"]))
        weights = archive["weights"]
    if metadata.get("format") != FORMAT_NAME:
        raise ValueError(not_a_model)
    if int(metadata.get("version", 0)) > FORMAT_VERSION:
        raise ValueError(
            f"{path} has model format version {metadata['version']}, but this floatsom "
            f"reads version {FORMAT_VERSION} or older. Update floatsom."
        )

    settings = metadata["params"]
    # A loaded SOM is for prediction on one GPU. The Ray paths of the training
    # machine are not valid here, and prediction does not need Ray.
    settings.setdefault("processing_config", {})["ray_config"] = None
    som = create_floatsom(FloatSOMParams.from_dict(settings))
    som.weights = cp.asarray(weights, dtype=cp.float32)
    som.is_trained = True
    som.feature_names = metadata.get("feature_names")  # None in files from older versions
    som.topology.finalize_topology(som.weights)
    reform = metadata.get("grid_reform")
    if reform:
        som._apply_grid_reform(
            node_to_grid={int(node): (int(row), int(col)) for node, row, col in reform["node_to_grid"]},
            grid_adjacency={int(node): [int(item) for item in items] for node, items in reform["grid_adjacency"]},
            grid_type=reform["grid_type"],
        )
    return som


def _grid_reform_state(som):
    if som.grid_node_mapping is None:
        return None
    return {
        "grid_type": som.topology.reformed_type,
        "node_to_grid": [
            [int(node), int(row), int(col)] for node, (row, col) in som.grid_node_mapping.items()
        ],
        "grid_adjacency": [
            [int(node), [int(item) for item in items]] for node, items in som.adjacency_list.items()
        ],
    }
