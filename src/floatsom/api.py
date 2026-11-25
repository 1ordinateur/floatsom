"""Train a SOM in one call, with defaults for every setting except the data."""

import math
import numbers
from typing import Optional

from .base.arrays import as_feature_matrix, count_rows_not_finite
from .base.floatsom import FloatSOM
from .base.floatsom_factories import create_floatsom
from .floatsom_params import FloatSOMParams, ProcessingConfig, SamplingConfig, TopologyConfig
from .processing.processing_params import calculate_auto_chunk_size_for_method

MIN_GRID_SIZE = 2
MAX_GRID_SIZE = 32


def suggest_grid_size(n_samples: int) -> int:
    """Return a grid side for about 5 * sqrt(n_samples) nodes, from 2 to 32.

    This is the default map size rule of the SOM Toolbox (Vesanto et al.).
    The side is the square root of the node count, rounded.
    """
    if n_samples < 1:
        raise ValueError("suggest_grid_size needs at least one sample.")
    side = round(math.sqrt(5.0 * math.sqrt(n_samples)))
    return min(MAX_GRID_SIZE, max(MIN_GRID_SIZE, side))


def train_som(
    data,
    *,
    grid_size: Optional[int] = None,
    topology: str = "hexagonal",
    iterations: int = 100,
    distance_metric: str = "euclidean",
    profile: str = "library",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> FloatSOM:
    """Train a SOM on ``data`` on one GPU, and return the trained ``FloatSOM``.

    Args:
        data: Array-like with shape (n_samples, n_features): a NumPy or CuPy
            array, a pandas DataFrame with numeric columns, or a nested list.
        grid_size: Side of the map; the map has grid_size ** 2 nodes. The
            default comes from ``suggest_grid_size(n_samples)``.
        topology: "hexagonal", "grid", "mst" or "rng".
        iterations: Number of training iterations.
        distance_metric: "euclidean", "cosine" or "manhattan".
        profile: Tuned defaults profile; see ``floatsom.PROFILE_NAMES``.
        seed: Random seed, for results that you can repeat.
        verbose: Log the training progress.

    For data that does not fit in memory, for subsampling or for more GPUs,
    build a ``FloatSOMParams`` and call ``create_floatsom`` instead.
    """
    matrix = as_feature_matrix(data, "train_som")
    n_samples, n_features = (int(size) for size in matrix.shape)
    side = suggest_grid_size(n_samples) if grid_size is None else _integer_option(
        grid_size, "grid_size", MIN_GRID_SIZE)
    iterations = _integer_option(iterations, "iterations", 1)

    def build_params(chunk_size):
        return FloatSOMParams(
            input_dim=n_features,
            total_iterations=iterations,
            seed=seed,
            verbose=verbose,
            defaults_profile=profile,
            sampling_config=SamplingConfig(method="full"),
            processing_config=ProcessingConfig(
                chunk_size=chunk_size,
                method="batch",
                batch_mode="full_batch",
                distance_metric=distance_metric,
            ),
            topology_config=TopologyConfig(topology_type=topology, grid_size=side),
        )

    build_params(None)  # Check the options on the CPU, before any GPU work.
    bad_rows = count_rows_not_finite(matrix)
    if bad_rows:
        raise ValueError(
            f"{bad_rows} of {n_samples} rows have NaN or infinite values. "
            "Remove or impute these rows before training."
        )

    chunk_size = min(n_samples, calculate_auto_chunk_size_for_method(n_features, "batch"))
    params = build_params(chunk_size)
    som = create_floatsom(params, data=matrix)
    som.train(matrix)
    return som


def _integer_option(value, name: str, minimum: int) -> int:
    """Return ``value`` as an int, or raise ValueError if it is not an integer >= minimum."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of {minimum} or more, got {value!r}.")
    return int(value)
