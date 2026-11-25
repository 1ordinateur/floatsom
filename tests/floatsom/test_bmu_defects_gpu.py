"""GPU tests: Zarr output of infer, metric options in find_bmus, and norm_p distances."""

import numpy as np
import pytest

from ..conftest import GPU_AVAILABLE

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")

if GPU_AVAILABLE:
    import cupy as cp
    import zarr

    from floatsom import FloatSOMParams, create_floatsom, train_som
    from floatsom.floatsom_params import ProcessingConfig, TopologyConfig
    from floatsom.processing.utils import calculate_distances, find_bmus


def blobs(n_samples=500, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=5.0, size=(3, n_features))
    labels = rng.integers(0, 3, size=n_samples)
    return (centres[labels] + rng.normal(size=(n_samples, n_features))).astype(np.float32)


def host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


def lp_distances(batch, weights, p):
    return (np.abs(batch[:, None, :] - weights[None, :, :]) ** p).sum(axis=2) ** (1.0 / p)


# --- infer writes the same assignments to Zarr as predict -------------------


@pytest.mark.parametrize(
    "chunk_size,use_cuda_streams",
    [(1000, True), (128, False)],  # one chunk; several chunks, no streams
)
def test_infer_zarr_output_matches_predict(tmp_path, chunk_size, use_cuda_streams):
    data = blobs()
    params = FloatSOMParams(
        input_dim=4,
        total_iterations=10,
        seed=1,
        processing_config=ProcessingConfig(chunk_size=500),
        topology_config=TopologyConfig(topology_type="hexagonal", grid_size=3),
    )
    som = create_floatsom(params, data=data)
    som.train(data)
    output_path = tmp_path / "assignments.zarr"
    som.infer(
        data, output_path=str(output_path), chunk_size=chunk_size, use_cuda_streams=use_cuda_streams
    )
    written = zarr.open_group(str(output_path), mode="r")["assignments"][:]
    np.testing.assert_array_equal(written, host(som.predict(data)).astype(np.int32))


# --- find_bmus gives the metric options to both node branches ---------------

# Sample [1, 0]. Default epsilon: node 0 [1, 0] is the cosine BMU. With epsilon 10, both
# norms clamp to 10, so the distance is 1 - dot / 100, and node 1 [3, 4] (dot 3) wins.
COSINE_BATCH = np.array([[1.0, 0.0]], dtype=np.float32)
COSINE_WEIGHTS = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)


@pytest.mark.parametrize("node_chunk_size", [900, 1])  # small-node branch; chunked branch
def test_find_bmus_uses_epsilon_in_both_node_branches(node_chunk_size):
    batch, weights = cp.asarray(COSINE_BATCH), cp.asarray(COSINE_WEIGHTS)
    default = find_bmus(batch, weights, metric="cosine", node_chunk_size=node_chunk_size)
    large_eps = find_bmus(
        batch, weights, metric="cosine", node_chunk_size=node_chunk_size, epsilon=10.0
    )
    assert int(host(default)[0]) == 0
    assert int(host(large_eps)[0]) == 1


@pytest.mark.parametrize("node_chunk_size", [900, 1])
def test_find_bmus_accepts_apply_sqrt_in_metric_options(node_chunk_size):
    rng = np.random.default_rng(2)
    batch = rng.normal(size=(20, 3)).astype(np.float32)
    weights = rng.normal(size=(5, 3)).astype(np.float32)
    _, distances = find_bmus(
        cp.asarray(batch), cp.asarray(weights), return_distances=True,
        metric="euclidean", node_chunk_size=node_chunk_size, apply_sqrt=False,
    )
    squared = ((batch[:, None, :] - weights[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    np.testing.assert_allclose(host(distances), squared, rtol=1e-4, atol=1e-5)


# --- norm_p distances --------------------------------------------------------


@pytest.mark.parametrize("p", [1.0, 3.0])
def test_calculate_distances_norm_p_matches_numpy(p):
    rng = np.random.default_rng(3)
    batch = rng.normal(size=(17, 6)).astype(np.float32)
    weights = rng.normal(size=(9, 6)).astype(np.float32)
    distances = calculate_distances(cp.asarray(batch), cp.asarray(weights), "norm_p", p=p)
    np.testing.assert_allclose(host(distances), lp_distances(batch, weights, p), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("node_chunk_size", [900, 4])
def test_find_bmus_norm_p_matches_numpy(node_chunk_size):
    rng = np.random.default_rng(4)
    batch = rng.normal(size=(50, 5)).astype(np.float32)
    weights = rng.normal(size=(12, 5)).astype(np.float32)
    bmus = find_bmus(
        cp.asarray(batch), cp.asarray(weights), metric="norm_p", node_chunk_size=node_chunk_size, p=4.0
    )
    np.testing.assert_array_equal(host(bmus), lp_distances(batch, weights, 4.0).argmin(axis=1))


def test_train_som_with_norm_p_predicts_lp_bmus():
    data = blobs(n_samples=400)
    som = train_som(data, grid_size=4, distance_metric="norm_p", iterations=10, seed=5)
    p = som.params.processing_config.distance_metric_params["p"]
    expected = lp_distances(data, som.get_weights(), p).argmin(axis=1)
    np.testing.assert_array_equal(host(som.predict(data)), expected)
