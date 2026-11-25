"""GPU tests: manhattan distances, the check of p, and norm_p range for large and small values."""

import math

import numpy as np
import pytest

from ..conftest import GPU_AVAILABLE

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")

if GPU_AVAILABLE:
    import cupy as cp

    from floatsom import FloatSOMParams
    from floatsom.floatsom_params import ProcessingConfig
    from floatsom.processing.utils import calculate_distances, find_bmus


def host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


def l1_distances(batch, weights):
    return np.abs(batch[:, None, :] - weights[None, :, :]).sum(axis=2)


def lp_distances_float64(batch, weights, p):
    diff = np.abs(batch.astype(np.float64)[:, None, :] - weights.astype(np.float64)[None, :, :])
    return (diff ** p).sum(axis=2) ** (1.0 / p)


# --- manhattan --------------------------------------------------------------


def test_calculate_distances_manhattan_matches_numpy():
    rng = np.random.default_rng(10)
    batch = rng.normal(size=(17, 6)).astype(np.float32)
    weights = rng.normal(size=(9, 6)).astype(np.float32)
    distances = calculate_distances(cp.asarray(batch), cp.asarray(weights), "manhattan")
    np.testing.assert_allclose(host(distances), l1_distances(batch, weights), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("node_chunk_size", [900, 4])  # small-node branch; chunked branch
def test_find_bmus_manhattan_matches_numpy(node_chunk_size):
    rng = np.random.default_rng(11)
    batch = rng.normal(size=(50, 5)).astype(np.float32)
    weights = rng.normal(size=(12, 5)).astype(np.float32)
    bmus = find_bmus(
        cp.asarray(batch), cp.asarray(weights), metric="manhattan", node_chunk_size=node_chunk_size
    )
    np.testing.assert_array_equal(host(bmus), l1_distances(batch, weights).argmin(axis=1))


# --- p must be finite and >= 1 -----------------------------------------------

BAD_P_VALUES = [0.0, 0.5, -1.0, math.nan, math.inf]


@pytest.mark.parametrize("p", BAD_P_VALUES)
def test_calculate_distances_rejects_bad_p(p):
    batch = cp.ones((3, 2), dtype=cp.float32)
    weights = cp.zeros((4, 2), dtype=cp.float32)
    with pytest.raises(ValueError, match="finite p >= 1"):
        calculate_distances(batch, weights, "norm_p", p=p)


@pytest.mark.parametrize("p", [math.nan, math.inf])
def test_params_reject_p_not_finite(p):
    config = ProcessingConfig(distance_metric="norm_p", distance_metric_params={"p": p})
    with pytest.raises(ValueError, match="finite p >= 1"):
        FloatSOMParams(input_dim=2, processing_config=config)


# --- norm_p on float32 data with no overflow and no underflow ----------------


@pytest.mark.parametrize(
    "scale,p",
    [(1e5, 8.5), (1e-3, 20.5)],  # float32 overflow of diff ** p; float32 underflow of diff ** p
)
def test_norm_p_float32_range(scale, p):
    rng = np.random.default_rng(12)
    batch = rng.uniform(0.0, scale, size=(6, 3)).astype(np.float32)
    weights = rng.uniform(0.0, scale, size=(5, 3)).astype(np.float32)
    expected = lp_distances_float64(batch, weights, p)
    distances = host(calculate_distances(cp.asarray(batch), cp.asarray(weights), "norm_p", p=p))
    assert distances.dtype == np.float32
    assert np.isfinite(distances).all()
    np.testing.assert_allclose(distances, expected, rtol=1e-4)
    bmus = find_bmus(cp.asarray(batch), cp.asarray(weights), metric="norm_p", p=p)
    np.testing.assert_array_equal(host(bmus), expected.argmin(axis=1))
