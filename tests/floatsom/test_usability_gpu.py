"""GPU tests for train_som, prediction metrics and save/load."""

import numpy as np
import pandas as pd
import pytest

from ..conftest import GPU_AVAILABLE

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")

if GPU_AVAILABLE:
    import cupy as cp

    from floatsom import FloatSOM, FloatSOMParams, create_floatsom, train_som
    from floatsom.floatsom_params import ProcessingConfig, SamplingConfig, TopologyConfig


def blobs(n_samples=2000, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=5.0, size=(4, n_features))
    labels = rng.integers(0, 4, size=n_samples)
    return (centres[labels] + rng.normal(size=(n_samples, n_features))).astype(np.float32)


def small_params(topology="hexagonal", metric="euclidean", **extra):
    topology_config = (
        TopologyConfig(topology_type=topology, grid_size=4)
        if topology in {"grid", "hexagonal"}
        else TopologyConfig(topology_type=topology, num_nodes=16)
    )
    return FloatSOMParams(
        input_dim=8,
        total_iterations=10,
        seed=3,
        sampling_config=SamplingConfig(method="full"),
        processing_config=ProcessingConfig(chunk_size=500, distance_metric=metric),
        topology_config=topology_config,
        **extra,
    )


def trained(params, data):
    som = create_floatsom(params, data=data)
    som.train(data)
    return som


def host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


# --- train_som -------------------------------------------------------------


def test_train_som_with_only_data():
    data = blobs()
    som = train_som(data, seed=1)
    assert isinstance(som, FloatSOM)
    assert som.is_trained
    assert som.get_weights().shape == (15 * 15, 8)
    assignments = host(som.predict(data))
    assert assignments.shape == (2000,)
    assert assignments.min() >= 0 and assignments.max() < 15 * 15


def test_train_som_accepts_dataframe_lists_and_float64():
    data = blobs(n_samples=400)
    reference = train_som(data, grid_size=5, seed=4).get_weights()
    frame = pd.DataFrame(data.astype(np.float64), columns=[f"m{i}" for i in range(8)])
    for value in (frame, data.astype(np.float64), data.tolist(), cp.asarray(data)):
        weights = train_som(value, grid_size=5, seed=4).get_weights()
        np.testing.assert_allclose(weights, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("topology", ["mst", "rng"])
def test_train_som_graph_topologies_use_grid_size_squared_nodes(topology):
    som = train_som(blobs(n_samples=500), grid_size=4, topology=topology, seed=2)
    assert som.get_weights().shape == (16, 8)


# --- predict and map_vectors -----------------------------------------------


def cosine_and_euclidean_disagree(metric):
    """Train a 2-feature SOM, then put known weights on two nodes."""
    params = FloatSOMParams(
        input_dim=2,
        total_iterations=2,
        seed=0,
        processing_config=ProcessingConfig(chunk_size=100, distance_metric=metric),
        topology_config=TopologyConfig(topology_type="grid", grid_size=2),
    )
    som = trained(params, np.random.default_rng(0).normal(size=(50, 2)).astype(np.float32))
    # Sample [9, 0]: Euclidean nearest is node 1 [10, 1]; cosine nearest is node 0 [1, 0].
    som.weights = cp.asarray([[1.0, 0.0], [10.0, 1.0], [-5.0, -5.0], [-6.0, 5.0]], dtype=cp.float32)
    return som, np.array([[9.0, 0.0]], dtype=np.float32)


def test_predict_uses_training_metric():
    som, sample = cosine_and_euclidean_disagree("cosine")
    assert int(host(som.predict(sample))[0]) == 0
    euclidean_som, _ = cosine_and_euclidean_disagree("euclidean")
    assert int(host(euclidean_som.predict(sample))[0]) == 1


def test_infer_uses_training_metric():
    som, sample = cosine_and_euclidean_disagree("cosine")
    assert int(som.infer(sample)[0]) == 0


def test_map_vectors_uses_training_metric():
    som, sample = cosine_and_euclidean_disagree("cosine")
    indices, distances = som.map_vectors(sample)
    assert int(host(indices)[0]) == 0
    assert float(host(distances)[0]) == pytest.approx(0.0, abs=1e-5)


def test_map_vectors_matches_brute_force_euclidean():
    data = blobs(n_samples=1200)
    som = trained(small_params(), data)
    weights = som.get_weights()
    distances_all = np.sqrt(((data[:, None, :] - weights[None, :, :]) ** 2).sum(axis=2))
    indices, distances = som.map_vectors(data)
    np.testing.assert_array_equal(host(indices), distances_all.argmin(axis=1))
    np.testing.assert_allclose(host(distances), distances_all.min(axis=1), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(host(indices), host(som.predict(data)))


@pytest.mark.parametrize("method", ["predict", "map_vectors", "infer"])
def test_wrong_feature_count_gives_clear_error(method):
    som = trained(small_params(), blobs(n_samples=300))
    with pytest.raises(ValueError, match="5 features.*8 features"):
        getattr(som, method)(np.zeros((3, 5), dtype=np.float32))


# --- save and load ---------------------------------------------------------


@pytest.mark.parametrize("topology", ["hexagonal", "grid", "mst", "rng"])
def test_save_load_round_trip(topology, tmp_path):
    data = blobs(n_samples=600)
    som = trained(small_params(topology=topology, metric="cosine"), data)
    path = tmp_path / "model.floatsom"
    returned = som.save(path)
    assert returned == path and path.exists()

    loaded = FloatSOM.load(path)
    assert loaded.is_trained
    np.testing.assert_array_equal(loaded.get_weights(), som.get_weights())
    assert loaded.params.to_dict() == som.params.to_dict()
    np.testing.assert_array_equal(host(loaded.predict(data)), host(som.predict(data)))
    assert getattr(loaded.topology, "adjacency_list", None) == getattr(
        som.topology, "adjacency_list", None
    )


def test_save_load_keeps_grid_reform(tmp_path):
    data = blobs(n_samples=600)
    som = trained(
        small_params(topology="mst", reform_grid=True, reform_delaunay_backend="local"), data
    )
    assert som.grid_node_mapping is not None
    som.save(tmp_path / "reformed.npz")
    loaded = FloatSOM.load(tmp_path / "reformed.npz")
    assert loaded.grid_node_mapping == som.grid_node_mapping
    assert loaded.adjacency_list == som.adjacency_list
    assert loaded.topology.adjacency_list == som.topology.adjacency_list
    assert loaded.original_adjacency_list == som.original_adjacency_list
    assert loaded.topology.is_reformed


def test_save_untrained_som_raises(tmp_path):
    som = create_floatsom(small_params(), data=blobs(n_samples=50))
    with pytest.raises(ValueError, match="trained"):
        som.save(tmp_path / "untrained.npz")


def test_train_som_accepts_numpy_integer_options():
    som = train_som(blobs(n_samples=300), grid_size=np.int64(4), iterations=np.int64(5), seed=1)
    assert som.get_weights().shape == (16, 8)


def test_save_load_when_input_dim_comes_from_data(tmp_path):
    data = blobs(n_samples=400)
    params = small_params()
    params.input_dim = None
    som = trained(params, data)
    som.save(tmp_path / "no_input_dim.npz")
    loaded = FloatSOM.load(tmp_path / "no_input_dim.npz")
    np.testing.assert_array_equal(host(loaded.predict(data)), host(som.predict(data)))


def test_load_of_a_ray_configured_som_gives_a_local_som(tmp_path):
    from floatsom.floatsom_params import RayConfig
    from floatsom.processing.batch_processor import BatchProcessor

    data = blobs(n_samples=400)
    som = trained(small_params(), data)
    som.params.processing_config.ray_config = RayConfig(
        chunk_size=500,
        storage_path=str(tmp_path / "shared"),
        local_storage_path=str(tmp_path / "local"),
    )
    som.save(tmp_path / "ray.npz")
    loaded = FloatSOM.load(tmp_path / "ray.npz")
    assert loaded.params.processing_config.ray_config is None
    assert isinstance(loaded.processor, BatchProcessor)
    np.testing.assert_array_equal(host(loaded.predict(data)), host(som.predict(data)))


def test_load_rejects_a_file_that_save_did_not_write(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, weights=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="FloatSOM.save"):
        FloatSOM.load(path)


def test_load_ignores_the_removed_minisom_defaults_setting(tmp_path):
    """Files saved before the MiniSOM removal store "minisom_defaults"; they still load."""
    import json

    data = blobs(n_samples=400)
    som = trained(small_params(), data)
    path = tmp_path / "old.npz"
    som.save(path)
    with np.load(path, allow_pickle=False) as archive:
        weights = archive["weights"]
        metadata = json.loads(str(archive["metadata"]))
    metadata["params"]["minisom_defaults"] = False
    np.savez(path, weights=weights, metadata=np.array(json.dumps(metadata)))

    loaded = FloatSOM.load(path)
    np.testing.assert_array_equal(host(loaded.predict(data)), host(som.predict(data)))
