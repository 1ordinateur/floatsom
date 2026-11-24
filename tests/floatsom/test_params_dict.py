"""FloatSOMParams.to_dict and FloatSOMParams.from_dict (no GPU needed)."""

import json

import pytest

from floatsom.floatsom_params import (
    FloatSOMParams,
    ProcessingConfig,
    RayConfig,
    SamplingConfig,
    TopologyConfig,
)
from floatsom.processing.processing_params import AsyncLoadingConfig

RUNTIME_KEYS = {
    "current_radius",
    "current_learning_rate",
    "current_iteration",
    "current_momentum",
    "delta_weights",
    "selector_callback",
    "bmu_scheduler",
}


def hexagonal_params():
    return FloatSOMParams(
        input_dim=8,
        total_iterations=30,
        seed=7,
        defaults_profile="publication",
        sampling_config=SamplingConfig(method="random", target_proportion=0.2),
        processing_config=ProcessingConfig(
            chunk_size=500, distance_metric="cosine", batch_mode="minibatch",
        ),
        topology_config=TopologyConfig(topology_type="hexagonal", grid_size=6),
    )


def ray_mst_params(tmp_path):
    return FloatSOMParams(
        input_dim=5,
        sampling_config=SamplingConfig(method="random"),
        processing_config=ProcessingConfig(
            chunk_size=1000,
            distance_metric="norm_p",
            distance_metric_params={"p": 4.0},
            async_loading_config=AsyncLoadingConfig(),
            ray_config=RayConfig(
                chunk_size=1000,
                storage_path=str(tmp_path / "shared"),
                local_storage_path=str(tmp_path / "local"),
                num_gpus=2,
            ),
        ),
        topology_config=TopologyConfig(topology_type="mst", num_nodes=50),
    )


def test_to_dict_is_json_safe_and_nested():
    data = hexagonal_params().to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["topology_config"]["grid_size"] == 6
    assert data["processing_config"]["distance_metric"] == "cosine"
    assert data["sampling_config"]["target_proportion"] == 0.2


def test_to_dict_leaves_out_run_state():
    params = hexagonal_params()
    params.current_radius = 3.0
    params.current_iteration = 5
    assert RUNTIME_KEYS.isdisjoint(params.to_dict())


@pytest.mark.parametrize("builder", ["hexagonal", "ray_mst"])
def test_round_trip_through_json(builder, tmp_path):
    params = hexagonal_params() if builder == "hexagonal" else ray_mst_params(tmp_path)
    data = json.loads(json.dumps(params.to_dict()))
    rebuilt = FloatSOMParams.from_dict(data)
    assert rebuilt.to_dict() == params.to_dict()
    assert rebuilt.total_nodes == params.total_nodes


def test_from_dict_builds_nested_config_objects(tmp_path):
    rebuilt = FloatSOMParams.from_dict(ray_mst_params(tmp_path).to_dict())
    assert isinstance(rebuilt.sampling_config, SamplingConfig)
    assert isinstance(rebuilt.topology_config, TopologyConfig)
    assert isinstance(rebuilt.processing_config, ProcessingConfig)
    assert isinstance(rebuilt.processing_config.ray_config, RayConfig)
    assert isinstance(rebuilt.processing_config.async_loading_config, AsyncLoadingConfig)
    assert rebuilt.processing_config.ray_config.num_gpus == 2


def test_from_dict_with_few_keys_uses_defaults():
    params = FloatSOMParams.from_dict(
        {"input_dim": 3, "topology_config": {"topology_type": "hexagonal", "grid_size": 4}}
    )
    reference = FloatSOMParams(
        input_dim=3, topology_config=TopologyConfig(topology_type="hexagonal", grid_size=4)
    )
    assert params.to_dict() == reference.to_dict()


def test_from_dict_processing_config_without_chunk_size_uses_default():
    params = FloatSOMParams.from_dict(
        {"input_dim": 3, "processing_config": {"distance_metric": "manhattan"}}
    )
    reference = FloatSOMParams(input_dim=3)
    assert params.processing_config.distance_metric == "manhattan"
    assert params.processing_config.chunk_size == reference.processing_config.chunk_size


def test_from_dict_rejects_run_state_keys():
    with pytest.raises(ValueError, match="current_radius"):
        FloatSOMParams.from_dict({"input_dim": 3, "current_radius": 1.0})


@pytest.mark.parametrize(
    "data,bad_key,suggestion",
    [
        ({"total_iteration": 5}, "total_iteration", "total_iterations"),
        ({"topology_config": {"grid_sise": 5}}, "topology_config.grid_sise", "grid_size"),
        (
            {"processing_config": {"chunk_size": 10, "distance_metrc": "cosine"}},
            "processing_config.distance_metrc",
            "distance_metric",
        ),
    ],
)
def test_from_dict_unknown_key_names_key_and_suggestion(data, bad_key, suggestion):
    with pytest.raises(ValueError) as error:
        FloatSOMParams.from_dict(data)
    message = str(error.value)
    assert bad_key in message
    assert suggestion in message


def test_from_dict_rejects_a_non_mapping_section():
    with pytest.raises(ValueError, match="topology_config"):
        FloatSOMParams.from_dict({"topology_config": 5})


def test_from_dict_keeps_validation_errors():
    with pytest.raises(ValueError, match="Invalid topology type"):
        FloatSOMParams.from_dict({"topology_config": {"topology_type": "triangle"}})


def test_processing_config_has_a_default_chunk_size():
    config = ProcessingConfig(distance_metric="cosine")
    assert config.chunk_size == FloatSOMParams().processing_config.chunk_size


def test_from_dict_rejects_none_for_a_required_section():
    with pytest.raises(ValueError, match="processing_config"):
        FloatSOMParams.from_dict({"input_dim": 3, "processing_config": None})


def test_to_dict_converts_numpy_arrays_in_settings():
    import numpy as np

    params = FloatSOMParams(
        input_dim=3,
        processing_config=ProcessingConfig(influence_params={"weights": np.array([1.0, 2.0])}),
    )
    data = json.loads(json.dumps(params.to_dict()))
    assert data["processing_config"]["influence_params"]["weights"] == [1.0, 2.0]
