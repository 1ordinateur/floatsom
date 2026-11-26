"""The NCCL group start gives an install hint when Ray finds no NCCL backend.

Ray is an external dependency, so the test replaces only
``ray.util.collective.create_collective_group``. The FloatSOM method
``RayWorkerManager._initialize_collective_group`` runs unchanged.
"""

import pytest

pytest.importorskip("cupy")
pytest.importorskip("ray")

from floatsom.processing.ray_ops import ray_pipeline_base  # noqa: E402

# The messages of Ray 2.56 and later, and of Ray 2.50 to 2.55.
RAY_NO_NCCL_MESSAGES = [
    "Backend NCCL is registered but not available. "
    "Please check the installation requirements for this backend.",
    "NCCL is not available.",
]


def manager_with_no_workers():
    manager = object.__new__(ray_pipeline_base.RayWorkerManager)
    manager.collective_group_created = False
    manager.collective_group_name = "test_group"
    manager.workers = []
    return manager


def raise_error(message):
    def create_collective_group(*args, **kwargs):
        raise RuntimeError(message)

    return create_collective_group


@pytest.mark.parametrize("ray_message", RAY_NO_NCCL_MESSAGES)
def test_missing_nccl_backend_gives_install_hint(monkeypatch, ray_message):
    monkeypatch.setattr(
        ray_pipeline_base.collective, "create_collective_group", raise_error(ray_message)
    )
    with pytest.raises(RuntimeError) as error:
        manager_with_no_workers()._initialize_collective_group()
    message = str(error.value)
    assert "floatsom[ray-cuda12]" in message
    assert "floatsom[ray-cuda13]" in message
    assert "PyTorch" in message
    assert ray_message in message


def test_other_collective_errors_are_not_changed(monkeypatch):
    monkeypatch.setattr(
        ray_pipeline_base.collective, "create_collective_group", raise_error("Ranks must be non-negative.")
    )
    with pytest.raises(RuntimeError, match=r"^Ranks must be non-negative\.$"):
        manager_with_no_workers()._initialize_collective_group()
