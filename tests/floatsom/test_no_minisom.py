"""FloatSOM has no MiniSOM backend, option or dependency.

These tests need no GPU: they use the params module, which imports without
CuPy, and the package files.
"""

import dataclasses
import importlib.util
import inspect
import tomllib
from pathlib import Path

import pytest

from floatsom import floatsom_params
from floatsom.floatsom_params import FloatSOMParams, ProcessingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_PATHS = ("src", "docs", "examples", "README.md", "pyproject.toml")
# load_som drops this removed setting, so that older model files still load.
ALLOWED_LINES = {("src/floatsom/base/persistence.py", 'settings.pop("minisom_defaults", None)')}


def test_minisom_processing_method_is_rejected():
    with pytest.raises(ValueError, match="Invalid processing method"):
        FloatSOMParams(
            input_dim=4,
            processing_config=ProcessingConfig(method="minisom", chunk_size=1000),
        )


def test_minisom_weighted_normalization_is_rejected():
    with pytest.raises(ValueError, match="Invalid normalization"):
        FloatSOMParams(
            input_dim=4,
            processing_config=ProcessingConfig(normalization="minisom_weighted", chunk_size=1000),
        )


def test_params_have_no_minisom_defaults_field():
    names = {field.name for field in dataclasses.fields(FloatSOMParams)}
    assert "minisom_defaults" not in names


@pytest.mark.parametrize(
    "function_name",
    [
        "calculate_radius",
        "calculate_learning_rate",
        "precompute_all_radii",
        "precompute_all_learning_rates",
    ],
)
def test_decay_functions_have_no_use_minisom_parameter(function_name):
    function = getattr(floatsom_params, function_name)
    assert "use_minisom" not in inspect.signature(function).parameters


def test_package_has_no_adapters_module():
    assert importlib.util.find_spec("floatsom.adapters") is None


def test_pyproject_has_no_minisom_extra():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        project = tomllib.load(handle)["project"]
    assert "minisom" not in project.get("optional-dependencies", {})


def test_package_sources_and_docs_do_not_mention_minisom():
    files = []
    for name in SCANNED_PATHS:
        path = REPO_ROOT / name
        files.extend([path] if path.is_file() else (p for p in path.rglob("*") if p.is_file()))
    hits = []
    for path in files:
        # Skip build output: "pip install -e ." and "python -m build" write
        # src/floatsom.egg-info, and its SOURCES.txt lists the test file names.
        if "__pycache__" in path.parts or any(part.endswith(".egg-info") for part in path.parts):
            continue
        name = path.relative_to(REPO_ROOT).as_posix()
        for line in path.read_bytes().decode("utf-8", errors="ignore").splitlines():
            if "minisom" in line.lower() and (name, line.strip()) not in ALLOWED_LINES:
                hits.append(f"{name}: {line.strip()}")
    assert hits == []
