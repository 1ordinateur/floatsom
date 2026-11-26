"""Install and import behaviour when an optional dependency is missing.

Each test runs Python in a subprocess. The subprocess blocks CuPy or Ray in
``sys.modules`` before it imports floatsom, so the test does not need a
machine without these packages. Only the external package is blocked; the
floatsom code under test runs unchanged.
"""

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CUPY_MODULES = ("cupy", "cupyx")
RAY_MODULES = ("ray",)


def run_python(code, blocked=(), path=None):
    """Run ``code`` in a new interpreter with the ``blocked`` modules absent.

    ``path`` replaces PATH, for example to hide the nvidia-smi of a GPU host.
    """
    prelude = "import sys\n" + "".join(
        f"sys.modules[{name!r}] = None\n" for name in blocked
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=None if path is None else {**os.environ, "PATH": str(path)},
        timeout=300,
    )


def assert_ok(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_config_classes_import_without_cupy():
    result = run_python(
        """
        from floatsom import (
            FloatSOMParams, SamplingConfig, ProcessingConfig, TopologyConfig, RayConfig,
        )
        params = FloatSOMParams(input_dim=4, topology_config=TopologyConfig(grid_size=3))
        print(params.total_nodes)
        """,
        blocked=CUPY_MODULES,
    )
    assert_ok(result)
    assert result.stdout.strip() == "9"


def test_params_module_imports_without_cupy():
    result = run_python(
        "from floatsom.floatsom_params import FloatSOMParams\nprint('ok')",
        blocked=CUPY_MODULES,
    )
    assert_ok(result)


@pytest.mark.parametrize(
    "statement",
    [
        "from floatsom import create_floatsom",
        "from floatsom import train_som",
        "from floatsom import FloatSOM",
        "from floatsom.base.floatsom_factories import create_floatsom",
        "from floatsom.base.floatsom import FloatSOM",
    ],
)
def test_training_imports_without_cupy_give_install_hint(statement):
    result = run_python(statement, blocked=CUPY_MODULES)
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert "floatsom[cuda12]" in result.stderr
    assert "floatsom[cuda13]" in result.stderr


def test_unknown_top_level_name_raises_attribute_error():
    result = run_python(
        """
        import floatsom
        try:
            floatsom.not_a_real_name
        except AttributeError as error:
            print("AttributeError", error)
        """
    )
    assert_ok(result)
    assert "not_a_real_name" in result.stdout


def test_ray_processor_without_ray_gives_install_hint(tmp_path):
    result = run_python(
        f"""
        from floatsom.floatsom_params import ProcessingConfig, RayConfig
        from floatsom.processing.processor_factory import ProcessorFactory
        ray_config = RayConfig(
            chunk_size=100,
            storage_path={str(tmp_path / 'shared')!r},
            local_storage_path={str(tmp_path / 'local')!r},
        )
        config = ProcessingConfig(chunk_size=100, method="batch", ray_config=ray_config)
        ProcessorFactory.create(config, 4, 16)
        """,
        blocked=RAY_MODULES,
    )
    assert result.returncode != 0
    assert "floatsom[ray]" in result.stderr


def test_ray_is_an_optional_extra():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    required = [dependency.split("[")[0].split(">")[0].strip() for dependency in project["dependencies"]]
    assert "ray" not in required
    assert any(item.startswith("ray") for item in project["optional-dependencies"]["ray"])


def test_check_environment_reports_missing_cupy_and_ray(tmp_path):
    # PATH has no nvidia-smi, so the report gives the general CuPy hint.
    result = run_python(
        """
        import json
        import floatsom
        report = floatsom.check_environment(print_report=False)
        print(json.dumps(report))
        """,
        blocked=CUPY_MODULES + RAY_MODULES,
        path=tmp_path,
    )
    assert_ok(result)
    import json

    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["cupy"] is None
    assert report["ray"] is None
    assert report["gpus"] == []
    assert any("floatsom[cuda12]" in problem for problem in report["problems"])


def test_module_command_prints_report_and_fails_without_cupy(tmp_path):
    # PATH has no nvidia-smi, so the report gives the general CuPy hint.
    result = run_python(
        "import runpy\nrunpy.run_module('floatsom', run_name='__main__')",
        blocked=CUPY_MODULES + RAY_MODULES,
        path=tmp_path,
    )
    assert result.returncode == 1
    assert "CuPy" in result.stdout
    assert "floatsom[cuda12]" in result.stdout
    assert "floatsom[ray]" in result.stdout
