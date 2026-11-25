"""GPU tests: output types, infer from files, column names, path errors and progress output."""

import logging
from pathlib import Path

import numpy as np
import pytest

from ..conftest import GPU_AVAILABLE

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")

if GPU_AVAILABLE:
    import cupy as cp
    import pandas as pd

    import floatsom
    from floatsom.data.zarr_utils import create_sharded_array

COLUMNS = ["CD3", "CD4", "CD8", "CD19"]


def blobs(n_samples=600, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=5.0, size=(3, len(COLUMNS)))
    labels = rng.integers(0, 3, size=n_samples)
    return (centres[labels] + rng.normal(size=(n_samples, len(COLUMNS)))).astype(np.float32)


@pytest.fixture(scope="module")
def data():
    return blobs()


@pytest.fixture(scope="module")
def frame(data):
    return pd.DataFrame(data, columns=COLUMNS)


@pytest.fixture(scope="module")
def som(data):
    return floatsom.train_som(data, grid_size=4, iterations=10, seed=3)


@pytest.fixture(scope="module")
def named_som(frame):
    return floatsom.train_som(frame, grid_size=4, iterations=10, seed=3)


# --- predict and map_vectors give the same kind of array as the input -------


def test_predict_gives_numpy_for_host_input(som, data, frame):
    expected = cp.asnumpy(som.predict(cp.asarray(data)))
    for host_input in (data, frame, data.tolist()):
        labels = som.predict(host_input)
        assert isinstance(labels, np.ndarray), type(host_input)
        np.testing.assert_array_equal(labels, expected)
    assert isinstance(som.predict(cp.asarray(data)), cp.ndarray)  # CuPy input keeps CuPy output


def test_map_vectors_follows_the_input_type(som, data):
    host_bmus, host_distances = som.map_vectors(data)
    assert isinstance(host_bmus, np.ndarray) and isinstance(host_distances, np.ndarray)
    gpu_bmus, gpu_distances = som.map_vectors(cp.asarray(data))
    assert isinstance(gpu_bmus, cp.ndarray) and isinstance(gpu_distances, cp.ndarray)
    np.testing.assert_array_equal(host_bmus, cp.asnumpy(gpu_bmus))
    np.testing.assert_allclose(host_distances, cp.asnumpy(gpu_distances), rtol=1e-6)


def test_dataframe_gets_a_label_column(som, frame):
    labelled = frame.copy()
    labelled["node"] = som.predict(frame)
    assert labelled["node"].between(0, 15).all()


# --- infer reads files and the same inputs as predict ------------------------


def write_npy(folder, data):
    path = folder / "cells.npy"
    np.save(path, data)
    return path


def write_zarr(folder, data):
    path = folder / "cells.zarr"
    handle = create_sharded_array(path, array_key="data", shape=data.shape, dtype="f4")
    handle.array[:] = data
    return path


@pytest.mark.parametrize("writer", [write_npy, write_zarr])
@pytest.mark.parametrize("as_text", [True, False])  # str path; pathlib.Path
@pytest.mark.parametrize("use_cuda_streams", [True, False])
def test_infer_reads_a_file(tmp_path, som, data, writer, as_text, use_cuda_streams):
    path = writer(tmp_path, data)
    labels = som.infer(str(path) if as_text else path, chunk_size=128, use_cuda_streams=use_cuda_streams)
    np.testing.assert_array_equal(np.asarray(labels), som.predict(data))


def test_infer_accepts_a_dataframe_and_a_list(som, data, frame):
    expected = som.predict(data)
    np.testing.assert_array_equal(np.asarray(som.infer(frame, chunk_size=128)), expected)
    np.testing.assert_array_equal(np.asarray(som.infer(data.tolist(), chunk_size=128)), expected)


# --- the model keeps the column names of a DataFrame -------------------------


def test_train_som_stores_the_column_names(named_som, som):
    assert named_som.feature_names == COLUMNS
    assert som.feature_names is None


def test_predict_refuses_columns_in_another_order(named_som, frame):
    reordered = frame[COLUMNS[::-1]]
    with pytest.raises(ValueError, match="order"):
        named_som.predict(reordered)
    with pytest.raises(ValueError, match="order"):
        named_som.infer(reordered, chunk_size=128)
    np.testing.assert_array_equal(
        named_som.predict(reordered[named_som.feature_names]), named_som.predict(frame)
    )
    # An array has no names, so only the column count is checked.
    np.testing.assert_array_equal(named_som.predict(frame.to_numpy()), named_som.predict(frame))


def test_predict_names_a_missing_column(named_som, frame):
    renamed = frame.rename(columns={"CD8": "CD8a"})
    with pytest.raises(ValueError, match="CD8a") as error:
        named_som.predict(renamed)
    assert "CD8" in str(error.value)


def test_save_and_load_keep_the_column_names(tmp_path, named_som, som, frame):
    loaded = floatsom.FloatSOM.load(named_som.save(tmp_path / "named.floatsom"))
    assert loaded.feature_names == COLUMNS
    with pytest.raises(ValueError, match="order"):
        loaded.predict(frame[COLUMNS[::-1]])
    assert floatsom.FloatSOM.load(som.save(tmp_path / "plain.floatsom")).feature_names is None


# --- a file path in train_som or predict gives a clear error -----------------


@pytest.mark.parametrize("path", ["cells.csv", Path("cells.csv")])
def test_train_som_explains_a_file_path(path):
    with pytest.raises(ValueError, match="read_csv"):
        floatsom.train_som(path)


def test_predict_explains_a_file_path(som):
    with pytest.raises(ValueError, match="infer"):
        som.predict("cells.npy")


# --- verbose=True shows the training progress ---------------------------------


def progress_records(caplog):
    return [r for r in caplog.records if r.name.startswith("floatsom") and r.levelno == logging.INFO]


def test_verbose_logs_training_progress(caplog, data):
    floatsom.train_som(data, grid_size=3, iterations=2, seed=1)
    assert not progress_records(caplog), "verbose=False must log no progress"
    floatsom_logger = logging.getLogger("floatsom")
    old_level, old_handlers = floatsom_logger.level, list(floatsom_logger.handlers)
    try:
        floatsom.train_som(data, grid_size=3, iterations=2, seed=1, verbose=True)
        assert progress_records(caplog), "verbose=True gave no INFO log records"
    finally:
        floatsom_logger.setLevel(old_level)
        floatsom_logger.handlers[:] = old_handlers


def test_progress_stops_after_the_verbose_call(caplog, tmp_path, som, data):
    floatsom_logger = logging.getLogger("floatsom")
    old_level, old_handlers = floatsom_logger.level, list(floatsom_logger.handlers)
    path = write_npy(tmp_path, data)
    try:
        floatsom.train_som(data, grid_size=3, iterations=2, seed=1, verbose=True)
        caplog.clear()
        som.infer(path, chunk_size=128)
        assert not progress_records(caplog), "a quiet call after a verbose call logged progress"
    finally:
        floatsom_logger.setLevel(old_level)
        floatsom_logger.handlers[:] = old_handlers


def test_infer_verbose_shows_progress(caplog, som, data):
    floatsom_logger = logging.getLogger("floatsom")
    old_level, old_handlers = floatsom_logger.level, list(floatsom_logger.handlers)
    try:
        som.infer(data, chunk_size=128, verbose=True)
        assert progress_records(caplog), "infer(verbose=True) gave no INFO log records"
    finally:
        floatsom_logger.setLevel(old_level)
        floatsom_logger.handlers[:] = old_handlers


# --- integer column names (pd.DataFrame(array)) stay integers ----------------


def test_integer_column_names_select_the_columns(tmp_path, data):
    frame = pd.DataFrame(data)                     # column labels 0, 1, 2, 3
    named = floatsom.train_som(frame, grid_size=3, iterations=5, seed=2)
    assert named.feature_names == [0, 1, 2, 3]
    reordered = frame[[3, 2, 1, 0]]
    with pytest.raises(ValueError, match="order"):
        named.predict(reordered)
    np.testing.assert_array_equal(named.predict(reordered[named.feature_names]), named.predict(frame))
    loaded = floatsom.FloatSOM.load(named.save(tmp_path / "integer_names.floatsom"))
    assert loaded.feature_names == [0, 1, 2, 3]
