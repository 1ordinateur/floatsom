"""train_som defaults and input checks that run before GPU work (no GPU needed)."""

import numpy as np
import pandas as pd
import pytest

from floatsom import suggest_grid_size, train_som


@pytest.mark.parametrize(
    "n_samples,expected",
    [
        (1, 2),  # minimum side
        (100, 7),  # round(sqrt(5 * sqrt(100))) = round(7.07)
        (2000, 15),  # round(sqrt(5 * sqrt(2000))) = round(14.95)
        (40_000, 32),  # round(sqrt(1000)) = 31.6 -> 32
        (10**8, 32),  # maximum side
    ],
)
def test_suggest_grid_size_uses_som_toolbox_rule(n_samples, expected):
    assert suggest_grid_size(n_samples) == expected


def test_suggest_grid_size_rejects_no_samples():
    with pytest.raises(ValueError, match="at least one sample"):
        suggest_grid_size(0)


@pytest.mark.parametrize(
    "data,message",
    [
        (np.arange(10, dtype=np.float32), "2-D"),
        (np.zeros((0, 4), dtype=np.float32), "no rows"),
        (np.zeros((5, 0), dtype=np.float32), "no columns"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), "NaN or infinite"),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), "NaN or infinite"),
        ([["a", "b"], ["c", "d"]], "numeric"),
    ],
)
def test_train_som_rejects_bad_arrays(data, message):
    with pytest.raises(ValueError, match=message):
        train_som(data)


def test_train_som_names_non_numeric_dataframe_columns():
    frame = pd.DataFrame({"CD4": [1.0, 2.0], "sample": ["a", "b"], "CD8": [3.0, 4.0]})
    with pytest.raises(ValueError) as error:
        train_som(frame)
    message = str(error.value)
    assert "sample" in message
    assert "CD4" not in message


def test_train_som_counts_rows_with_missing_values():
    data = np.ones((10, 3), dtype=np.float32)
    data[[2, 7], 1] = np.nan
    with pytest.raises(ValueError, match="2 of 10 rows"):
        train_som(data)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"grid_size": 1}, "grid_size"),
        ({"iterations": 0}, "iterations"),
        ({"topology": "triangle"}, "topology"),
        ({"distance_metric": "chebyshev"}, "distance_metric"),
        ({"profile": "unknown"}, "profile"),
    ],
)
def test_train_som_rejects_bad_options(kwargs, message):
    data = np.ones((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        train_som(data, **kwargs)
