"""Make a 2-D float32 feature matrix from user data, with clear errors."""

import os
from typing import List, Optional, Union

import cupy as cp
import numpy as np


def as_feature_matrix(data, caller: str):
    """Return ``data`` as a 2-D float32 NumPy or CuPy array.

    ``data`` can be a NumPy array, a CuPy array, a pandas DataFrame or a
    nested list. A CuPy array stays on the GPU. ``caller`` names the public
    function in the error messages.
    """
    if isinstance(data, (str, os.PathLike)):
        raise ValueError(
            f"{caller} needs an array or a DataFrame, not a file path ({os.fspath(data)}). "
            "Read the file first, for example df = pandas.read_csv(path). "
            "To label a large file in chunks, use som.infer(path)."
        )
    if _is_dataframe(data):
        data = _dataframe_values(data, caller)
    if isinstance(data, cp.ndarray):
        matrix = data.astype(cp.float32, copy=False)
    else:
        try:
            matrix = np.asarray(data, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{caller} needs numeric data, but the values cannot be converted "
                f"to numbers ({error})."
            ) from error
    if matrix.ndim != 2:
        raise ValueError(
            f"{caller} needs a 2-D array with shape (n_samples, n_features), "
            f"but got {matrix.ndim}-D data with shape {tuple(matrix.shape)}. "
            "For one feature, use data.reshape(-1, 1). For one sample, use data.reshape(1, -1)."
        )
    if matrix.shape[0] == 0:
        raise ValueError(f"{caller} got data with no rows (samples).")
    if matrix.shape[1] == 0:
        raise ValueError(f"{caller} got data with no columns (features).")
    return matrix


def feature_names_of(data) -> Optional[List[Union[str, int]]]:
    """Return the column labels of a DataFrame, or None for other data.

    A str or int label stays as it is, so data[names] selects the columns. Other labels
    (for example tuples) become str, so that the model file can store them in JSON.
    """
    if not _is_dataframe(data):
        return None
    return [label if isinstance(label, (str, int)) else str(label) for label in data.columns.tolist()]


def _is_dataframe(data) -> bool:
    return hasattr(data, "columns") and hasattr(data, "to_numpy")


def _dataframe_values(frame, caller: str):
    import pandas as pd

    bad = [
        str(column)
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if bad:
        raise ValueError(
            f"{caller} needs numeric columns. Remove or encode these columns: {', '.join(bad)}."
        )
    return frame.to_numpy(dtype=np.float32)


def count_rows_not_finite(matrix) -> int:
    """Return the number of rows that have a NaN or infinite value."""
    xp = cp.get_array_module(matrix)
    return int((~xp.isfinite(matrix)).any(axis=1).sum())
