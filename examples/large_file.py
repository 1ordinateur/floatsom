"""Label a large file in chunks, and write the labels to Zarr: python examples/large_file.py.

The example makes a small Zarr file. For your data, change a CSV file to Zarr
once, as the user guide shows, and then use its path.
"""
import tempfile
from pathlib import Path

import numpy as np
import zarr

import floatsom


def main():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(200_000, 10)).astype(np.float32)

    with tempfile.TemporaryDirectory() as folder:
        data_path = Path(folder) / "cells.zarr"
        zarr.save_array(str(data_path), data)

        som = floatsom.train_som(data[:20_000], grid_size=10, seed=42)    # train on a part

        labels_path = Path(folder) / "labels.zarr"
        som.infer(data_path, output_path=labels_path, chunk_size=50_000)
        labels = zarr.open_group(str(labels_path), mode="r")["assignments"][:]

        same = np.array_equal(labels, som.predict(data))
        print(f"Labelled {len(labels)} rows from {data_path.name}; the same as predict: {same}.")


if __name__ == "__main__":
    main()
