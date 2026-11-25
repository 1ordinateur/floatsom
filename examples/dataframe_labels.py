"""CSV file in, node and metacluster columns out: python examples/dataframe_labels.py.

The example makes a small CSV file with synthetic cytometry-like data. Use your
own file and marker names in place of it.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import floatsom

MARKERS = ["CD3", "CD4", "CD8", "CD19", "CD56"]


def make_csv(path):
    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 2000, size=(6, len(MARKERS)))
    groups = rng.integers(0, len(centres), size=5000)
    values = np.abs(centres[groups] + rng.normal(scale=150, size=(len(groups), len(MARKERS))))
    cells = pd.DataFrame(values, columns=MARKERS)
    cells.insert(0, "sample", "donor_1")              # a column that is not a marker
    cells.to_csv(path, index=False)


def main():
    with tempfile.TemporaryDirectory() as folder:
        csv_path = Path(folder) / "cells.csv"
        make_csv(csv_path)

        cells = pd.read_csv(csv_path)
        features = np.arcsinh(cells[MARKERS] / 5)     # cytometry transform, cofactor 5

        som = floatsom.train_som(features, grid_size=10, seed=42)
        cells["node"] = som.predict(features)

        node_metacluster = AgglomerativeClustering(n_clusters=6).fit_predict(som.get_weights())
        cells["metacluster"] = node_metacluster[cells["node"]]

        labelled_path = Path(folder) / "cells_labelled.csv"
        cells.to_csv(labelled_path, index=False)
        print(cells.head())
        print(f"Cells in each metacluster:\n{cells['metacluster'].value_counts().sort_index()}")
        print(f"The model keeps the marker names: {som.feature_names}")


if __name__ == "__main__":
    main()
