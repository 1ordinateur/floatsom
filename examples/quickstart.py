"""Small single-GPU example: python examples/quickstart.py."""
import tempfile
from pathlib import Path

import numpy as np

import floatsom


def main():
    data = np.random.default_rng(42).normal(size=(2000, 8)).astype(np.float32)

    som = floatsom.train_som(data, seed=42)
    clusters = som.predict(data).get()
    print(f"Assigned {len(clusters)} observations to {som.get_weights().shape[0]} SOM nodes.")

    with tempfile.TemporaryDirectory() as folder:
        path = som.save(Path(folder) / "quickstart.floatsom")
        loaded = floatsom.FloatSOM.load(path)
        same = np.array_equal(loaded.predict(data).get(), clusters)
        print(f"Saved and loaded the model; the same assignments: {same}.")


if __name__ == "__main__":
    main()
