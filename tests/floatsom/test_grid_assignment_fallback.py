"""
Contract tests for the grid assignment fallback without RAPIDS.

RAPIDS (cuDF, cuGraph, cuML) has no AMD build. When the RAPIDS import
fails, grid assignment uses scikit-learn PCA, a networkx graph and the
networkx ForceAtlas2 layout. The CPU tests use NumPy and SciPy only.
The GPU tests need CuPy. They skip when CuPy has no usable GPU.
"""

import logging
import sys

import numpy as np
import pytest
from scipy.spatial import Delaunay

from floatsom.topology import grid_layout_backends

from ..conftest import GPU_AVAILABLE

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires CuPy with a usable GPU")

# A unit square with one center point. The Delaunay triangulation is
# unique: four triangles meet at the center and no diagonal exists.
SQUARE_WITH_CENTER = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
    dtype=np.float32,
)
SQUARE_WITH_CENTER_EDGES = {
    (0, 1), (0, 2), (1, 3), (2, 3),
    (0, 4), (1, 4), (2, 4), (3, 4),
}


def _random_points(n_points=40, seed=7):
    rng = np.random.default_rng(seed)
    return rng.random((n_points, 2)).astype(np.float32)


def _scipy_neighbor_edges(tri):
    indptr, indices = tri.vertex_neighbor_vertices
    edges = set()
    for vertex in range(len(indptr) - 1):
        for neighbor in indices[indptr[vertex]:indptr[vertex + 1]]:
            edges.add((min(vertex, int(neighbor)), max(vertex, int(neighbor))))
    return edges


def _edge_set(edges):
    return {(int(a), int(b)) for a, b in np.asarray(edges)}


class TestRapidsImportGuard:
    def test_import_rapids_returns_none_when_cudf_import_fails(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cudf", None)
        assert grid_layout_backends.import_rapids() is None

    def test_import_rapids_returns_none_when_cuml_import_fails(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cuml", None)
        monkeypatch.setitem(sys.modules, "cuml.decomposition", None)
        assert grid_layout_backends.import_rapids() is None


class TestDelaunayEdges:
    def test_square_with_center_has_expected_edges(self):
        tri = Delaunay(SQUARE_WITH_CENTER)
        edges = grid_layout_backends.delaunay_edges(tri.simplices, np)

        assert edges.shape == (len(SQUARE_WITH_CENTER_EDGES), 2)
        assert _edge_set(edges) == SQUARE_WITH_CENTER_EDGES

    def test_edges_match_scipy_neighbors_and_are_sorted_and_unique(self):
        tri = Delaunay(_random_points())
        edges = np.asarray(grid_layout_backends.delaunay_edges(tri.simplices, np))

        assert _edge_set(edges) == _scipy_neighbor_edges(tri)
        assert np.all(edges[:, 0] < edges[:, 1])
        assert len(np.unique(edges, axis=0)) == len(edges)


class TestNetworkxGraph:
    def test_graph_has_the_delaunay_edges_without_weights(self):
        tri = Delaunay(SQUARE_WITH_CENTER)
        edges = grid_layout_backends.delaunay_edges(tri.simplices, np)
        graph = grid_layout_backends.build_networkx_graph(edges, len(SQUARE_WITH_CENTER))

        assert not graph.is_directed()
        assert graph.number_of_edges() == len(SQUARE_WITH_CENTER_EDGES)
        assert {tuple(sorted(edge)) for edge in graph.edges()} == SQUARE_WITH_CENTER_EDGES
        assert all(not data for _, _, data in graph.edges(data=True))

    def test_graph_keeps_every_node_in_index_order(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        graph = grid_layout_backends.build_networkx_graph(edges, 5)

        assert list(graph.nodes()) == [0, 1, 2, 3, 4]
        assert graph.number_of_edges() == 2


class TestForceAtlas2Fallback:
    @staticmethod
    def _graph_and_positions():
        points = _random_points(n_points=30, seed=3)
        tri = Delaunay(points)
        edges = grid_layout_backends.delaunay_edges(tri.simplices, np)
        graph = grid_layout_backends.build_networkx_graph(edges, len(points))
        return graph, points

    def test_layout_returns_finite_float32_positions_for_every_node(self):
        graph, points = self._graph_and_positions()
        positions = grid_layout_backends.force_atlas2_networkx(graph, points, seed=0)

        assert positions.shape == (len(points), 2)
        assert positions.dtype == np.float32
        assert np.all(np.isfinite(positions))
        assert not np.allclose(positions, points)

    def test_layout_puts_graph_neighbors_closer_than_other_nodes(self):
        import networkx as nx

        # A 6 x 6 grid graph that starts from mixed positions.
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 6), ordering="sorted")
        start = np.random.default_rng(3).random((graph.number_of_nodes(), 2))
        positions = grid_layout_backends.force_atlas2_networkx(graph, start, seed=0)

        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        adjacent = nx.to_numpy_array(graph, nodelist=range(graph.number_of_nodes())) > 0
        others = ~adjacent & ~np.eye(len(positions), dtype=bool)
        assert distances[adjacent].mean() < 0.5 * distances[others].mean()

    def test_layout_does_not_change_the_initial_positions(self):
        graph, points = self._graph_and_positions()
        original = points.copy()
        grid_layout_backends.force_atlas2_networkx(graph, points, seed=0)

        np.testing.assert_array_equal(points, original)

    def test_layout_uses_the_cugraph_parameters(self, monkeypatch):
        import networkx as nx

        calls = []
        real_layout = nx.forceatlas2_layout

        def recording_layout(*args, **kwargs):
            calls.append(kwargs)
            return real_layout(*args, **kwargs)

        monkeypatch.setattr(nx, "forceatlas2_layout", recording_layout)
        graph, points = self._graph_and_positions()
        grid_layout_backends.force_atlas2_networkx(graph, points, seed=5)

        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["max_iter"] == 500
        assert kwargs["distributed_action"] is True
        assert kwargs["linlog"] is False
        assert kwargs["jitter_tolerance"] == 1.0
        assert kwargs["scaling_ratio"] == 2.0
        assert kwargs["strong_gravity"] is False
        assert kwargs["gravity"] == 1.0
        assert kwargs["weight"] is None
        assert kwargs["seed"] == 5
        np.testing.assert_allclose(
            np.array([kwargs["pos"][node] for node in range(len(points))]), points
        )


class TestPcaFallback:
    @staticmethod
    def _weights():
        rng = np.random.default_rng(13)
        scales = np.array([5.0, 3.0, 1.0, 0.5, 0.2, 0.1], dtype=np.float32)
        return (rng.standard_normal((60, 6)) * scales + 2.0).astype(np.float32)

    def test_projection_matches_numpy_svd_up_to_sign(self):
        weights = self._weights()
        projected = grid_layout_backends.pca_project_2d(weights)

        centered = weights.astype(np.float64) - weights.astype(np.float64).mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        reference = centered @ vt[:2].T

        assert projected.shape == (len(weights), 2)
        assert projected.dtype == np.float32
        for column in range(2):
            same = np.allclose(projected[:, column], reference[:, column], atol=1e-4)
            flipped = np.allclose(projected[:, column], -reference[:, column], atol=1e-4)
            assert same or flipped, f"column {column} differs from the SVD reference"

    def test_projection_uses_the_cuml_sign_rule(self):
        # cuML fit_transform makes the largest absolute value of each
        # output column positive.
        projected = grid_layout_backends.pca_project_2d(self._weights())
        rows = np.argmax(np.abs(projected), axis=0)

        assert np.all(projected[rows, np.arange(2)] > 0)


@requires_gpu
class TestGridAssignerWithoutRapids:
    @staticmethod
    def _weights(num_nodes=25, input_dim=4):
        import cupy as cp

        rng = np.random.default_rng(21)
        return cp.asarray(rng.standard_normal((num_nodes, input_dim)).astype(np.float32))

    @pytest.mark.parametrize("assigner_name", ["RegularGridAssigner", "HexagonalGridAssigner"])
    def test_assign_nodes_to_grid_uses_the_fallback(self, monkeypatch, caplog, assigner_name):
        from floatsom.topology import grid_assignment

        monkeypatch.setattr(grid_layout_backends, "import_rapids", lambda: None)
        assigner = getattr(grid_assignment, assigner_name)(delaunay_backend="local")

        with caplog.at_level(logging.INFO, logger=grid_assignment.__name__):
            node_to_grid, adjacency = assigner.assign_nodes_to_grid(self._weights(), 25)

        assert sorted(node_to_grid) == list(range(25))
        assert len(set(node_to_grid.values())) == 25
        for node, neighbors in adjacency.items():
            for neighbor in neighbors:
                assert node in adjacency[neighbor]
        fallback_messages = [r for r in caplog.records if "RAPIDS" in r.getMessage()]
        assert len(fallback_messages) == 1

    def test_reform_to_grid_runs_end_to_end_without_rapids(self, monkeypatch):
        import cupy as cp

        from floatsom.base.floatsom_factories import create_floatsom
        from floatsom.floatsom_params import FloatSOMParams, ProcessingConfig, TopologyConfig

        monkeypatch.setattr(grid_layout_backends, "import_rapids", lambda: None)
        rng = np.random.default_rng(42)
        data = cp.asarray(rng.standard_normal((400, 4)).astype(np.float32))
        params = FloatSOMParams(
            input_dim=4,
            total_iterations=5,
            initial_learning_rate=0.5,
            initial_radius=2.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            topology_config=TopologyConfig(topology_type="mst", num_nodes=25),
            processing_config=ProcessingConfig(chunk_size=100, method="batch"),
            reform_grid=True,
            reform_grid_type="regular",
            reform_delaunay_backend="local",
        )
        som = create_floatsom(params)

        stats = som.train(data)

        assert stats["topology_reformed"] is True
        assert sorted(som.grid_node_mapping) == list(range(25))
        assert len(set(som.grid_node_mapping.values())) == 25
        assert som.topology.adjacency_list is som.adjacency_list
