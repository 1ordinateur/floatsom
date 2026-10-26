"""
Graph, layout and PCA backends for grid assignment.

RAPIDS (cuDF, cuGraph, cuML) is the first choice. RAPIDS has no AMD
build. When the RAPIDS import fails, grid assignment uses the CPU
fallbacks in this module: scikit-learn PCA, a networkx graph and the
networkx ForceAtlas2 layout. This module does not import CuPy, so the
fallbacks also run without a GPU.
"""

from typing import NamedTuple, Optional

import numpy as np


class RapidsModules(NamedTuple):
    """The RAPIDS modules that grid assignment uses."""

    cudf: object
    cugraph: object
    PCA: type


def import_rapids() -> Optional[RapidsModules]:
    """Return the RAPIDS modules, or None when one of the imports fails."""
    try:
        import cudf
        import cugraph
        from cuml.decomposition import PCA
    except ImportError:
        return None
    return RapidsModules(cudf=cudf, cugraph=cugraph, PCA=PCA)


def delaunay_edges(simplices, xp):
    """
    Return the unique undirected edges of a 2D Delaunay triangulation.

    Args:
        simplices: Triangle vertex indices (n_triangles, 3)
        xp: The array module of ``simplices`` (NumPy or CuPy)

    Returns:
        Edges (n_edges, 2) with the lower vertex index first, in sorted order
    """
    edges = xp.vstack([
        xp.stack([simplices[:, i], simplices[:, (i + 1) % 3]], axis=1)
        for i in range(3)
    ])
    return xp.unique(xp.sort(edges, axis=1), axis=0)


def build_networkx_graph(edges: np.ndarray, num_nodes: int):
    """
    Return an undirected networkx graph without edge weights.

    The graph contains the nodes 0 to num_nodes - 1 in index order, also
    the nodes that have no edge.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(src), int(dst)) for src, dst in np.asarray(edges))
    return graph


def force_atlas2_networkx(graph, initial_pos: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Apply the networkx ForceAtlas2 layout with the parameters of the cuGraph call.

    Parameter map from cugraph.layout.force_atlas2:
        max_iter=500                          -> max_iter=500
        pos_list                              -> pos
        outbound_attraction_distribution=True -> distributed_action=True
        lin_log_mode=False                    -> linlog=False
        jitter_tolerance=1.0                  -> jitter_tolerance=1.0
        scaling_ratio=2.0                     -> scaling_ratio=2.0
        strong_gravity_mode=False             -> strong_gravity=False
        gravity=1.0                           -> gravity=1.0
        edge_weight_influence=1.0             -> weight=None (the graph has no weights)

    Differences from cuGraph (the result is a similar layout, not a numerical match):
        - cuGraph outbound_attraction_distribution multiplies the attraction
          by the mean node mass before it divides by the node mass. networkx
          distributed_action only divides by the node mass. Thus the networkx
          attraction is weaker relative to repulsion and gravity (about 7x
          for a Delaunay graph).
        - The networkx speed control (swing and traction) is not the same as
          the Gephi and cuGraph speed control.
        - networkx can stop before max_iter when the update is very small.
        - networkx has no Barnes-Hut option. It calculates the exact
          repulsion between all node pairs (barnes_hut_optimize and
          barnes_hut_theta have no equivalent).
        - The networkx gravity pulls the nodes to their mean position.
          The cuGraph gravity pulls the nodes to the origin.
        - The time and the memory increase as num_nodes ** 2. On one CPU node,
          1024 nodes take about 40 s and 4096 nodes take about 10 min. 10,000
          nodes take about 1 h and about 8 GB of RAM.
        - The seed has an effect only on nodes without an initial position.
          This function gives an initial position to each node.

    Args:
        graph: networkx graph with the nodes 0 to len(initial_pos) - 1
        initial_pos: Initial 2D positions (num_nodes, 2)
        seed: Random seed for networkx

    Returns:
        Positions (num_nodes, 2) as float32, in node index order
    """
    import networkx as nx

    initial = np.array(initial_pos, dtype=np.float64)
    layout = nx.forceatlas2_layout(
        graph,
        pos={node: initial[node].copy() for node in range(len(initial))},
        max_iter=500,
        jitter_tolerance=1.0,
        scaling_ratio=2.0,
        gravity=1.0,
        distributed_action=True,
        strong_gravity=False,
        weight=None,
        linlog=False,
        seed=seed,
    )
    return np.array([layout[node] for node in range(len(initial))], dtype=np.float32)


def pca_project_2d(weights: np.ndarray) -> np.ndarray:
    """
    Project weights to 2D with scikit-learn PCA.

    cuML PCA.fit_transform makes the largest absolute value of each output
    column positive. This function applies the same sign rule. When two
    values of a column have almost the same absolute value, small float
    differences can still give a different sign than cuML.

    Args:
        weights: Weights (num_nodes, input_dim)

    Returns:
        2D projection (num_nodes, 2), with the dtype of the input for float32
    """
    from sklearn.decomposition import PCA

    projected = PCA(n_components=2, svd_solver="full").fit_transform(weights)
    rows = np.argmax(np.abs(projected), axis=0)
    flip = projected[rows, np.arange(projected.shape[1])] < 0
    projected[:, flip] *= -1
    return projected
