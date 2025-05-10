import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProblemInputs:
    """
    Class storing the inputs, i.e., the data, of the problem.

    Attributes
    ----------
    fused_signals : list[np.ndarray]
        List of signals to be fused by the variable.
    fused_constants : list[np.ndarray] | None = None
        List of constant arrays to be fused by the variable. Similar to the signals, but do not change in time.
    fused_quadratics : list[np.ndarray] | None = None
        List of block-diagonal arrays to be fused by the variable from both sides, e.g., X.T @ M @ X.
    global_parameters : list[np.ndarray] | None = None
        List of arrays that are not fused by the variable, but are needed to solve the optimization problem.
    """

    def __init__(
        self,
        fused_signals: list[np.ndarray],
        fused_constants: list[np.ndarray] | None = None,
        fused_quadratics: list[np.ndarray] | None = None,
        global_parameters: list[np.ndarray] | None = None,
    ) -> None:
        self.fused_signals = fused_signals
        self.fused_constants = fused_constants
        self.fused_quadratics = fused_quadratics
        self.global_parameters = global_parameters


@dataclass
class NetworkGraph:
    """
    Class storing the properties of the network graph.

    Attributes
    ----------
    nb_nodes : int
        Number of nodes in the network.
    nb_sensors_per_node : np.ndarray
        Array storing the number of sensors per node in the network.
    adjacency_matrix : np.ndarray
        Adjacency matrix of the network.
    nb_sensors_total : int
        Total number of sensors in the network.
    """

    def __init__(
        self,
        nb_nodes: int,
        nb_sensors_per_node: np.ndarray,
        adjacency_matrix: np.ndarray,
    ) -> None:
        self.nb_nodes = nb_nodes
        self.nb_sensors_per_node = nb_sensors_per_node
        self.adjacency_matrix = adjacency_matrix
        self.nb_sensors_total = np.sum(nb_sensors_per_node)

        if self.adjacency_matrix.shape != (self.nb_nodes, self.nb_nodes):
            raise ValueError("The adjacency matrix does not have the correct shape.")
        if np.any(self.adjacency_matrix != self.adjacency_matrix.T):
            raise ValueError("The adjacency matrix must be symmetric.")
        if np.diag(self.adjacency_matrix).any() != 0:
            np.fill_diagonal(self.adjacency_matrix, 0)
            logger.warning(
                "The adjacency matrix is expected to be a hollow matrix. Its diagonal is now set to 0."
            )

        if len(self.nb_sensors_per_node) != self.nb_nodes:
            raise ValueError(
                "The number of sensors per node does not match the number of nodes."
            )

    def plot_graph(self) -> None:
        """Plots the adjacency matrix and the graph of the network."""
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.adjacency_matrix)
        ax[0].set_xticks(np.arange(0, self.nb_nodes, 1))
        ax[0].set_yticks(np.arange(0, self.nb_nodes, 1))
        ax[0].set_xticklabels(np.arange(1, self.nb_nodes + 1, 1))
        ax[0].set_yticklabels(np.arange(1, self.nb_nodes + 1, 1))
        ax[0].set_xticks(np.arange(-0.5, self.nb_nodes, 1), minor=True)
        ax[0].set_yticks(np.arange(-0.5, self.nb_nodes, 1), minor=True)
        ax[0].grid(which="minor", color="w", linestyle="-", linewidth=2)
        ax[0].set_title("Adjacency matrix")

        graph = nx.from_numpy_array(self.adjacency_matrix)
        nx.draw_circular(graph)
        ax[1].set_title("Graph of the network")
        ax[1].axis("equal")

        return fig


@dataclass
class ConvergenceParameters:
    """
    Class storing the convergence parameters of an optimization problem. This class is used both by the centralized solver and the DASF solver.

    Attributes
    ----------
    max_iterations : int | None
        Maximum number of iterations the solver applies.
    objective_tolerance : float | None
        Threshold for two consecutive objective function values. If the absolute difference is below this value, the solver stops.
    argument_tolerance : float | None
        Threshold for two consecutive iterates. If the norm of the difference is below this value, the solver stops.

    Note
    -----
    If all parameters are None, the default value for `max_iterations` is set to a default value of 100.
    """

    max_iterations: int | None = None
    objective_tolerance: float | None = None
    argument_tolerance: float | None = None

    def __post_init__(self) -> None:
        if (
            self.max_iterations is None
            and self.objective_tolerance is None
            and self.argument_tolerance is None
        ):
            self.max_iterations = 100
            logger.warning(
                f"No convergence conditions specified, setting max iterations to {self.max_iterations}"
            )
