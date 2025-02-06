import numpy as np
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt


class ProblemInputs:
    def __init__(
        self,
        fused_data: list[np.ndarray],
        fused_constants: list[np.ndarray] | None = None,
        fused_quadratics: list[np.ndarray] | None = None,
        global_constants: list[np.ndarray] | None = None,
    ) -> None:
        self.fused_data = fused_data
        self.fused_constants = fused_constants
        self.fused_quadratics = fused_quadratics
        self.global_constants = global_constants


@dataclass
class NetworkGraph:
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
        if len(self.nb_sensors_per_node) != self.nb_nodes:
            raise ValueError(
                "The number of sensors per node does not match the number of nodes."
            )

    def plot_graph(self) -> None:
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
    max_iterations: int = 100
    objective_tolerance: float = 1e-6
    argument_tolerance: float = 1e-6


class ProblemParameters:
    def __init__(
        self,
        nb_filters: int,
        nb_samples: int,
        network_graph: NetworkGraph,
    ) -> None:
        self.nb_filters = nb_filters
        self.nb_samples = nb_samples
        self.network_graph = network_graph
