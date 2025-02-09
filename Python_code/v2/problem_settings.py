import numpy as np
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProblemInputs:
    def __init__(
        self,
        fused_data: list[np.ndarray],
        fused_constants: list[np.ndarray] | None = None,
        fused_quadratics: list[np.ndarray] | None = None,
        global_parameters: list[np.ndarray] | None = None,
    ) -> None:
        self.fused_data = fused_data
        self.fused_constants = fused_constants
        self.fused_quadratics = fused_quadratics
        self.global_parameters = global_parameters


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
    objective_tolerance: float | None = None
    argument_tolerance: float | None = None


@dataclass
class DataParameters:
    nb_samples: int
    window_length: int | None = None
    nb_window_reuse: int = 1
    sliding_window_offset: int | None = None

    def __post_init__(self) -> None:
        if self.window_length is None:
            self.window_length = self.nb_samples
        if self.sliding_window_offset is None:
            self.sliding_window_offset = self.window_length


class DataSegmenter:
    def __init__(self, data: list[np.ndarray], data_parameters: DataParameters) -> None:
        self.data = data
        self.data_parameters = data_parameters
        self.end_of_data_reached = [False] * len(data)

    def get_window(self, window_id) -> list[np.ndarray]:
        start = window_id * self.data_parameters.sliding_window_offset
        end = start + self.data_parameters.window_length
        nb_data = len(self.data)
        data_window = []
        for ind in range(nb_data):
            data_ind = self.data[ind]
            if start > np.size(data_ind, 1) - self.data_parameters.window_length:
                if self.end_of_data_reached[ind] is False:
                    self.end_of_data_reached[ind] = True
                    logger.warning(
                        f"Reached end of data at window {window_id}, last window will be reused"
                    )
                start = np.size(data_ind, 1) - self.data_parameters.window_length
            data_window.append(data_ind[:, start:end])

        return data_window
