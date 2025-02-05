import numpy as np
from dataclasses import dataclass


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


class NetworkGraph:
    def __init__(
        self,
        nb_nodes: int,
        nb_sensors_per_node: np.ndarray,
        adjacency_matrix: np.ndarray,
        updating_order: np.ndarray,
    ) -> None:
        self.nb_nodes = nb_nodes
        self.nb_sensors_per_node = nb_sensors_per_node
        self.adjacency_matrix = adjacency_matrix
        self.updating_order = updating_order
        self.nb_sensors_total = np.sum(nb_sensors_per_node)


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
