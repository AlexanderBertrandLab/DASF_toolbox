import numpy as np


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


class ConvergenceParameters:
    def __init__(
        self,
        max_iterations: int,
        objective_tolerance: float,
        argument_tolerance: float,
    ) -> None:
        self.max_iterations = max_iterations
        self.objective_tolerance = objective_tolerance
        self.argument_tolerance = argument_tolerance


class ProblemParameters:
    def __init__(
        self,
        nb_filters: int,
        nb_samples: int,
        network_graph: NetworkGraph,
        convergence_parameters: ConvergenceParameters,
    ) -> None:
        self.nb_filters = nb_filters
        self.nb_samples = nb_samples
        self.network_graph = network_graph
        self.convergence_parameters = convergence_parameters
