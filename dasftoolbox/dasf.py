from __future__ import annotations
import numpy as np
from dasftoolbox.problem_settings import (
    NetworkGraph,
    ProblemInputs,
    ConvergenceParameters,
)
from dasftoolbox.data_retrievers.data_retriever import DataRetriever
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import logging
from dataclasses import dataclass, replace
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DynamicPlotParameters:
    """
    Parameters for dynamic plots.

    Attributes
    ----------
    tau : int
        Sampling period for the dynamic plot for :math:`X^T\\mathbf{y}(t)`. One sample every `tau` will be shown.
    show_x : bool
        Flag to show the current estimate of :math:`X`, i.e., the filter weights.
    show_xTY : bool
        Flag to show the current estimate of :math:`X^T\\mathbf{y}(t)`, i.e., the filtered signal, per sample.
    X_col : int
        Index of the column of :math:`X` (i.e., the filter) to be plotted.
    XTY_col : int
        Index of the column of :math:`X^T\\mathbf{y}(t)` (i.e., the filtered signal) to be plotted for the filtered signal.

    """

    tau: int = 10
    show_x: bool = True
    show_xTY: bool = True
    X_col: int = 0
    XTY_col: int = 0
    Y_id: int = 0

    def apply_correction(
        self, nb_filters: int, nb_samples: int
    ) -> DynamicPlotParameters:
        """
        Function ensuring that parameters are set correctly, and making modifications if necessary.
        """
        new_values = self.__dict__.copy()

        if self.tau > nb_samples / 2:
            new_values["tau"] = 10
            logger.warning(
                "Subsample value exceeds the total number of samples, setting it to 10"
            )
        if self.X_col >= nb_filters:
            new_values["X_col"] = nb_filters - 1
            logger.warning(
                f"Column to show for X exceeds total number of columns, setting it to {nb_filters - 1}"
            )
        if self.XTY_col >= nb_filters:
            new_values["XTY_col"] = nb_filters - 1
            logger.warning(
                f"Filtered signal index exceeds total number of filtered signals, setting it to {nb_filters - 1}"
            )
        if self.show_x is False and self.show_xTY is False:
            new_values["show_x"] = True
            logger.warning("Both dynamic plots were set to false, Showing only X.")

        return replace(self, **new_values)


class DASF:
    """
    Base class for the DASF algorithm.

    Attributes
    ----------

    problem : OptimizationProblem
        The optimization problem to be solved.
    data_retriever : DataRetriever
        The data retriever object retrieving data for the optimization problem.
    network_graph : NetworkGraph
        The network graph representing the sensor network.
    dasf_convergence_params : ConvergenceParameters
        The convergence parameters for the DASF algorithm.
    updating_path : np.ndarray | None
        The path followed to select updating nodes.
    initial_estimate : np.ndarray | None
        The initial estimate for the optimization problem.
    rng : np.random.Generator | None
        Random number generator for reproducibility.
    solver_convergence_parameters : ConvergenceParameters | None
        The convergence parameters of the provided optimization problem. Adds an additional degree of freedom to select different parameters than the ones used by the centralized solver.
    dynamic_plot : bool
        Flag to enable dynamic plotting during the algorithm run.
    dynamic_plot_params : DynamicPlotParameters | None
        Parameters for dynamic plotting.

    """

    def __init__(
        self,
        problem: OptimizationProblem,
        data_retriever: DataRetriever,
        network_graph: NetworkGraph,
        dasf_convergence_params: ConvergenceParameters,
        updating_path: np.ndarray | None = None,
        initial_estimate: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        solver_convergence_parameters: ConvergenceParameters | None = None,
        dynamic_plot: bool = False,
        dynamic_plot_params: DynamicPlotParameters | None = None,
    ) -> None:
        self.problem = problem
        self.data_retriever = data_retriever
        self.network_graph = network_graph
        self.dasf_convergence_params = dasf_convergence_params
        self.solver_convergence_parameters = solver_convergence_parameters
        if solver_convergence_parameters is None:
            if problem.convergence_parameters is not None:
                self.solver_convergence_parameters = problem.convergence_parameters
                logger.warning(
                    "Using same convergence parameters as centralized solver"
                )
            else:
                self.solver_convergence_parameters = None
                logger.info(
                    "No convergence parameters provided for the solver, assuming it is not neccessary (e.g., closed form solution)."
                )
        if updating_path is not None:
            self.updating_path = updating_path
        else:
            self.updating_path = (
                rng.permutation(range(network_graph.nb_nodes))
                if rng is not None
                else np.arange(1, network_graph.nb_nodes + 1)
            )
        if initial_estimate is not None:
            self.initial_estimate = initial_estimate
        else:
            self.initial_estimate = (
                rng.standard_normal(
                    (network_graph.nb_sensors_total, problem.nb_filters)
                )
                if rng is not None
                else np.random.standard_normal(
                    (network_graph.nb_sensors_total, problem.nb_filters)
                )
            )
        self.nb_variables = problem.nb_variables
        self.dynamic_plot = dynamic_plot
        if dynamic_plot:
            if dynamic_plot_params is not None:
                self.dynamic_plot_params = dynamic_plot_params.apply_correction(
                    nb_filters=self.problem.nb_filters,
                    nb_samples=self.data_retriever.data_window_params.window_length,
                )
            else:
                self.dynamic_plot_params = DynamicPlotParameters()
        self.X_over_iterations = []
        self.f_over_iterations = []
        self.X_star_over_iterations = []
        self.f_star_over_iterations = []

        self._validate_problem()

    def centralized_solution_for_input(
        self,
        problem_inputs: ProblemInputs | list[ProblemInputs],
        initial_estimate: np.ndarray | list[np.ndarray] | None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Solves the centralized problem, used for comparison.

        Parameters
        ----------
        problem_inputs : ProblemInputs | list[ProblemInputs]
            The inputs of the problem.
        initial_estimate : np.ndarray | list[np.ndarray] | None
            The initial estimate for the problem.
        Returns
        -------
        np.ndarray | list[np.ndarray]
            The solution to the centralized problem.
        """
        return self.problem.solve(
            problem_inputs=problem_inputs,
            save_solution=False,
            convergence_parameters=self.problem.convergence_parameters
            if self.problem.convergence_parameters is not None
            else None,
            initial_estimate=initial_estimate,
        )

    @property
    def X_estimate(self):
        """
        Final estimate of te optimal solution using DASF.

        Returns
        -------
        np.ndarray | None
            The final estimate of the optimal solution using DASF."""
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
            return None
        else:
            X_estimate = self.problem.resolve_ambiguity(
                self.X_over_iterations[-1], self.X_star_over_iterations[-1]
            )
            return X_estimate

    @property
    def normed_difference_over_iterations(self):
        """
        Returns the sequence :math:`\\frac{1}{MQ}\\|X^{i+1}-X^i\\|_F^2`, where :math:`\\{X^i\\}_i\\subset \\mathbb{R}^{M\\times Q}`.
        If there are multiple variables, the variables are first stacked vertically.
        """
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
            return None
        else:
            return [
                np.linalg.norm(X_new - X, "fro") ** 2 / X.size
                for X, X_new in zip(
                    self.X_over_iterations[:-1], self.X_over_iterations[1:]
                )
            ]

    @property
    def normed_error_over_iterations(self):
        """
        Returns the sequence :math:`\\frac{\\|X^i-X^*\\|_F^2}{\\|X^*\\|_F^2}`.
        If there are multiple variables, the variables are first stacked vertically.
        """
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
            return None
        else:
            return [
                np.linalg.norm(X - X_star, "fro") ** 2
                / np.linalg.norm(X_star, "fro") ** 2
                for X, X_star in zip(
                    self.X_over_iterations, self.X_star_over_iterations
                )
            ]

    @property
    def absolute_objective_error_over_iterations(self):
        """
        Returns the sequence :math:`|f(X^{i})-f(X^*)|`.
        If there are multiple variables, the variables are first stacked vertically.
        """
        if hasattr(self.problem, "evaluate_objective"):
            if len(self.X_over_iterations) == 0:
                logger.warning(
                    "No iterates have been computed, use the run method first."
                )
                return None
            else:
                return [
                    np.abs(f - f_star)
                    for f, f_star in zip(
                        self.f_over_iterations, self.f_star_over_iterations
                    )
                ]
        else:
            logger.warning(
                "The problem does not have an evaluate_objective method. The objective has not been evaluated."
            )
            return None

    @property
    def total_iterations(self):
        """Returns the number of iterations performed during the run."""
        return len(self.X_over_iterations) - 1

    def run(self) -> None:
        """
        Main method to run the DASF algorithm.
        Values summarizing the result are stored in various attributes of the class.
        """
        self.X_over_iterations.clear()
        self.f_over_iterations.clear()
        self.X_star_over_iterations.clear()
        self.f_star_over_iterations.clear()

        problem_inputs = self.data_retriever.get_data_window(window_id=0)

        X = self.initial_estimate
        self.X_over_iterations.append(X)
        X_star_current_window = self.centralized_solution_for_input(
            problem_inputs=problem_inputs, initial_estimate=X
        )
        self.X_star_over_iterations.append(X_star_current_window)
        if hasattr(self.problem, "evaluate_objective"):
            f = self.problem.evaluate_objective(X=X, problem_inputs=problem_inputs)
            self.f_over_iterations.append(f)
            self.f_star_over_iterations.append(
                self.problem.evaluate_objective(
                    X=X_star_current_window, problem_inputs=problem_inputs
                )
            )

        if self.dynamic_plot:
            Y = problem_inputs.fused_signals[self.dynamic_plot_params.Y_id]
            line_x, line_xs, line_xTY, line_xsTY = self._init_dynamic_plot(
                X=X, X_star=X_star_current_window, Y=Y
            )

        i = 0
        window_id = 0
        while i < self.dasf_convergence_params.max_iterations:
            # Select updating node
            updating_node = self.updating_path[i % self.network_graph.nb_nodes]

            # Prune the network
            # Find shortest path
            neighbors, path = self._find_path(updating_node=updating_node)

            # Neighborhood clusters
            clusters = self._find_clusters(neighbors=neighbors, path=path)

            # Global - local transition matrix
            Cq = self._build_Cq(
                X=X, updating_node=updating_node, neighbors=neighbors, clusters=clusters
            )

            # Get current data window
            if i % self.data_retriever.data_window_params.nb_window_reuse == 0:
                problem_inputs = self.data_retriever.get_data_window(
                    window_id=window_id
                )
                X_star_current_window = self.centralized_solution_for_input(
                    problem_inputs=problem_inputs, initial_estimate=X
                )
                window_id += 1

            # Compute the compressed data
            compressed_inputs = self._compress(problem_inputs=problem_inputs, Cq=Cq)

            # Compute the local variable
            # Solve the local problem with the algorithm for the global problem using the compressed data
            Xq = self._get_block_q(X=X, updating_node=updating_node)
            X_tilde = np.concatenate(
                (
                    Xq,
                    np.tile(np.eye(self.problem.nb_filters), (len(neighbors), 1)),
                ),
                axis=0,
            )
            X_tilde_new = self.problem.solve(
                problem_inputs=compressed_inputs,
                convergence_parameters=self.solver_convergence_parameters,
                initial_estimate=X_tilde,
            )

            # Select a solution among potential ones if the problem has multiple solutions
            X_tilde_new = self.problem.resolve_ambiguity(
                X_reference=X_tilde, X_current=X_tilde_new, updating_node=updating_node
            )

            # Global variable
            X_new = Cq @ X_tilde_new
            self.X_over_iterations.append(X_new)
            X_star_current_window = self.problem.resolve_ambiguity(
                X_reference=X_new,
                X_current=X_star_current_window,
                updating_node=updating_node,
            )
            self.X_star_over_iterations.append(X_star_current_window)
            if hasattr(self.problem, "evaluate_objective"):
                f_new = self.problem.evaluate_objective(
                    X=X_tilde_new, problem_inputs=compressed_inputs
                )
                self.f_over_iterations.append(f_new)
                self.f_star_over_iterations.append(
                    self.problem.evaluate_objective(
                        X=X_star_current_window, problem_inputs=problem_inputs
                    )
                )

            if self.dynamic_plot:
                Y = problem_inputs.fused_signals[self.dynamic_plot_params.Y_id]
                self._update_dynamic_plot(
                    X=X_new,
                    X_star=X_star_current_window,
                    Y=Y,
                    line_x=line_x,
                    line_xs=line_xs,
                    line_xTY=line_xTY,
                    line_xsTY=line_xsTY,
                )

            i += 1

            if (
                hasattr(self.problem, "evaluate_objective")
                and (self.dasf_convergence_params.objective_tolerance is not None)
                and (
                    np.absolute(f_new - f)
                    <= self.dasf_convergence_params.objective_tolerance
                )
            ):
                logger.warning(
                    f"Stopped after {i} iterations due to reaching the threshold in difference in objectives"
                )
                break

            if (self.dasf_convergence_params.argument_tolerance is not None) and (
                np.linalg.norm(X_new - X, "fro")
                <= self.dasf_convergence_params.argument_tolerance
            ):
                logger.warning(
                    f"Stopped after {i} iterations due to reaching the threshold in difference in arguments"
                )
                break

            X = X_new.copy()
            if hasattr(self.problem, "evaluate_objective"):
                f = f_new

        if self.dynamic_plot:
            plt.ioff()
            # plt.show(block=False)
            plt.close()

        return None

    def _find_path(self, updating_node: int) -> Tuple[list[int], list[list[int]]]:
        """
        Finds the neighbors of a given node and determines the shortest path to every other node in the network.

        Parameters
        ----------
        updating_node : int
            The source node for which neighbors and shortest paths are computed.

        Returns
        -------
        tuple
            A tuple containing:
            - neighbors : list of int
                A sorted list of nodes that are direct neighbors of `updating_node`.
            - path : list of list of int
                A list where the element at index `k` contains the shortest path from `updating_node` to node `k`.
        """
        dist, path = self._shortest_path(updating_node)
        neighbors = [x for x in range(len(path)) if len(path[x]) == 2]
        neighbors.sort()

        return neighbors, path

    def _shortest_path(self, updating_node: int) -> Tuple[np.ndarray, list[list[int]]]:
        """
        Computes the shortest path distances from a given source node to all other nodes
        in the network using Dijkstra's algorithm.

        Note
        ----
        - This implementation assumes that all edges have a weight of 1.
        - Uses an adjacency matrix representation for the graph.

        Parameters
        ----------
        updating_node : int
            The source node from which shortest paths are calculated.

        Returns
        -------
        tuple
            A tuple containing:
            - dist : np.ndarray
                A 1D array where the value at index `k` represents the shortest distance from `updating_node` to node `k`.
            - path : list of list of int
                A list where the element at index `k` contains the shortest path from `updating_node` to node `k`.
        """
        nb_nodes = self.network_graph.nb_nodes
        dist = np.inf * np.ones(self.network_graph.nb_nodes)
        dist[updating_node] = 0

        visited = []
        pred = np.zeros(nb_nodes, dtype=int)

        def diff(l1, l2):
            return [x for x in l1 if x not in l2]

        def intersect(l1, l2):
            return [x for x in l1 if x in l2]

        unvisited = diff(list(range(nb_nodes)), visited)
        path = []

        while len(visited) < nb_nodes:
            inds = np.argwhere(dist == np.min(dist[unvisited])).T[0]

            for ind in inds:
                visited.append(ind)
                unvisited = diff(list(range(nb_nodes)), visited)
                neighbors_ind = [
                    i
                    for i, x in enumerate(self.network_graph.adjacency_matrix[ind, :])
                    if x == 1
                ]
                for m in intersect(neighbors_ind, unvisited):
                    if dist[ind] + 1 < dist[m]:
                        dist[m] = dist[ind] + 1
                        pred[m] = ind

        for k in range(nb_nodes):
            jmp = k
            path_k = [k]
            while jmp != updating_node:
                jmp = pred[jmp]
                path_k.insert(0, jmp)

            path.append(path_k)

        return dist, path

    def _find_clusters(
        self, neighbors: list[int], path: list[list[int]]
    ) -> list[list[int]]:
        """
        Obtains clusters of nodes for each neighbor by removing their direct connection to the source node.

        Parameters
        ----------
        neighbors : list of int
            List of neighbors of the source node.
        path : list of list of int
            A list where the element at index `k` contains the shortest path from the source node to node `k`.

        Returns
        -------
        list of list of int
            A list of clusters. Each sublist corresponds to a neighbor and contains the nodes that belong to the subgraph
            formed after removing the direct connection between the source node and that neighbor.
        """
        clusters = []
        for k in neighbors:
            clusters.append([x for x in range(len(path)) if k in path[x]])

        return clusters

    def _build_Cq(
        self,
        X: np.ndarray,
        updating_node: int,
        neighbors: list[int],
        clusters: list[list[int]],
    ) -> np.ndarray:
        """
        Constructs the transition matrix that maps the local data and variables to the global ones.

        Parameters
        ----------
        X : np.ndarray
            A matrix of shape (nb_sensors, nb_filters) representing the global variable,
            structured as [X1; ...; Xq; ...; XK].
        updating_node : int
            The current updating node.
        neighbors : list of int
            A list of neighbors of the updating node.
        clusters : list of list of int
            A list of clusters, where each sublist corresponds to a neighbor and contains nodes in the subgraph formed by
            cutting the link between the updating node and that neighbor.

        Returns
        -------
        np.ndarray
            The transition matrix `Cq`, which facilitates the transition between local and global data representations.
        """
        nb_sensors_per_node = self.network_graph.nb_sensors_per_node
        nb_filters = self.problem.nb_filters
        nb_neighbors = len(neighbors)

        ind = np.arange(self.network_graph.nb_nodes)

        Cq = np.zeros(
            (
                np.sum(nb_sensors_per_node),
                nb_sensors_per_node[updating_node] + nb_neighbors * nb_filters,
            )
        )
        Cq[:, 0 : nb_sensors_per_node[updating_node]] = np.vstack(
            (
                np.zeros(
                    (
                        np.sum(nb_sensors_per_node[0:updating_node]),
                        nb_sensors_per_node[updating_node],
                    )
                ),
                np.identity(nb_sensors_per_node[updating_node]),
                np.zeros(
                    (
                        np.sum(nb_sensors_per_node[updating_node + 1 :]),
                        nb_sensors_per_node[updating_node],
                    )
                ),
            )
        )
        for k in range(nb_neighbors):
            ind_k = ind[k]
            for n in range(len(clusters[k])):
                clusters_k = clusters[k]
                l = clusters_k[n]
                X_curr = X[
                    np.sum(nb_sensors_per_node[0:l]) : np.sum(
                        nb_sensors_per_node[0 : l + 1]
                    ),
                    :,
                ]
                Cq[
                    np.sum(nb_sensors_per_node[0:l]) : np.sum(
                        nb_sensors_per_node[0 : l + 1]
                    ),
                    nb_sensors_per_node[updating_node]
                    + ind_k * nb_filters : nb_sensors_per_node[updating_node]
                    + ind_k * nb_filters
                    + nb_filters,
                ] = X_curr

        return Cq

    def _compress(self, problem_inputs: ProblemInputs, Cq: np.ndarray) -> ProblemInputs:
        """
        Compresses the data using the transition matrix :math:`C_q`.

        Parameters
        ----------
        problem_inputs : ProblemInputs
            An object containing the original problem data, including:
            - fused_signals : list
                List of signals to be fused.
            - fused_constants : list, optional
                List of constants to be fused (if present).
            - fused_quadratics : list, optional
                List of quadratic matrices to be fused (if present).
            - global_parameters : object, optional
                Additional global parameters (if present).
        Cq : np.ndarray
            The transition matrix that maps the local data to the global one.

        Returns
        -------
        ProblemInputs
            A new instance of `ProblemInputs` containing the compressed versions of:
            - fused_signals
            - fused_constants
            - fused_quadratics
            - global_parameters (unchanged)
        """
        fused_data = problem_inputs.fused_signals
        fused_constants = problem_inputs.fused_constants
        fused_quadratics = problem_inputs.fused_quadratics
        global_constants = problem_inputs.global_parameters

        data_list_compressed = []
        nb_data = len(fused_data)
        for ind in range(nb_data):
            data_list_compressed.append(Cq.T @ fused_data[ind])

        if fused_constants is not None:
            constants_list_compressed = []
            nb_constants = len(fused_constants)
            for ind in range(nb_constants):
                constants_list_compressed.append(Cq.T @ fused_constants[ind])
        else:
            constants_list_compressed = None

        if fused_quadratics is not None:
            quadratics_list_compressed = []
            nb_quadratics = len(fused_quadratics)
            for ind in range(nb_quadratics):
                quadratics_list_compressed.append(Cq.T @ fused_quadratics[ind] @ Cq)
        else:
            quadratics_list_compressed = None

        compressed_inputs = ProblemInputs(
            fused_signals=data_list_compressed,
            fused_constants=constants_list_compressed,
            fused_quadratics=quadratics_list_compressed,
            global_parameters=global_constants,
        )

        return compressed_inputs

    def _get_block_q(self, X: np.ndarray, updating_node: int) -> np.ndarray:
        """
        Extracts the block of `X` corresponding to the specified updating node.

        Parameters
        ----------
        X : np.ndarray
            A matrix of shape (nb_sensors, nb_filters) representing the global variable,
            structured as :math:`\\begin{bmatrix}X_1^T, ..., X_q^T, ..., X_K^T\\end{bmatrix}^T`.
        updating_node : int
            The node for which the corresponding block of :math:`X` is extracted.

        Returns
        -------
        np.ndarray
            A matrix :math:`X_q` of shape (nb_sensors_per_node[updating_node], nb_filters), containing the block of :math:`X`
            corresponding to `updating_node`.
        """
        Mq = self.network_graph.nb_sensors_per_node[updating_node]
        row_blk = np.cumsum(self.network_graph.nb_sensors_per_node)
        row_blk = np.append(0, row_blk[0:-1])
        row_blk_q = row_blk[updating_node]
        Xq = X[row_blk_q : row_blk_q + Mq, :]

        return Xq

    def _init_dynamic_plot(
        self, X: np.ndarray, X_star: np.ndarray, Y: np.ndarray
    ) -> Tuple[Line2D]:
        """
        Initializes the dynamic plot.

        Parameters
        ----------
        X : np.ndarray
            Current estimate matrix.
        X_star : np.ndarray
            Optimal solution matrix.
        Y : np.ndarray
            Signal fused using :math:`X`.
        """
        plt.ion()
        if self.dynamic_plot_params.show_x & self.dynamic_plot_params.show_xTY:
            fig, axes = plt.subplots(2, 1, figsize=(7, 6))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(7, 4))

        (line_x, line_xs, line_xTY, line_xsTY) = (None, None, None, None)
        if self.dynamic_plot_params.show_x:
            ax1 = axes[0] if self.dynamic_plot_params.show_xTY else axes
            (line_x,) = ax1.plot(
                X[:, self.dynamic_plot_params.X_col],
                color="r",
                marker="x",
                label="Current estimate",
            )
            (line_xs,) = ax1.plot(
                X_star[:, self.dynamic_plot_params.X_col],
                color="b",
                label="Centralized solution",
            )
            if self.nb_variables & self.nb_variables > 1:
                for k in range(0, self.nb_variables - 1):
                    pos = (k + 1) * (X.shape[0] / self.nb_variables)
                    ax1.axvline(
                        x=pos,
                        color="k",
                        linestyle="--",
                        label="Delimiter for variables",
                    )
            ax1.set_xlim(1, X.shape[0])
            ax1.set_ylim(
                2 * np.min(X_star[:, self.dynamic_plot_params.X_col]),
                2 * np.max(X_star[:, self.dynamic_plot_params.X_col]),
            )
            ax1.legend(loc="upper right")
            ax1.set_xlabel("Sensors")
            ax1.set_ylabel("Weight values")
            ax1.set_title(
                rf"$X$: Weights per sensor for filter {self.dynamic_plot_params.X_col + 1}"
            )
            ax1.grid()

        if self.dynamic_plot_params.show_xTY:
            ax2 = axes[1] if self.dynamic_plot_params.show_x else axes
            xTY = X[:, [self.dynamic_plot_params.XTY_col]].T @ Y
            xsTY = X_star[:, [self.dynamic_plot_params.XTY_col]].T @ Y
            sampled_indices = np.arange(0, xTY.shape[1], self.dynamic_plot_params.tau)
            (line_xTY,) = ax2.plot(
                sampled_indices,
                xTY[0, sampled_indices],
                color="orange",
                marker="x",
                label="Filtered signal estimation",
            )
            (line_xsTY,) = ax2.plot(
                sampled_indices,
                xsTY[0, sampled_indices],
                color="b",
                label="Centralized filtered signal",
            )
            if self.nb_variables > 1:
                for k in range(0, self.nb_variables - 1):
                    pos = (k + 1) * (sampled_indices[-1] / self.nb_variables)
                    ax2.axvline(
                        x=pos,
                        color="k",
                        linestyle="--",
                        label="Delimiter for variables",
                    )
            ax2.set_xlim(0, sampled_indices[-1])
            ax2.set_ylim(
                2 * np.min(xsTY[0, :]),
                2 * np.max(xsTY[0, :]),
            )
            ax2.legend(loc="upper right")
            ax2.set_xlabel("Samples")
            ax2.set_ylabel("Filtered signal values")
            ax2.set_title(
                rf"$X^Ty(t)$: Filtered signal {self.dynamic_plot_params.Y_id + 1} for filter {self.dynamic_plot_params.XTY_col + 1}, shown every {self.dynamic_plot_params.tau} sample"
            )
            ax2.grid()

        plt.subplots_adjust(hspace=0.4)
        plt.show()
        return line_x, line_xs, line_xTY, line_xsTY

    def _update_dynamic_plot(
        self,
        X: np.ndarray,
        X_star: np.ndarray,
        Y: np.ndarray,
        line_x: Line2D | None,
        line_xs: Line2D | None,
        line_xTY: Line2D | None,
        line_xsTY: Line2D | None,
    ) -> None:
        """
        Updates the dynamic plot.

        Parameters
        ----------
        X : np.ndarray
            Current estimate matrix.
        X_star : np.ndarray
            Optimal solution matrix.
        Y : np.ndarray
            Signal fused using :math:`X`.
        line_x : object
            Line object for :math:`X`.
        line_xs : object
            Line object for :math:`X^*`.
        line_xTY : object
            Line object for :math:`X^T\\mathbf{y}(t)`.
        line_xsTY : object
            Line object for :math:`X^{*T}\\mathbf{y}(t)`.
        """

        if self.dynamic_plot_params.show_x:
            line_x.set_ydata(X[:, self.dynamic_plot_params.X_col])
            line_xs.set_ydata(X_star[:, self.dynamic_plot_params.X_col])

        if self.dynamic_plot_params.show_xTY:
            xTY = X[:, [self.dynamic_plot_params.XTY_col]].T @ Y
            xsTY = X_star[:, [self.dynamic_plot_params.XTY_col]].T @ Y
            sampled_indices = np.arange(0, xTY.shape[1], self.dynamic_plot_params.tau)
            line_xTY.set_xdata(sampled_indices)
            line_xTY.set_ydata(xTY[0, sampled_indices])
            line_xsTY.set_ydata(xsTY[0, sampled_indices])

        plt.draw()
        plt.pause(0.05)

    def _validate_problem(self) -> None:
        """Validates the problem and its inputs."""
        problem_inputs = self.data_retriever.get_data_window(window_id=0)
        nb_sensor = self.network_graph.nb_sensors_total
        for index, signal in enumerate(problem_inputs.fused_signals):
            if np.size(signal, 0) != nb_sensor:
                raise ValueError(
                    f"The number of rows in data {index} does not match the number of sensors in the network graph."
                )
        if problem_inputs.fused_constants is not None:
            for index, constant in enumerate(problem_inputs.fused_constants):
                if np.size(constant, 0) != nb_sensor:
                    raise ValueError(
                        f"The number of rows in the fused constant {index} does not match the number of sensors in the network graph."
                    )
        if problem_inputs.fused_quadratics is not None:
            for index, quadratic in enumerate(problem_inputs.fused_quadratics):
                if (np.size(quadratic, 0) != nb_sensor) or (
                    np.size(quadratic, 1) != nb_sensor
                ):
                    raise ValueError(
                        f"The number of rows or columns in the fused quadratic {index} does not match the number of sensors in the network graph."
                    )
        if self.initial_estimate.shape != (
            nb_sensor,
            self.problem.nb_filters,
        ):
            raise ValueError(
                "The initial estimate does not have the correct shape for the problem."
            )
        if not hasattr(self.problem, "solve"):
            raise ValueError("The problem does not have a solve method.")
        if not hasattr(self.problem, "evaluate_objective"):
            logger.warning(
                "The problem does not have an evaluate_objective method. The objective will not be evaluated."
            )

    def plot_error(self) -> Figure:
        """
        Plots the sequence :math:`\\frac{\\|X^i-X^*\\|_F^2}{\\|X^*\\|_F^2}`.
        If there are multiple variables, the variables are first stacked vertically.
        """
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            range(1, self.total_iterations + 1),
            self.normed_error_over_iterations[1:],
            color="b",
        )
        ax.set_xlabel(r"Iterations $i$")
        ax.set_ylabel(r"$\varepsilon(i)=\frac{\|X^i-X^*\|_F^2}{\|X^*\|_F^2}$")
        ax.grid(True, which="both")
        return fig

    def plot_error_over_batches(self) -> Figure:
        """
        Plots the sequence :math:`\\frac{\\|X^i-X^*\\|_F^2}{\\|X^*\\|_F^2}` at the end of each batch of data.
        If there are multiple variables, the variables are first stacked vertically.
        """
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.semilogy(
            range(
                1,
                int(
                    self.total_iterations
                    / self.data_retriever.data_window_params.nb_window_reuse
                )
                + 1,
            ),
            self.normed_error_over_iterations[
                1 :: self.data_retriever.data_window_params.nb_window_reuse
            ],
            color="b",
        )
        ax.set_xlabel(r"Batch $i$")
        ax.set_ylabel(r"$\varepsilon(i)=\frac{\|X^i-X^*\|_F^2}{\|X^*\|_F^2}$")
        ax.grid(True, which="both")
        return fig

    def plot_iterate_difference(self) -> Figure:
        """
        Plots the sequence :math:`\\frac{1}{MQ}\\|X^{i+1}-X^i\\|_F^2`, where :math:`\\{X^i\\}_i\\subset \\mathbb{R}^{M\\times Q}`.`
        If there are multiple variables, the variables are first stacked vertically.
        """
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            range(1, self.total_iterations + 1),
            self.normed_difference_over_iterations,
            color="b",
        )
        ax.set_xlabel(r"Iterations $i$")
        ax.set_ylabel(r"$\frac{\|X^i-X^{i-1}\|_F^2}{MQ}$")
        ax.grid(True, which="both")
        return fig

    def plot_objective_error(self) -> Figure:
        """
        Plots the sequence :math:`|f(X^{i})-f(X^*)|`.
        If there are multiple variables, the variables are first stacked vertically.
        """
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            range(1, self.total_iterations + 1),
            self.absolute_objective_error_over_iterations[1:],
            color="b",
        )
        ax.set_xlabel(r"Iterations $i$")
        ax.set_ylabel(r"$|f(X^i)-f(X^*)|$")
        ax.grid(True, which="both")
        return fig

    def get_summary_df(self) -> pd.DataFrame | None:
        """Returns a pandas DataFrame summarizing the simulation results."""
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
            return None

        df_summary = pd.DataFrame(
            {
                "iterations": range(1, self.total_iterations + 1),
                "iterate_difference": self.normed_difference_over_iterations,
                "error": self.normed_error_over_iterations[1:],
                "objective_error": self.absolute_objective_error_over_iterations[1:],
                "updating_node": np.tile(
                    self.updating_path,
                    (self.total_iterations // len(self.updating_path)) + 1,
                )[: self.total_iterations],
            }
        )
        nb_neighbors = np.sum(
            self.network_graph.adjacency_matrix[df_summary["updating_node"]], axis=1
        )
        df_summary["number_of_neighbors"] = nb_neighbors
        batch_list = []
        batch_index = 1
        while len(batch_list) < self.total_iterations:
            batch_list.extend(
                [batch_index] * self.data_retriever.data_window_params.nb_window_reuse
            )
            batch_index += 1
        batch_list = batch_list[: self.total_iterations]
        df_summary["batch"] = batch_list

        return df_summary


class DASFMultiVar(DASF):
    """
    Class inheriting from the DASF class and implementing the DASF algorithm for a setting with multiple variables.

    Attributes
    ----------

    problem : OptimizationProblem
        The optimization problem to be solved.
    data_retriever : DataRetriever
        The data retriever object retrieving data for the optimization problem.
    network_graph : NetworkGraph
        The network graph representing the sensor network.
    dasf_convergence_params : ConvergenceParameters
        The convergence parameters for the DASF algorithm.
    updating_path : np.ndarray | None
        The path followed to select updating nodes.
    initial_estimate : list[np.ndarray] | None
        The list of initial estimates for the optimization problem, one for each variable.
    rng : np.random.Generator | None
        Random number generator for reproducibility.
    solver_convergence_parameters : ConvergenceParameters | None
        The convergence parameters of the provided optimization problem. Adds an additional degree of freedom to select different parameters than the ones used by the centralized solver.
    dynamic_plot : bool
        Flag to enable dynamic plotting during the algorithm run.
    dynamic_plot_params : DynamicPlotParameters | None
        Parameters for dynamic plotting.

    """

    def __init__(
        self,
        problem: OptimizationProblem,
        data_retriever: DataRetriever,
        network_graph: NetworkGraph,
        dasf_convergence_params: ConvergenceParameters,
        updating_path: np.ndarray | None = None,
        initial_estimate: list[np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
        solver_convergence_parameters: ConvergenceParameters | None = None,
        dynamic_plot: bool = False,
        dynamic_plot_params: DynamicPlotParameters | None = None,
    ) -> None:
        self._validate_problem = self._validate_problem

        if initial_estimate is not None:
            self.initial_estimate = initial_estimate
        else:
            initial_estimate = []
            for k in range(problem.nb_variables):
                initial_estimate.append(
                    rng.standard_normal(
                        (network_graph.nb_sensors_total, problem.nb_filters)
                    )
                    if rng is not None
                    else np.random.standard_normal(
                        (network_graph.nb_sensors_total, problem.nb_filters)
                    )
                )

        super().__init__(
            problem=problem,
            data_retriever=data_retriever,
            network_graph=network_graph,
            dasf_convergence_params=dasf_convergence_params,
            updating_path=updating_path,
            initial_estimate=initial_estimate,
            rng=rng,
            solver_convergence_parameters=solver_convergence_parameters,
            dynamic_plot=dynamic_plot,
            dynamic_plot_params=dynamic_plot_params,
        )

    @property
    def normed_difference_over_iterations(self):
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
            return None
        else:
            return [
                np.linalg.norm(np.vstack(X_new) - np.vstack(X), "fro") ** 2
                / np.vstack(X).size
                for X, X_new in zip(
                    self.X_over_iterations[:-1], self.X_over_iterations[1:]
                )
            ]

    @property
    def normed_error_over_iterations(self):
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
            return None
        else:
            return [
                np.linalg.norm(np.vstack(X) - np.vstack(X_star), "fro") ** 2
                / np.linalg.norm(np.vstack(X_star), "fro") ** 2
                for X, X_star in zip(
                    self.X_over_iterations, self.X_star_over_iterations
                )
            ]

    def run(self) -> None:
        """Main method to run the DASF algorithm in a setting with multiple variables."""
        self.X_over_iterations.clear()
        self.f_over_iterations.clear()
        self.X_star_over_iterations.clear()
        self.f_star_over_iterations.clear()

        problem_inputs = self.data_retriever.get_data_window(window_id=0)

        X = self.initial_estimate
        self.X_over_iterations.append(X)
        X_star_current_window = self.centralized_solution_for_input(
            problem_inputs=problem_inputs, initial_estimate=X
        )
        self.X_star_over_iterations.append(X_star_current_window)
        if hasattr(self.problem, "evaluate_objective"):
            f = self.problem.evaluate_objective(X=X, problem_inputs=problem_inputs)
            self.f_over_iterations.append(f)
            self.f_star_over_iterations.append(
                self.problem.evaluate_objective(
                    X=X_star_current_window, problem_inputs=problem_inputs
                )
            )

        if self.dynamic_plot:
            X_stack = np.vstack(X)
            X_star_stack = np.vstack(X_star_current_window)
            Y = np.vstack(
                [
                    p_i.fused_signals[self.dynamic_plot_params.Y_id]
                    for p_i in problem_inputs
                ]
            )
            line_x, line_xs, line_xTY, line_xsTY = self._init_dynamic_plot(
                X=X_stack,
                X_star=X_star_stack,
                Y=Y,
            )

        i = 0
        window_id = 0
        while i < self.dasf_convergence_params.max_iterations:
            # Select updating node
            updating_node = self.updating_path[i % self.network_graph.nb_nodes]

            # Prune the network
            # Find shortest path
            neighbors, path = self._find_path(updating_node=updating_node)

            # Neighborhood clusters
            clusters = self._find_clusters(neighbors=neighbors, path=path)

            # Get current data window
            if i % self.data_retriever.data_window_params.nb_window_reuse == 0:
                problem_inputs = self.data_retriever.get_data_window(
                    window_id=window_id
                )
                X_star_current_window = self.centralized_solution_for_input(
                    problem_inputs=problem_inputs, initial_estimate=X
                )
                window_id += 1

            # Global - local transition matrix
            Cq = []

            compressed_inputs = []
            X_tilde = []
            for k in range(self.nb_variables):
                Cq_k = self._build_Cq(
                    X=X[k],
                    updating_node=updating_node,
                    neighbors=neighbors,
                    clusters=clusters,
                )
                Cq.append(Cq_k)
                # Compute the compressed data for each input
                compressed_inputs_k = self._compress(
                    problem_inputs=problem_inputs[k], Cq=Cq[k]
                )
                compressed_inputs.append(compressed_inputs_k)

                # Compute each local variable
                Xq_k = self._get_block_q(X=X[k], updating_node=updating_node)
                X_tilde_k = np.concatenate(
                    (
                        Xq_k,
                        np.tile(np.eye(self.problem.nb_filters), (len(neighbors), 1)),
                    ),
                    axis=0,
                )
                X_tilde.append(X_tilde_k)

            # Solve the local problem with the algorithm for the global problem using the compressed data
            X_tilde_new = self.problem.solve(
                problem_inputs=compressed_inputs,
                convergence_parameters=self.solver_convergence_parameters,
                initial_estimate=X_tilde,
            )

            # Select a solution among potential ones if the problem has multiple solutions
            X_tilde_new = self.problem.resolve_ambiguity(
                X_reference=X_tilde, X_current=X_tilde_new, updating_node=updating_node
            )

            # Global variable
            X_new = []
            for k in range(self.nb_variables):
                X_new.append(Cq[k] @ X_tilde_new[k])
            self.X_over_iterations.append(X_new)
            X_star_current_window = self.problem.resolve_ambiguity(
                X_reference=X_new,
                X_current=X_star_current_window,
                updating_node=updating_node,
            )
            self.X_star_over_iterations.append(X_star_current_window)
            if hasattr(self.problem, "evaluate_objective"):
                f_new = self.problem.evaluate_objective(
                    X=X_tilde_new, problem_inputs=compressed_inputs
                )
                self.f_over_iterations.append(f_new)
                self.f_star_over_iterations.append(
                    self.problem.evaluate_objective(
                        X=X_star_current_window, problem_inputs=problem_inputs
                    )
                )

            if self.dynamic_plot:
                X_stack = np.vstack(X_new)
                X_star_stack = np.vstack(X_star_current_window)
                Y = np.vstack(
                    [
                        p_i.fused_signals[self.dynamic_plot_params.Y_id]
                        for p_i in problem_inputs
                    ]
                )
                self._update_dynamic_plot(
                    X=X_stack,
                    X_star=X_star_stack,
                    Y=Y,
                    line_x=line_x,
                    line_xs=line_xs,
                    line_xTY=line_xTY,
                    line_xsTY=line_xsTY,
                )

            i += 1

            if (
                hasattr(self.problem, "evaluate_objective")
                and (self.dasf_convergence_params.objective_tolerance is not None)
                and (
                    np.absolute(f_new - f)
                    <= self.dasf_convergence_params.objective_tolerance
                )
            ):
                logger.warning(
                    f"Stopped after {i} iterations due to reaching the threshold in difference in objectives"
                )
                break

            if (self.dasf_convergence_params.argument_tolerance is not None) and (
                np.linalg.norm(np.vstack(X_new) - np.vstack(X), "fro")
                <= self.dasf_convergence_params.argument_tolerance
            ):
                logger.warning(
                    f"Stopped after {i} iterations due to reaching the threshold in difference in arguments"
                )
                break

            X = X_new.copy()
            if hasattr(self.problem, "evaluate_objective"):
                f = f_new

        if self.dynamic_plot:
            plt.ioff()
            # plt.show(block=False)
            plt.close()

        return None

    def _validate_problem(self):
        """Validates the problem and its inputs."""
        problem_inputs = self.data_retriever.get_data_window(window_id=0)
        if self.nb_variables != len(problem_inputs):
            raise ValueError(
                f"The number of variables {self.nb_variables} does not match the number of problem inputs {len(problem_inputs)}"
            )
        nb_sensor = self.network_graph.nb_sensors_total
        for input_id in range(len(problem_inputs)):
            for index, signal in enumerate(problem_inputs[input_id].fused_signals):
                if np.size(signal, 0) != nb_sensor:
                    raise ValueError(
                        f"The number of rows in data {index} does not match the number of sensors in the network graph for input {input_id}."
                    )
            if problem_inputs[input_id].fused_constants is not None:
                for index, constant in enumerate(
                    problem_inputs[input_id].fused_constants
                ):
                    if np.size(constant, 0) != nb_sensor:
                        raise ValueError(
                            f"The number of rows in the fused constant {index} does not match the number of sensors in the network graph for input {input_id}."
                        )
            if problem_inputs[input_id].fused_quadratics is not None:
                for index, quadratic in enumerate(
                    problem_inputs[input_id].fused_quadratics
                ):
                    if (np.size(quadratic, 0) != nb_sensor) or (
                        np.size(quadratic, 1) != nb_sensor
                    ):
                        raise ValueError(
                            f"The number of rows or columns in the fused quadratic {index} does not match the number of sensors in the network graph for input {input_id}."
                        )
            if self.initial_estimate[input_id].shape != (
                nb_sensor,
                self.problem.nb_filters,
            ):
                raise ValueError(
                    f"The initial estimate of the variable corresponding to input {input_id} does not have the correct shape for the problem."
                )
        if not hasattr(self.problem, "solve"):
            raise ValueError("The problem does not have a solve method.")
        if not hasattr(self.problem, "evaluate_objective"):
            logger.warning(
                "The problem does not have an evaluate_objective method. The objective will not be evaluated."
            )
