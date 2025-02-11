import numpy as np
from problem_settings import (
    NetworkGraph,
    ProblemInputs,
    DataWindowParameters,
    ConvergenceParameters,
)
from data_retriever import DataRetriever
from optimization_problems import OptimizationProblem
import logging
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DASF:
    def __init__(
        self,
        problem: OptimizationProblem,
        data_retriever: DataRetriever,
        network_graph: NetworkGraph,
        data_window_params: DataWindowParameters,
        dasf_convergence_params: ConvergenceParameters,
        updating_path: np.ndarray | None = None,
        initial_estimate: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        solver_convergence_parameters: ConvergenceParameters | None = None,
        dynamic_plot: bool = False,
    ):
        self.problem = problem
        self.data_retriever = data_retriever
        self.network_graph = network_graph
        self.dasf_convergence_params = dasf_convergence_params
        self.data_window_params = data_window_params
        self.solver_convergence_parameters = solver_convergence_parameters
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
        self.dynamic_plot = dynamic_plot
        self.X_over_iterations = []
        self.f_over_iterations = []
        self.X_star_over_iterations = []
        self.f_star_over_iterations = []

        self._validate_problem()

    def centralized_solution_for_input(
        self,
        problem_inputs: ProblemInputs,
        initial_estimate: np.ndarray | list[np.ndarray] | None,
    ) -> np.ndarray | list[np.ndarray]:
        return self.problem.solve(
            problem_inputs=problem_inputs,
            save_solution=False,
            convergence_parameters=self.problem.convergence_parameters
            if self.problem.convergence_parameters is not None
            else None,
            initial_estimate=initial_estimate,
        )

    @property
    def X_star(self):
        if len(self.X_over_iterations) == 0:
            logger.warning("No iterates have been computed, use the run method first.")
            return None
        else:
            X_star = self.problem.resolve_ambiguity(
                self.X_over_iterations[-1], self.X_star_over_iterations[-1]
            )
            return X_star

    @property
    def normed_difference_over_iterations(self):
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
                        self.f_over_iterations[1:], self.f_star_over_iterations[1:]
                    )
                ]
        else:
            logger.warning(
                "The problem does not have an evaluate_objective method. The objective has not been evaluated."
            )
            return None

    @property
    def total_iterations(self):
        return len(self.X_over_iterations) - 1

    def run(self) -> None:
        self.X_over_iterations.clear()
        self.X_star_over_iterations.clear()
        self.f_star_over_iterations.clear()
        self.f_over_iterations.clear()

        problem_inputs = self.data_retriever.get_current_window(window_id=0)

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
            plt.ion()
            fig, ax = plt.subplots()
            (line1,) = ax.plot(X[:, 1], color="r", marker="x", label="Current estimate")
            (line2,) = ax.plot(
                X_star_current_window[:, 1],
                color="b",
                label="Centralized solution",
            )
            plt.axis(
                [
                    0,
                    self.network_graph.nb_sensors_total,
                    2 * np.min(X_star_current_window[:, 1]),
                    2 * np.max(X_star_current_window[:, 1]),
                ]
            )
            ax.legend()
            ax.set_xlabel("Sensors")
            ax.set_ylabel("Weight values")
            ax.set_title("Weights per sensor for first filter")
            ax.grid()
            plt.show()

        i = 0
        window_id = 0
        while i < self.dasf_convergence_params.max_iterations:
            # Select updating node
            updating_node = self.updating_path[i % self.network_graph.nb_nodes]

            # Prune the network
            # Find shortest path
            neighbors, path = self._find_path(updating_node)

            # Neighborhood clusters
            clusters = self._find_clusters(neighbors, path)

            # Global - local transition matrix
            Cq = self._build_Cq(X, updating_node, neighbors, clusters)

            # Get current data window
            if i % self.data_window_params.nb_window_reuse == 0:
                problem_inputs = self.data_retriever.get_current_window(
                    window_id=window_id
                )
                X_star_current_window = self.centralized_solution_for_input(
                    problem_inputs=problem_inputs, initial_estimate=X
                )
                window_id += 1

            # Compute the compressed data
            compressed_inputs = self._compress(problem_inputs, Cq)

            # Compute the local variable
            # Solve the local problem with the algorithm for the global problem using the compressed data
            Xq = self._get_block_q(X, updating_node)
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
                X_ref=X_tilde, X=X_tilde_new, updating_node=updating_node
            )

            # Global variable
            X_new = Cq @ X_tilde_new
            self.X_over_iterations.append(X_new)
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
                X_compare = self.problem.resolve_ambiguity(
                    X_star_current_window, X, updating_node
                )
                self._plot_dynamically(X_compare, X_star_current_window, line1, line2)

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

    def _find_path(self, updating_node: int) -> Tuple[list, list]:
        """Function finding the neighbors of node q and the shortest path to other every other node in the network.

        INPUTS:

        q: Source node.

        adj (nbnodes x nbnodes): Adjacency (binary) matrix with adj[i,j]=1 if i and j are  connected. Otherwise 0. adj[i,i]=0.

        OUTPUTS:

        neighbors: List containing the neighbors of node q.

        path (nbnodes x 1): List of lists containining at index k the shortest path from node q to node k.
        """
        dist, path = self._shortest_path(updating_node)
        neighbors = [x for x in range(len(path)) if len(path[x]) == 2]
        neighbors.sort()

        return neighbors, path

    def _shortest_path(self, updating_node: int) -> Tuple[np.ndarray, list]:
        """Function computing the shortest path distance between a source node and all nodes
        in the network using Dijkstra's method.
        Note: This implementation is only for graphs for which the weight at each edge is equal to 1.

        INPUTS:

        q: Source node.

        adj (nbnodes x nbnodes): Adjacency (binary) matrix where K is the number of nodes
        in the network with adj[i,j]=1 if i and j are  connected. Otherwise 0. adj[i,i]=0.

        OUTPUTS:

        dist (nbnodes x 1): Distances between the source node and other nodes.

        path (nbnodes x 1): List of lists containining at index k the shortest path from node q to node k.
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

    def _find_clusters(self, neighbors: list, path: list) -> list:
        """Function to obtain clusters of nodes for each neighbor.

        INPUTS:

        neighbors: List containing the neighbors of node q.

        adj (nbnodes x nbnodes): Adjacency (binary) matrix where K is the number of nodes in the network with adj[i,j]=1 if i and j are  connected. Otherwise 0. adj[i,i]=0.

        OUTPUTS:

        clusters: List of lists. For each neighbor k of q, there is a corresponding list with the nodes of the subgraph containing k, obtained by cutting the link between nodes q and k.
        """
        clusters = []
        for k in neighbors:
            clusters.append([x for x in range(len(path)) if k in path[x]])

        return clusters

    def _build_Cq(
        self, X: np.ndarray, updating_node: int, neighbors: list, clusters: list
    ) -> np.ndarray:
        """Function to construct the transition matrix between the local data and
        variables and the global ones.

        INPUTS:

        X (nbsensors x Q): Global variable equal to [X1;...;Xq;...;XK].

        q: Updating node.

        prob_params: Dictionary related to the problem parameters.

        neighbors: List containing the neighbors of node q.

        clusters: List of lists. For each neighbor k of q, there is a corresponding
        list with the nodes of the subgraph containing k, obtained by cutting the link between nodes q and k.

        OUTPUTS:

        Cq: Transformation matrix making the transition between local and global data.
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
        """Function to compress the data.

        INPUTS:

        data: Dictionary related to the data.

        Cq: Transformation matrix making the transition between local and global data.

        OUTPUTS:

        data_compressed: Dictionary containing the compressed data. Contains the same keys as 'data'.
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

    def _get_block_q(self, X: np.ndarray, updating_node: int):
        """Function to extract the block of X corresponding to node q.

        INPUTS:

        X (nbsensors x Q): Global variable equal to [X1;...;Xq;...;XK].

        q: Updating node.

        nbsensors_vec (nbnodes x nbnodes): Vector containing the number of sensors for each node.

        OUTPUTS:

        Xq (nbsensors_vec(q) x Q): Block of X corresponding to node q.
        """
        Mq = self.network_graph.nb_sensors_per_node[updating_node]
        row_blk = np.cumsum(self.network_graph.nb_sensors_per_node)
        row_blk = np.append(0, row_blk[0:-1])
        row_blk_q = row_blk[updating_node]
        Xq = X[row_blk_q : row_blk_q + Mq, :]

        return Xq

    def _plot_dynamically(self, X, X_star, line1, line2):
        """Plot the first column of X and X_star.

        INPUTS:

        X (nbsensors x Q): Global variable equal.

        X_star (nbsensors x Q): Optimal solution.

        line1: Figure handle for X.

        line2: Figure handle for X_star.
        """
        line1.set_ydata(X[:, 1])
        line2.set_ydata(X_star[:, 1])
        plt.draw()
        plt.pause(0.05)

    def _update_X_block(
        self,
        X_block: np.ndarray,
        X_tilde: np.ndarray,
        updating_node: int,
        prob_params,
        neighbors,
        clusters,
        prob_select_sol,
    ):
        """Function to update the cell containing the blocks of X for each corresponding node.

        INPUTS:

        X_block (nbnodes x 1): Vector of cells containing the current blocks Xk^i at each cell k, where X=[X1;...;Xk;...;XK].

        X_tilde: Solution of the local problem at the current updating node.

        q: Updating node.

        prob_params: Structure related to the problem parameters.

        neighbors: Vector containing the neighbors of node q.

        Nu: List of lists. For each neighbor k of q, there is a corresponding
            list with the nodes of the subgraph containing k, obtained by cutting
            the link between nodes q and k.

        prob_select_sol : (Optional) Function resolving the uniqueness ambiguity.

        OUTPUTS:

        X_block_upd (nbnodes x 1): Vector of cells with updated block Xk^(i+1).
        """
        nb_nodes = self.network_graph.nb_nodes
        nb_sensors_per_node = self.network_graph.nb_sensors_per_node
        nb_filters = self.problem.nb_filters

        if prob_select_sol is not None:
            Xq_old = X_block[updating_node]
            X_tilde_old = np.concatenate(
                (Xq_old, np.tile(np.eye(nb_filters), (len(neighbors), 1))), axis=0
            )
            X_tilde = prob_select_sol(X_tilde_old, X_tilde, prob_params, updating_node)

        X_block_updated = []

        nb_neighbors = len(neighbors)
        ind = np.arange(nb_nodes)

        for node_id in range(updating_node):
            for neighbor_id in range(nb_neighbors):
                if node_id in clusters[neighbor_id]:
                    start_r = (
                        nb_sensors_per_node[updating_node]
                        + ind[neighbor_id] * nb_filters
                    )
                    stop_r = (
                        nb_sensors_per_node[updating_node]
                        + ind[neighbor_id] * nb_filters
                        + nb_filters
                    )

            X_block_updated.append(X_block[node_id] @ X_tilde[start_r:stop_r, :])

        X_block_updated.append(X_tilde[0 : nb_sensors_per_node[updating_node], :])

        for node_id in range(updating_node + 1, nb_nodes):
            for neighbor_id in range(nb_neighbors):
                if node_id in clusters[neighbor_id]:
                    start_r = (
                        nb_sensors_per_node[updating_node]
                        + ind[neighbor_id] * nb_filters
                    )
                    stop_r = (
                        nb_sensors_per_node[updating_node]
                        + ind[neighbor_id] * nb_filters
                        + nb_filters
                    )

            X_block_updated.append(X_block[node_id] @ X_tilde[start_r:stop_r, :])

        return X_block_updated

    def _validate_problem(self):
        problem_inputs = self.data_retriever.get_current_window(window_id=0)
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
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            range(
                1,
                int(self.total_iterations / self.data_window_params.nb_window_reuse)
                + 1,
            ),
            self.normed_error_over_iterations[
                1 :: self.data_window_params.nb_window_reuse
            ],
            color="b",
        )
        ax.set_xlabel(r"Iterations $i$")
        ax.set_ylabel(r"$\varepsilon(i)=\frac{\|X^i-X^*\|_F^2}{\|X^*\|_F^2}$")
        ax.grid(True, which="both")
        return fig

    def plot_iterate_difference(self) -> Figure:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            range(1, self.total_iterations),
            self.normed_difference_over_iterations,
            color="b",
        )
        ax.set_xlabel(r"Iterations $i$")
        ax.set_ylabel(r"$\frac{\|X^i-X^{i-1}\|_F^2}{MQ}$")
        ax.grid(True, which="both")
        return fig

    def plot_objective_error(self) -> Figure:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            range(1, self.total_iterations),
            self.absolute_objective_error_over_iterations,
            color="b",
        )
        ax.set_xlabel(r"Iterations $i$")
        ax.set_ylabel(r"$|f(X^i)-f(X^*)|$")
        ax.grid(True, which="both")
        return fig
