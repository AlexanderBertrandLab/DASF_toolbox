import numpy as np
from problem_settings import (
    NetworkGraph,
    ProblemInputs,
    ProblemParameters,
    ConvergenceParameters,
)
from optimization_problems import OptimizationProblem
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DASF:
    def __init__(
        self,
        problem: OptimizationProblem,
        problem_inputs: ProblemInputs,
        network_graph: NetworkGraph,
        convergence_params: ConvergenceParameters,
        problem_params: ProblemParameters,
        updating_path: np.ndarray | None = None,
        initial_estimate: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.problem = problem
        self.problem_inputs = problem_inputs
        self.network_graph = network_graph
        self.convergence_params = convergence_params
        self.problem_params = problem_params
        if updating_path is not None:
            self.updating_path = updating_path
        else:
            self.updating_path = (
                rng.permutation(range(problem_params.nb_samples))
                if rng is not None
                else np.arange(1, network_graph.nb_nodes + 1)
            )

        if initial_estimate is not None:
            self.initial_estimate = initial_estimate
        else:
            self.initial_estimate = (
                rng.standard_normal(
                    (network_graph.nb_sensors_total, problem_params.nb_filters)
                )
                if rng is not None
                else np.random.standard_normal(
                    (network_graph.nb_sensors_total, problem_params.nb_filters)
                )
            )

        self.validate_problem()

    def run(self):
        i = 0
        X = self.initial_estimate
        while i < self.convergence_params.max_iterations:
            # Select updating node
            q = self.updating_path[i % self.network_graph.nb_nodes]

            # Prune the network
            # Find shortest path
            neighbors, path = self.find_path(q)

            # Neighborhood clusters
            clusters = self.find_clusters(neighbors, path)

            # Global - local transition matrix.
            Cq = self.build_Cq(X, q, neighbors, clusters)

            # Compute the compressed data.
            compressed_inputs = self.compress(self.problem_inputs, Cq)

    def find_path(self, q: int) -> Tuple[list, list]:
        """Function finding the neighbors of node q and the shortest path to other every other node in the network.

        INPUTS:

        q: Source node.

        adj (nbnodes x nbnodes): Adjacency (binary) matrix with adj[i,j]=1 if i and j are  connected. Otherwise 0. adj[i,i]=0.

        OUTPUTS:

        neighbors: List containing the neighbors of node q.

        path (nbnodes x 1): List of lists containining at index k the shortest path from node q to node k.
        """
        dist, path = self.shortest_path(q)
        neighbors = [x for x in range(len(path)) if len(path[x]) == 2]
        neighbors.sort()

        return neighbors, path

    def shortest_path(self, q: int) -> Tuple[np.ndarray, list]:
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
        dist[q] = 0

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
            while jmp != q:
                jmp = pred[jmp]
                path_k.insert(0, jmp)

            path.append(path_k)

        return dist, path

    def find_clusters(neighbors: list, path: list) -> list:
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

    def build_Cq(
        self, X: np.ndarray, q: int, neighbors: list, clusters: list
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
        nb_filters = self.problem_params.nb_filters
        nb_neighbors = len(neighbors)

        ind = np.arange(self.network_graph.nb_nodes)

        Cq = np.zeros(
            (
                np.sum(nb_sensors_per_node),
                nb_sensors_per_node[q] + nb_neighbors * nb_filters,
            )
        )
        Cq[:, 0 : nb_sensors_per_node[q]] = np.vstack(
            (
                np.zeros((np.sum(nb_sensors_per_node[0:q]), nb_sensors_per_node[q])),
                np.identity(nb_sensors_per_node[q]),
                np.zeros(
                    (np.sum(nb_sensors_per_node[q + 1 :]), nb_sensors_per_node[q])
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
                    nb_sensors_per_node[q] + ind_k * nb_filters : nb_sensors_per_node[q]
                    + ind_k * nb_filters
                    + nb_filters,
                ] = X_curr

        return Cq

    def compress(self, problem_inputs: ProblemInputs, Cq: np.ndarray) -> ProblemInputs:
        """Function to compress the data.

        INPUTS:

        data: Dictionary related to the data.

        Cq: Transformation matrix making the transition between local and global data.

        OUTPUTS:

        data_compressed: Dictionary containing the compressed data. Contains the same keys as 'data'.
        """
        fused_data = problem_inputs.fused_data
        fused_constants = problem_inputs.fused_constants
        fused_quadratics = problem_inputs.fused_quadratics
        global_constants = problem_inputs.global_constants

        data_compressed = {
            "Y_list": [],
            "B_list": [],
            "Gamma_list": [],
            "Glob_Const_list": [],
        }

        nb_data = len(fused_data)
        data_list_compressed = []
        for ind in range(nb_data):
            data_list_compressed.append(Cq.T @ fused_data[ind])

        nb_constants = len(fused_constants)
        constants_list_compressed = []
        for ind in range(nb_constants):
            constants_list_compressed.append(Cq.T @ fused_constants[ind])

        nb_quadratics = len(fused_quadratics)
        quadratics_list_compressed = []
        for ind in range(nb_quadratics):
            quadratics_list_compressed.append(Cq.T @ fused_quadratics[ind] @ Cq)

        compressed_inputs = ProblemInputs(
            fused_data=data_list_compressed,
            fused_constants=constants_list_compressed,
            fused_quadratics=quadratics_list_compressed,
            global_constants=global_constants,
        )

        return compressed_inputs

    def block_q(X, q, nbsensors_vec):
        """Function to extract the block of X corresponding to node q.

        INPUTS:

        X (nbsensors x Q): Global variable equal to [X1;...;Xq;...;XK].

        q: Updating node.

        nbsensors_vec (nbnodes x nbnodes): Vector containing the number of sensors for each node.

        OUTPUTS:

        Xq (nbsensors_vec(q) x Q): Block of X corresponding to node q.
        """
        M_q = nbsensors_vec[q]
        row_blk = np.cumsum(nbsensors_vec)
        row_blk = np.append(0, row_blk[0:-1])
        row_blk_q = row_blk[q]
        Xq = X[row_blk_q : row_blk_q + M_q, :]

        return Xq

    def validate_problem(self):
        nb_sensor = self.network_graph.nb_sensors_total
        for index, signal in enumerate(self.problem_inputs.fused_data):
            if np.size(signal, 0) != nb_sensor:
                raise ValueError(
                    f"The number of rows in data {index} does not match the number of sensors in the network graph."
                )
        if self.problem_inputs.fused_constants is not None:
            for index, constant in enumerate(self.problem_inputs.fused_constants):
                if np.size(constant, 0) != nb_sensor:
                    raise ValueError(
                        f"The number of rows in the fused constant {index} does not match the number of sensors in the network graph."
                    )
        if self.problem_inputs.fused_quadratics is not None:
            for index, quadratic in enumerate(self.problem_inputs.fused_quadratics):
                if (np.size(quadratic, 0) != nb_sensor) or (
                    np.size(quadratic, 1) != nb_sensor
                ):
                    raise ValueError(
                        f"The number of rows or columns in the fused quadratic {index} does not match the number of sensors in the network graph."
                    )
        if self.initial_estimate.shape != (
            nb_sensor,
            self.problem_params.nb_filters,
        ):
            raise ValueError(
                "The initial estimate does not have the correct shape for the problem."
            )
