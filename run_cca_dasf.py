import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Choose plot backend.
mpl.use("macosx")
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')
# mpl.use("Agg")
from problem_settings import (
    NetworkGraph,
    ConvergenceParameters,
    get_stationary_setting,
)
from optimization_problems import CCAProblem
from data_retriever import CCADataRetriever
from dasf import DASFMultiVar

random_seed = 2025
rng = np.random.default_rng(random_seed)

# Number of nodes
nb_nodes = 10
# Number of channels per node
nb_sensors_per_node = (10 * np.ones(nb_nodes)).astype(int)
# Create adjacency matrix (hollow matrix) of a random graph
adjacency_matrix = rng.integers(0, 1, size=(nb_nodes, nb_nodes), endpoint=True)
adjacency_matrix = np.triu(adjacency_matrix, 1) + np.tril(adjacency_matrix.T, -1)
network_graph = NetworkGraph(
    nb_nodes=nb_nodes,
    nb_sensors_per_node=nb_sensors_per_node,
    adjacency_matrix=adjacency_matrix,
)

# Number of samples per window of the signals
nb_samples_per_window = 10000

# Number of windows in total
nb_windows = 1

# Number of filters of X
nb_filters = 2

cca_data_retriever = CCADataRetriever(
    nb_samples=nb_samples_per_window,
    nb_sensors=network_graph.nb_sensors_total,
    nb_sources=nb_filters,
    nb_windows=nb_windows,
    rng=rng,
)

cca_problem = CCAProblem(nb_filters=nb_filters)

# DASF solver convergence parameters
dasf_iterations = 300
dasf_convergence_parameters = ConvergenceParameters(max_iterations=dasf_iterations)

data_window_params = get_stationary_setting(
    window_length=nb_samples_per_window, iterations=dasf_iterations
)

update_path = rng.permutation(range(nb_nodes))

# DASF solver convergence parameters
dasf_iterations = 300
dasf_convergence_parameters = ConvergenceParameters(max_iterations=dasf_iterations)

data_window_params = get_stationary_setting(
    window_length=nb_samples_per_window, iterations=dasf_iterations
)

update_path = rng.permutation(range(nb_nodes))

dasf_solver = DASFMultiVar(
    problem=cca_problem,
    data_retriever=cca_data_retriever,
    network_graph=network_graph,
    dasf_convergence_params=dasf_convergence_parameters,
    data_window_params=data_window_params,
    updating_path=update_path,
    rng=rng,
    dynamic_plot=True,
)
dasf_solver.run()

fig = dasf_solver.plot_error()

plt.show()
