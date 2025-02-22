import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Choose plot backend.
mpl.use("macosx")
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')
# mpl.use("Agg")
from problem_settings import NetworkGraph, ConvergenceParameters
from optimization_problems import GEVDProblem
from data_retriever import GEVDDataRetriever, DataWindowParameters
from dasf import DASF

random_seed = 2025
rng = np.random.default_rng(random_seed)

# Number of nodes
nb_nodes = 30
# Number of channels per node
nb_sensors_per_node = (15 * np.ones(nb_nodes)).astype(int)
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
nb_windows = 200

# Number of filters of X
nb_filters = 5

# Number of times each window will be repeated
nb_window_reuse = 2

data_window_params = DataWindowParameters(
    window_length=nb_samples_per_window,
    nb_window_reuse=nb_window_reuse,
)

gevd_data_retriever = GEVDDataRetriever(
    data_window_params=data_window_params,
    nb_sensors=network_graph.nb_sensors_total,
    nb_sources=nb_filters,
    nb_windows=nb_windows,
    rng=rng,
)
gevd_problem = GEVDProblem(nb_filters=nb_filters)

max_iterations = nb_windows * data_window_params.nb_window_reuse
dasf_convergence_parameters = ConvergenceParameters(max_iterations=max_iterations)

update_path = rng.permutation(range(nb_nodes))

dasf_solver = DASF(
    problem=gevd_problem,
    data_retriever=gevd_data_retriever,
    network_graph=network_graph,
    dasf_convergence_params=dasf_convergence_parameters,
    updating_path=update_path,
    rng=rng,
    dynamic_plot=True,
)
dasf_solver.run()

fig = dasf_solver.plot_error_over_batches()
ax = plt.gca()
ax.set_xscale("linear")
ax2 = ax.twinx()
ax2.plot(
    range(
        1,
        int(dasf_solver.total_iterations / data_window_params.nb_window_reuse) + 1,
    ),
    gevd_data_retriever.weight_function(nb_windows),
    "r",
)
ax2.set_ylabel("Weight function", color="r")
plt.show()
