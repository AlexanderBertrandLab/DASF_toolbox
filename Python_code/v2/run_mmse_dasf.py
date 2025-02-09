import numpy as np
import matplotlib as mpl

# Choose plot backend.
mpl.use("macosx")
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')
# mpl.use("Agg")
from problem_settings import NetworkGraph, ConvergenceParameters, DataParameters
from optimization_problems import MMSEProblem
from synthetic_data import mmse_generate_synthetic_inputs
from dasf import DASF

random_seed = 2025
rng = np.random.default_rng(random_seed)


nb_nodes = 10

nb_sensors_per_node = (5 * np.ones(nb_nodes)).astype(int)

adjacency_matrix = rng.integers(0, 1, size=(nb_nodes, nb_nodes), endpoint=True)
adjacency_matrix = np.triu(adjacency_matrix, 1) + np.tril(adjacency_matrix.T, -1)
network_graph = NetworkGraph(
    nb_nodes=nb_nodes,
    nb_sensors_per_node=nb_sensors_per_node,
    adjacency_matrix=adjacency_matrix,
)

nb_samples = 10000

nb_filters = 5

mmse_data_params = DataParameters(nb_samples=nb_samples)

mmse_inputs = mmse_generate_synthetic_inputs(
    nb_samples=nb_samples,
    nb_sensors=network_graph.nb_sensors_total,
    rng=rng,
    nb_sources=nb_filters,
)

mmse_problem = MMSEProblem(nb_filters=nb_filters)
X_star = mmse_problem.solve(problem_inputs=mmse_inputs, save_solution=True)
f_star = mmse_problem.evaluate_objective(X_star, problem_inputs=mmse_inputs)

update_path = rng.permutation(range(nb_nodes))
dasf_convergence_parameters = ConvergenceParameters(max_iterations=500)
dasf_mmse_solver = DASF(
    problem=mmse_problem,
    problem_inputs=mmse_inputs,
    network_graph=network_graph,
    dasf_convergence_params=dasf_convergence_parameters,
    data_params=mmse_data_params,
    updating_path=update_path,
    rng=rng,
    dynamic_plot=False,
)


dasf_mmse_solver.run()
