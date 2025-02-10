import numpy as np
import sys
import matplotlib as mpl
# Choose plot backend.
# mpl.use('macosx')
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt
from time import sleep

sys.path.append('../dasf_toolbox/')
import cca_functions as cca
from dasf_toolbox import dasf_multivar

# Number of Monte-Carlo runs.
mc_runs = 5

# Number of nodes.
nbnodes = 10
# Number of channels per node.
nbsensors_vec = 5 * np.ones(nbnodes)
nbsensors_vec = nbsensors_vec.astype(int)
# Number of channels in total.
nbsensors = np.sum(nbsensors_vec)

# Number of samples of the signals.
nbsamples = 10000

# Number of filters of X.
Q = 2

# Number of variables.
nbvariables = 2

norm_error = []

rng = np.random.default_rng()

for k in range(mc_runs):
    # Create the data.
    Y, V = cca.create_data(nbsensors, nbsamples)

    # Dictionary related to the data of the problem.
    Y_list = [Y]
    V_list = [V]
    Gamma_list = []
    B_list = []
    Glob_Const_list = []

    data_X = {'Y_list': Y_list, 'B_list': B_list,
            'Gamma_list': Gamma_list, 'Glob_Const_list': Glob_Const_list}
    data_W = {'Y_list': V_list, 'B_list': B_list,
              'Gamma_list': Gamma_list, 'Glob_Const_list': Glob_Const_list}

    data = [data_X, data_W]

    # Dictionary related to parameters of the problem.
    prob_params = {'nbnodes': nbnodes, 'nbsensors_vec': nbsensors_vec,
                   'nbsensors': nbsensors, 'Q': Q, 'nbsamples': nbsamples, 'nbvariables': nbvariables}

    # Create adjacency matrix (hollow matrix) of a random graph.
    graph_adj = rng.integers(0, 1, size=(nbnodes, nbnodes), endpoint=True)
    graph_adj = np.triu(graph_adj, 1) + np.tril(graph_adj.T, -1)
    prob_params['graph_adj'] = graph_adj

    # Random updating order.
    update_path = rng.permutation(range(nbnodes))
    prob_params['update_path'] = update_path

    # Estimate filter using the centralized algorithm.
    X_star = cca.cca_solver(prob_params, data, X0=None, solver_params=None)
    f_star = cca.cca_eval(X_star, data)

    prob_params['X_star'] = X_star
    # Compute the distance to X_star if "True".
    prob_params['compare_opt'] = True
    # Show a dynamic plot if "True".
    prob_params['plot_dynamic'] = False

    # Dictionary related to stopping conditions. We fix the number of iterations the DASF algorithm will perform to 200.
    nbiter = 1000
    conv = {'nbiter': nbiter}

    # Solve the CCA in a distributed way using the DASF framework.
    X_est, norm_diff, norm_err, f_seq = dasf_multivar(prob_params, data, cca.cca_solver,
                                             conv, solver_params=None, prob_select_sol=cca.cca_select_sol, prob_eval=cca.cca_eval)

    norm_error.append(norm_err)

    sys.stdout.write('\r')
    j = (k + 1) / mc_runs
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
    sys.stdout.flush()
    sleep(0.25)

sys.stdout.write('\n')


# Plot the normalized error.
q5 = np.quantile(norm_error, 0.5, axis=0)
q25 = np.quantile(norm_error, 0.25, axis=0)
q75 = np.quantile(norm_error, 0.75, axis=0)
iterations = np.arange(1, nbiter + 1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.loglog(iterations, q5, color='b')
ax.fill_between(iterations, q25, q75)
ax.set_xlabel('Iterations')
ax.set_ylabel('Normalized error')
ax.grid(True, which='both')
plt.show()
plt.savefig("cca_convergence.pdf")
