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
import tro_functions as tro
from dasf_toolbox import dasf
from dasf_toolbox import fdasf
from dasf_toolbox import dasf_block

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
Q = 5

norm_error = []
norm_error_fdasf = []

rng = np.random.default_rng()

for k in range(mc_runs):
    # Create the data.
    Y, V = tro.create_data(nbsensors, nbsamples)

    # Dictionary related to the data of the problem.
    Y_list = [Y, V]
    Gamma_list = [np.identity(nbsensors)]
    B_list = []
    Glob_Const_list = []

    data = {'Y_list': Y_list, 'B_list': B_list,
            'Gamma_list': Gamma_list, 'Glob_Const_list': Glob_Const_list}

    # Dictionary related to parameters of the problem.
    prob_params = {'nbnodes': nbnodes, 'nbsensors_vec': nbsensors_vec,
                   'nbsensors': nbsensors, 'Q': Q, 'nbsamples': nbsamples}

    # Create adjacency matrix (hollow matrix) of a random graph.
    graph_adj = rng.integers(0, 1, size=(nbnodes, nbnodes), endpoint=True)
    graph_adj = np.triu(graph_adj, 1) + np.tril(graph_adj.T, -1)
    prob_params['graph_adj'] = graph_adj

    # Random updating order.
    update_path = rng.permutation(range(nbnodes))
    prob_params['update_path'] = update_path

    # Estimate filter using the centralized algorithm.
    X_star = tro.tro_solver(prob_params, data)
    f_star = tro.tro_eval(X_star, data)

    prob_params['X_star'] = X_star
    # Compute the distance to X_star if "True".
    prob_params['compare_opt'] = True
    # Show a dynamic plot if "True".
    prob_params['plot_dynamic'] = False

    # Dictionary related to stopping conditions. We fix the number of iterations the DASF algorithm will perform to 200.
    nbiter = 200
    conv = {'nbiter': nbiter}

    # Solve the TRO in a distributed way using the DASF framework.
    X_est, norm_diff, norm_err, f_seq = dasf(prob_params, data, tro.tro_solver,
                                             conv, tro.tro_select_sol, tro.tro_eval)

    norm_error.append(norm_err)

    X_est_fdasf, norm_diff_fdasf, norm_err_fdasf, f_seq_fdasf = fdasf(
        prob_params, data, tro.tro_aux_solver, tro.tro_eval, conv, tro.tro_select_sol)
    
    norm_error_fdasf.append(norm_err_fdasf)

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
q5_fdasf = np.quantile(norm_error_fdasf, 0.5, axis=0)
q25_fdasf = np.quantile(norm_error_fdasf, 0.25, axis=0)
q75_fdasf = np.quantile(norm_error_fdasf, 0.75, axis=0)
iterations = np.arange(1, nbiter + 1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.loglog(iterations, q5, color='b', label='DASF')
ax.fill_between(iterations, q25, q75)
ax.loglog(iterations, q5_fdasf, color='r', label='F-DASF')
ax.fill_between(iterations, q25_fdasf, q75_fdasf)
ax.set_xlabel('Iterations')
ax.set_ylabel('Normalized error')
ax.grid(True, which='both')
ax.legend()
plt.show()
plt.savefig("tro_convergence.pdf")
