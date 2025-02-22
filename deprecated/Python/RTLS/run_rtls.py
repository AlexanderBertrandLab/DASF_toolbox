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
import pygsp as pg

sys.path.append('../dasf_toolbox/')
import rtls_functions as rtls
from dasf_toolbox import dasf
from dasf_toolbox import fdasf
from dasf_toolbox import dasf_block
import scipy.io

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
Q = 1

norm_error = []
norm_error_fdasf = []

rng = np.random.default_rng()

for k in range(mc_runs):
    # Create the data.
    Y, d = rtls.create_data(nbsensors, nbsamples, Q)
    delta = 1
    rng = np.random.default_rng()
    L = np.diag(1 + rng.normal(loc=0, scale=0.1, size=(nbsensors, )))

    # Dictionary related to the data of the problem.
    Y_list = [Y]
    Gamma_list = [np.identity(nbsensors)]
    B_list = [L]
    Glob_Const_list = [d, delta]

    data = {'Y_list': Y_list, 'B_list': B_list,
            'Gamma_list': Gamma_list, 'Glob_Const_list': Glob_Const_list}

    # Dictionary related to parameters of the problem.
    prob_params = {'nbnodes': nbnodes, 'nbsensors_vec': nbsensors_vec,
                   'nbsensors': nbsensors, 'Q': Q, 'nbsamples': nbsamples}

    # Create adjacency matrix (hollow matrix) of a random graph.
    G = pg.graphs.ErdosRenyi(nbnodes, p=0.8, connected=True)
    graph_adj = G.W.toarray()
    prob_params['graph_adj'] = graph_adj

    # Random updating order.
    update_path = rng.permutation(range(nbnodes))
    prob_params['update_path'] = update_path

    # Estimate filter using the centralized algorithm.
    solver_params = {'tol_X': 1e-8, 'maxiter': 10}
    X_star = rtls.rtls_solver(prob_params, data, X0=None, solver_params=solver_params)
    f_star = rtls.rtls_eval(X_star, data)

    prob_params['X_star'] = X_star
    # Compute the distance to X_star if "True".
    prob_params['compare_opt'] = True
    # Show a dynamic plot if "True".
    prob_params['plot_dynamic'] = False

    # Dictionary related to stopping conditions. We fix the number of iterations the DASF algorithm will perform to 50.
    nbiter = 50
    conv = {'nbiter': nbiter}

    # Solve the RTLS in a distributed way using the DASF algorithm.
    X_est, norm_diff, norm_err, f_seq = dasf(prob_params, data, rtls.rtls_solver,
                                             conv, solver_params, prob_eval=rtls.rtls_eval)

    norm_error.append(norm_err)

    # Solve the RTLS in a distributed way using the F-DASF algorithm.
    X_est_fdasf, norm_diff_fdasf, norm_err_fdasf, f_seq_fdasf = dasf(prob_params, data, rtls.rtls_aux_solver, conv, solver_params, prob_eval=rtls.rtls_eval)
    
    norm_error_fdasf.append(norm_err_fdasf)

    sys.stdout.write('\r')
    j = (k + 1) / mc_runs
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
    sys.stdout.flush()
    sleep(0.25)

sys.stdout.write('\n')

np.savetxt("dasf_error.csv", norm_error, delimiter=",")
np.savetxt("fdasf_error.csv", norm_error_fdasf, delimiter=",")

# Plot the normalized error.
q5 = np.quantile(norm_error, 0.5, axis=0)
q5_fdasf = np.quantile(norm_error_fdasf, 0.5, axis=0)
iterations = np.arange(1, nbiter + 1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.loglog(iterations, q5, color='b', label='DASF')
ax.loglog(iterations, q5_fdasf, color='r', label='F-DASF')
ax.set_ylim([10e-11,10])
ax.set_xlim([1,nbiter])
ax.set_xlabel(r'Iteration index $i$')
ax.set_ylabel(r'MedSE $\epsilon$')
ax.grid(True, which='both')
ax.legend()
plt.show()
plt.savefig("rtls_convergence.pdf")
