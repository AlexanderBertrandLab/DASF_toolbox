import numpy as np
import matplotlib as mpl
# Choose plot backend.
# mpl.use('macosx')
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt
import warnings

# This module implements in a generic way the DASF algorithm.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering
# (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data
# Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be


def dasf(prob_params, data, prob_solver,
         conv=None, solver_params=None, prob_select_sol=None, prob_eval=None):
    """ Function running the DASF algorithm for a given problem.

    INPUTS :

    prob_params : Dictionary related to the problem parameters containing the following keys:

        - nbnodes : Number of nodes in the network.
        - nbsensors_vec : Vector containing the number of sensors for each node.
        - nbsensors : Sum of the number of sensors for each node (dimension of the network-wide signals).
            Is equal to sum(nbsensors_vec).
        - Q : Number of filters to use (dimension of projected space)
        - nbsamples : Number of time samples of the signals per iteration.
        - graph_adj : Adjacency (binary) matrix, with graph_adj[i,j]=1 if i and j are connected.
            Otherwise 0. graph_adj[i,i]=0.
        - update_path : (Optional) Vector of nodes representing the updating path followed by the algorithm.
            If not provided, a random path is created.
        - X_init : (Optional) Initial estimate for X.
        - X_star : (Optional) Optimal argument solving the problem (for comparison, e.g., to compute norm_err).
        - compare_opt : (Optional, binary) If "True" and X_star is given, compute norm_err. "False" by default.
        - plot_dynamic : (Optional, binary) If "True" X_star is given, plot dynamically the first column
            of X_star and the current estimate X. "False" by default.

    data : Dictionary related to the data containing the following keys:

        - Y_list : List containing matrices of size
              (nbsensors x nbsamples) corresponding to the
              stochastic signals.
        - B_list : List containing matrices or vectors with (nbsamples)
              rows corresponding to the constant parameters.
        - Gamma_list : List containing matrices of size
                  (nbsensors x nbsensors) corresponding to the
                  quadratic parameters.
        - Glob_Const_list : List containing the global constants which
                  are not filtered through X.

    prob_solver : Function solving the centralized problem.

    conv:  (Optional) Dictionary related to the convergence and stopping criteria of the algorithm, containing
    the following keys:

        - nbiter : Maximum number of iterations.
        - tol_f : Tolerance in objective: |f^(i+1)-f^(i)|>tol_f
        - tol_X : Tolerance in arguments: ||X^(i+1)-X^(i)||_F>tol_X

    By default, the number of iterations is 200, unless specified otherwise.
    If other fields are given and valid, the first condition to be achieved
    stops the algorithm.

    solver_params: (Optional) Internal parameters for the solver for flexibility.

    prob_select_sol : (Optional) Function resolving the uniqueness ambiguity.

    prob_eval : (Optional) Function evaluating the objective of the problem.

    OUTPUTS :

    X               : Estimation of the optimal variable.

    norm_diff       : Sequence of ||X^(i+1)-X^(i)||_F^2/(nbsensors*Q).

    norm_err        : Sequence of ||X^(i)-X_star||_F^2/||X_star||_F^2.

    f_seq           : Sequence of objective values across iterations.
    """
    rng = np.random.default_rng()
    Q = prob_params['Q']
    nbsensors = prob_params['nbsensors']
    nbnodes = prob_params['nbnodes']
    nbsensors_vec = prob_params['nbsensors_vec']
    graph_adj = prob_params['graph_adj']

    if "update_path" in prob_params:
        update_path = prob_params['update_path']
    else:
        # Random updating order.
        update_path = rng.permutation(range(nbnodes))
        prob_params['update_path'] = update_path

    compare_opt_flag = False
    plot_dynamic_flag = False

    if "X_star" in prob_params:
        X_star = prob_params['X_star']
    else:
        X_star = []
        compare_opt_flag = True
        plot_dynamic_flag = True

    if ("compare_opt" in prob_params) and (not compare_opt_flag):
        compare_opt = prob_params['compare_opt']
    else:
        compare_opt = False

    if ("plot_dynamic" in prob_params) and (not plot_dynamic_flag):
        plot_dynamic = prob_params['plot_dynamic']
    else:
        plot_dynamic = False

    if conv is None:
        tol_f_break = False
        tol_X_break = False
        nbiter = 200
        warnings.warn("Performing 200 iterations")
    elif(("nbiter" in conv and conv['nbiter'] > 0)
         or ("tol_f" in conv and conv['tol_f'] > 0)
         or ("tol_X" in conv['tol_X'] and conv['tol_X'] > 0)):

        if ("nbiter" not in conv or conv['nbiter'] <= 0):
            nbiter = 200
            warnings.warn("Performing at most 200 iterations")
        else:
            nbiter = conv['nbiter']

        if ("tol_f" not in conv or conv['tol_f'] <= 0):
            tol_f_break = False
        else:
            tol_f = conv['tol_f']
            tol_f_break = True

        if ("tol_X" not in conv or conv['tol_X'] <= 0):
            tol_X_break = False
        else:
            tol_X = conv['tol_X']
            tol_X_break = True
    else:
        tol_f_break = False
        tol_X_break = False
        nbiter = 200
        warnings.warn("Performing 200 iterations")

    if "X_init" in prob_params:
        X = prob_params['X_init']
    else:
        X = rng.standard_normal(size=(nbsensors, Q))

    X_old = X[:]

    if prob_eval is None:
        tol_f_break = False
    else:
        f = prob_eval(X,data)

    if plot_dynamic:
        plt.ion()
        fig, ax = plt.subplots()
        line1, = ax.plot(X[:, 1], color='r')
        line2, = ax.plot(X_star[:, 1], color='b')
        plt.axis([0, nbsensors, 1.2 * np.min(X_star[:, 1]), 1.2 * np.max(X_star[:, 1])])
        plt.show()

    i = 0

    norm_diff = []
    f_seq = []

    X_list = []

    while i < nbiter:
        # Select updating node.
        q = update_path[i % nbnodes]

        # Prune the network.
        # Find shortest path.
        neighbors, path = find_path(q, graph_adj)

        # Neighborhood clusters.
        clusters = find_clusters(neighbors, path)

        # Global - local transition matrix.
        Cq = build_Cq(X, q, prob_params, neighbors, clusters)

        # Compute the compressed data.
        data_compressed = compress(data, Cq)

        # Compute the local variable.
        # Solve the local problem with the algorithm for the global problem using the compressed data.
        Xq_old = block_q(X_old, q, nbsensors_vec)
        X_tilde_old = np.concatenate((Xq_old, np.tile(np.eye(Q), (len(neighbors), 1))), axis=0)
        X_tilde = prob_solver(prob_params, data_compressed, X_tilde_old, solver_params)

        # Select a solution among potential ones if the problem has a non-unique solution.
        if prob_select_sol is not None:
            X_tilde = prob_select_sol(X_tilde_old, X_tilde, prob_params, q)

        # Evaluate the objective.
        if prob_eval is not None:
            f_old = f
            f = prob_eval(X_tilde, data_compressed)
            f_seq.append(f)

        # Global variable.
        X = Cq @ X_tilde

        if i > 0:
            norm_diff.append(np.linalg.norm(X - X_old, 'fro') ** 2 / X.size)

        if plot_dynamic:
            if prob_select_sol is not None:
                X_compare = prob_select_sol(X_star, X, nbsensors_vec, q)
            dynamic_plot(X_compare, X_star, line1, line2)

        X_old = X[:]

        i = i + 1

        X_list.append(X)

        if (tol_f_break and np.absolute(f - f_old) <= tol_f) \
                or (tol_X_break and np.linalg.norm(X - X_old, 'fro') <= tol_X):
            break

    if compare_opt:
        # Resolve uniqueness ambiguity on X_star for comparison
        if prob_select_sol is not None:
            X_star = prob_select_sol(X, X_star, prob_params, q)

        total_iterations = len(X_list)
        norm_err = [None] * total_iterations
        for k in range(total_iterations):
            norm_err[k] = (np.linalg.norm(X_list[k] - X_star, 'fro') ** 2
                        / np.linalg.norm(X_star, 'fro') ** 2)

    if plot_dynamic:
        plt.ioff()
        #plt.show(block=False)
        plt.close()

    return X, norm_diff, norm_err, f_seq


def find_path(q, graph_adj):
    """ Function finding the neighbors of node q and the shortest path to other every other node in the network.

    INPUTS:

    q: Source node.

    adj (nbnodes x nbnodes): Adjacency (binary) matrix where K is the number of nodes
    in the network with adj[i,j]=1 if i and j are  connected. Otherwise 0. adj[i,i]=0.

    OUTPUTS:

    neighbors: List containing the neighbors of node q.

    path (nbnodes x 1): List of lists containining at index k the shortest path from node q to node k.
    """
    dist, path = shortest_path(q, graph_adj)
    neighbors = [x for x in range(len(path)) if len(path[x]) == 2]
    neighbors.sort()

    return neighbors, path


def shortest_path(q, graph_adj):
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
    nbnodes = np.size(graph_adj, 0)
    dist = np.inf * np.ones(nbnodes)
    dist[q] = 0

    visited = []
    pred = np.zeros(nbnodes, dtype=int)

    def diff(l1, l2):
        return [x for x in l1 if x not in l2]

    def intersect(l1, l2):
        return [x for x in l1 if x in l2]

    unvisited = diff(list(range(nbnodes)), visited)
    path = []

    while len(visited) < nbnodes:
        inds = np.argwhere(dist == np.min(dist[unvisited])).T[0]

        for ind in inds:
            visited.append(ind)
            unvisited = diff(list(range(nbnodes)), visited)
            neighbors_ind = [i for i, x in enumerate(graph_adj[ind, :]) if x == 1]
            for m in intersect(neighbors_ind, unvisited):
                if dist[ind] + 1 < dist[m]:
                    dist[m] = dist[ind] + 1
                    pred[m] = ind

    for k in range(nbnodes):
        jmp = k
        path_k = [k]
        while jmp != q:
            jmp = pred[jmp]
            path_k.insert(0, jmp)

        path.append(path_k)

    return dist, path


def find_clusters(neighbors, path):
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


def build_Cq(X, q, prob_params, neighbors, clusters):
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
    nbnodes = prob_params['nbnodes']
    nbsensors_vec = prob_params['nbsensors_vec']
    Q = prob_params['Q']
    nbneighbors = len(neighbors)

    ind = np.arange(nbnodes)

    Cq = np.zeros((np.sum(nbsensors_vec), nbsensors_vec[q] + nbneighbors * Q))
    Cq[:, 0:nbsensors_vec[q]] = np.vstack((np.zeros((np.sum(nbsensors_vec[0:q]), nbsensors_vec[q])),
                                           np.identity(nbsensors_vec[q]),
                                           np.zeros((np.sum(nbsensors_vec[q + 1:]), nbsensors_vec[q]))))
    for k in range(nbneighbors):
        ind_k = ind[k]
        for n in range(len(clusters[k])):
            clusters_k = clusters[k]
            l = clusters_k[n]
            X_curr = X[np.sum(nbsensors_vec[0:l]):np.sum(nbsensors_vec[0:l + 1]), :]
            Cq[np.sum(nbsensors_vec[0:l]):np.sum(nbsensors_vec[0:l + 1]),
            nbsensors_vec[q] + ind_k * Q: nbsensors_vec[q] + ind_k * Q + Q] = X_curr

    return Cq


def compress(data, Cq):
    """Function to compress the data.

    INPUTS:

    data: Dictionary related to the data.

    Cq: Transformation matrix making the transition between local and global data.

    OUTPUTS:

    data_compressed: Dictionary containing the compressed data. Contains the same keys as 'data'.
    """
    Y_list = data['Y_list']
    B_list = data['B_list']
    Gamma_list = data['Gamma_list']
    Glob_Const_list = data['Glob_Const_list']

    data_compressed = {'Y_list': [], 'B_list': [],
                       'Gamma_list': [], 'Glob_Const_list': []}

    nbsignals = len(Y_list)
    Y_list_compressed = []
    for ind in range(nbsignals):
        Y_list_compressed.append(Cq.T @ Y_list[ind])

    data_compressed['Y_list'] = Y_list_compressed

    nblin = len(B_list)
    B_list_compressed = []
    for ind in range(nblin):
        B_list_compressed.append(Cq.T @ B_list[ind])

    data_compressed['B_list'] = B_list_compressed

    nbquadr = len(Gamma_list)
    Gamma_list_compressed = []
    for ind in range(nbquadr):
        Gamma_list_compressed.append(Cq.T @ Gamma_list[ind] @ Cq)

    data_compressed['Gamma_list'] = Gamma_list_compressed

    data_compressed['Glob_Const_list'] = Glob_Const_list

    return data_compressed


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
    Xq = X[row_blk_q:row_blk_q + M_q, :]

    return Xq


def dynamic_plot(X, X_star, line1, line2):
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


def update_X_block(X_block, X_tilde, q, prob_params, neighbors, clusters, prob_select_sol):
    """ Function to update the cell containing the blocks of X for each corresponding node.

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
    nbnodes = prob_params['nbnodes']
    nbsensors_vec = prob_params['nbsensors_vec']
    Q = prob_params['Q']

    if prob_select_sol is not None:
        Xq_old = X_block[q]
        X_tilde_old = np.concatenate((Xq_old, np.tile(np.eye(Q), (len(neighbors), 1))), axis=0)
        X_tilde = prob_select_sol(X_tilde_old, X_tilde, prob_params, q)

    X_block_upd = []

    nbneighbors = len(neighbors)
    ind = np.arange(nbnodes)

    for l in range(q):
        for k in range(nbneighbors):
            if l in clusters[k]:
                start_r = nbsensors_vec[q] + ind[k] * Q
                stop_r = nbsensors_vec[q] + ind[k] * Q + Q

        X_block_upd.append(X_block[l] @ X_tilde[start_r: stop_r,:])

    X_block_upd.append(X_tilde[0:nbsensors_vec[q],:])

    for l in range(q + 1,nbnodes):
        for k in range(nbneighbors):
            if l in clusters[k]:
                start_r = nbsensors_vec[q] + ind[k] * Q
                stop_r = nbsensors_vec[q] + ind[k] * Q + Q

        X_block_upd.append(X_block[l] @ X_tilde[start_r: stop_r,:])

    return X_block_upd


def dasf_block(prob_params, data, prob_solver,
         conv=None, solver_params=None, prob_select_sol=None, prob_eval=None):
    """ Function running the DASF algorithm for a given problem.

    INPUTS :

    prob_params : Dictionary related to the problem parameters containing the following keys:

        - nbnodes : Number of nodes in the network.
        - nbsensors_vec : Vector containing the number of sensors for each node.
        - nbsensors : Sum of the number of sensors for each node (dimension of the network-wide signals).
            Is equal to sum(nbsensors_vec).
        - Q : Number of filters to use (dimension of projected space)
        - nbsamples : Number of time samples of the signals per iteration.
        - graph_adj : Adjacency (binary) matrix, with graph_adj[i,j]=1 if i and j are connected.
            Otherwise 0. graph_adj[i,i]=0.
        - update_path : (Optional) Vector of nodes representing the updating path followed by the algorithm.
            If not provided, a random path is created.
        - X_init : (Optional) Initial estimate for X.
        - X_star : (Optional) Optimal argument solving the problem (for comparison, e.g., to compute norm_err).
        - compare_opt : (Optional, binary) If "True" and X_star is given, compute norm_err. "False" by default.
        - plot_dynamic : (Optional, binary) If "True" X_star is given, plot dynamically the first column
            of X_star and the current estimate X. "False" by default.

    data : Dictionary related to the data containing the following keys:

        - Y_list : List containing matrices of size
              (nbsensors x nbsamples) corresponding to the
              stochastic signals.
        - B_list : List containing matrices or vectors with (nbsamples)
              rows corresponding to the constant parameters.
        - Gamma_list : List containing matrices of size
                  (nbsensors x nbsensors) corresponding to the
                  quadratic parameters.
        - Glob_Const_list : List containing the global constants which
                  are not filtered through X.

    prob_solver : Function solving the centralized problem.

    conv:  (Optional) Dictionary related to the convergence and stopping criteria of the algorithm, containing
    the following keys:

        - nbiter : Maximum number of iterations.
        - tol_f : Tolerance in objective: |f^(i+1)-f^(i)|>tol_f
        - tol_X : Tolerance in arguments: ||X^(i+1)-X^(i)||_F>tol_f

    By default, the number of iterations is 200, unless specified otherwise.
    If other fields are given and valid, the first condition to be achieved
    stops the algorithm.

    solver_params: (Optional) Internal parameters for the solver for flexibility.

    prob_select_sol : (Optional) Function resolving the uniqueness ambiguity.

    prob_eval : (Optional) Function evaluating the objective of the problem.

    OUTPUTS :

    X               : Estimation of the optimal variable.

    norm_diff       : Sequence of ||X^(i+1)-X^(i)||_F^2/(nbsensors*Q).

    norm_err        : Sequence of ||X^(i)-X_star||_F^2/||X_star||_F^2.

    f_seq           : Sequence of objective values across iterations.
    """
    rng = np.random.default_rng()
    Q = prob_params['Q']
    nbsensors = prob_params['nbsensors']
    nbnodes = prob_params['nbnodes']
    nbsensors_vec = prob_params['nbsensors_vec']
    graph_adj = prob_params['graph_adj']

    if "update_path" in prob_params:
        update_path = prob_params['update_path']
    else:
        # Random updating order.
        update_path = rng.permutation(range(nbnodes))
        prob_params['update_path'] = update_path

    compare_opt_flag = False
    plot_dynamic_flag = False

    if "X_star" in prob_params:
        X_star = prob_params['X_star']
    else:
        X_star = []
        compare_opt_flag = True
        plot_dynamic_flag = True

    if ("compare_opt" in prob_params) and (not compare_opt_flag):
        compare_opt = prob_params['compare_opt']
    else:
        compare_opt = False

    if ("plot_dynamic" in prob_params) and (not plot_dynamic_flag):
        plot_dynamic = prob_params['plot_dynamic']
    else:
        plot_dynamic = False

    if conv is None:
        tol_f_break = False
        tol_X_break = False
        nbiter = 200
        warnings.warn("Performing 200 iterations")
    elif(("nbiter" in conv and conv['nbiter'] > 0)
         or ("tol_f" in conv and conv['tol_f'] > 0)
         or ("tol_X" in conv['tol_X'] and conv['tol_X'] > 0)):

        if ("nbiter" not in conv or conv['nbiter'] <= 0):
            nbiter = 200
            warnings.warn("Performing at most 200 iterations")
        else:
            nbiter = conv['nbiter']

        if ("tol_f" not in conv or conv['tol_f'] <= 0):
            tol_f_break = False
        else:
            tol_f = conv['tol_f']
            tol_f_break = True

        if ("tol_X" not in conv or conv['tol_X'] <= 0):
            tol_X_break = False
        else:
            tol_X = conv['tol_X']
            tol_X_break = True
    else:
        tol_f_break = False
        tol_X_break = False
        nbiter = 200
        warnings.warn("Performing 200 iterations")

    if "X_init" in prob_params:
        X = prob_params['X_init']
    else:
        X = rng.standard_normal(size=(nbsensors, Q))

    X_old = X[:]
    X_block = np.vsplit(X, np.cumsum(nbsensors_vec)[:-1])

    if prob_eval is None:
        tol_f_break = False
    else:
        f = prob_eval(X,data)

    if plot_dynamic:
        plt.ion()
        fig, ax = plt.subplots()
        line1, = ax.plot(X[:, 1], color='r')
        line2, = ax.plot(X_star[:, 1], color='b')
        plt.axis([0, nbsensors, 1.2 * np.min(X_star[:, 1]), 1.2 * np.max(X_star[:, 1])])
        plt.show()

    i = 0

    norm_diff = []
    f_seq = []

    X_list = []

    while i < nbiter:
        # Select updating node.
        q = update_path[i % nbnodes]

        # Prune the network.
        # Find shortest path.
        neighbors, path = find_path(q, graph_adj)

        # Neighborhood clusters.
        clusters = find_clusters(neighbors, path)

        # Global - local transition matrix.
        Cq = build_Cq(X, q, prob_params, neighbors, clusters)

        # Compute the compressed data.
        data_compressed = compress(data, Cq)

        # Compute the local variable.
        # Solve the local problem with the algorithm for the global problem using the compressed data.
        Xq_old = block_q(X_old, q, nbsensors_vec)
        X_tilde_old = np.concatenate((Xq_old, np.tile(np.eye(Q), (len(neighbors), 1))), axis=0)
        X_tilde = prob_solver(prob_params, data_compressed, X_tilde_old, solver_params)

        # Evaluate the objective.
        if prob_eval is not None:
            f_old = f
            f = prob_eval(X_tilde, data_compressed)
            f_seq.append(f)

        # Global variable.
        X_block = update_X_block(X_block, X_tilde, q, prob_params, neighbors, clusters, prob_select_sol)
        X = np.vstack(X_block)

        if i > 0:
            norm_diff.append(np.linalg.norm(X - X_old, 'fro') ** 2 / X.size)

        if plot_dynamic:
            if prob_select_sol is not None:
                X_compare = prob_select_sol(X_star, X, prob_params, q)
            dynamic_plot(X_compare, X_star, line1, line2)

        X_old = X[:]
        X_block = np.vsplit(X, np.cumsum(nbsensors_vec)[:-1])

        i = i + 1

        X_list.append(X)

        if (tol_f_break and np.absolute(f - f_old) <= tol_f) \
                or (tol_X_break and np.linalg.norm(X - X_old, 'fro') <= tol_X):
            break

    if compare_opt:
        # Resolve uniqueness ambiguity on X_star for comparison
        if prob_select_sol is not None:
            X_star = prob_select_sol(X, X_star, prob_params, q)

        total_iterations = len(X_list)
        norm_err = [None] * total_iterations
        for k in range(total_iterations):
            norm_err[k] = (np.linalg.norm(X_list[k] - X_star, 'fro') ** 2
                           / np.linalg.norm(X_star, 'fro') ** 2)

    if plot_dynamic:
        plt.ioff()
        #plt.show(block=False)
        plt.close()

    return X, norm_diff, norm_err, f_seq


def dasf_multivar(prob_params, data, prob_solver,
         conv=None, solver_params=None, prob_select_sol=None, prob_eval=None):
    """ Function running the DASF for a given problem.

    INPUTS :

    prob_params : Dictionary related to the problem parameters containing the following keys:

        - nbnodes : Number of nodes in the network.
        - nbsensors_vec : Vector containing the number of sensors for each node.
        - nbsensors : Sum of the number of sensors for each node (dimension of the network-wide signals).
            Is equal to sum(nbsensors_vec).
        - Q : Number of filters to use (dimension of projected space)
        - nbsamples : Number of time samples of the signals per iteration.
        - nbvariables : Number of variables.
        - graph_adj : Adjacency (binary) matrix, with graph_adj[i,j]=1 if i and j are connected.
            Otherwise 0. graph_adj[i,i]=0.
        - update_path : (Optional) Vector of nodes representing the updating path followed by the algorithm.
            If not provided, a random path is created.
        - X_init : (Optional) Initial estimate for X.
        - X_star : (Optional) Optimal argument solving the problem (for comparison, e.g., to compute norm_err).
        - compare_opt : (Optional, binary) If "True" and X_star is given, compute norm_err. "False" by default.
        - plot_dynamic : (Optional, binary) If "True" X_star is given, plot dynamically the first column
            of X_star and the current estimate X. "False" by default.

    data : Dictionary related to the data containing the following keys:

        - Y_list : List containing matrices of size
              (nbsensors x nbsamples) corresponding to the
              stochastic signals.
        - B_list : List containing matrices or vectors with (nbsamples)
              rows corresponding to the constant parameters.
        - Gamma_list : List containing matrices of size
                  (nbsensors x nbsensors) corresponding to the
                  quadratic parameters.
        - Glob_Const_list : List containing the global constants which
                  are not filtered through X.

    prob_solver : Function solving the centralized problem.

    conv:  (Optional) Dictionary related to the convergence and stopping criteria of the algorithm, containing
    the following keys:

        - nbiter : Maximum number of iterations.
        - tol_f : Tolerance in objective: |f^(i+1)-f^(i)|>tol_f
        - tol_X : Tolerance in arguments: ||X^(i+1)-X^(i)||_F>tol_X

    By default, the number of iterations is 200, unless specified otherwise.
    If other fields are given and valid, the first condition to be achieved
    stops the algorithm.

    solver_params: (Optional) Internal parameters for the solver for flexibility.

    prob_select_sol : (Optional) Function resolving the uniqueness ambiguity.

    prob_eval : (Optional) Function evaluating the objective of the problem.

    OUTPUTS :

    X               : Estimation of the optimal variable.

    norm_diff       : Sequence of ||X^(i+1)-X^(i)||_F^2/(nbsensors*Q).

    norm_err        : Sequence of ||X^(i)-X_star||_F^2/||X_star||_F^2.

    f_seq           : Sequence of objective values across iterations.
    """
    rng = np.random.default_rng()
    Q = prob_params['Q']
    nbsensors = prob_params['nbsensors']
    nbnodes = prob_params['nbnodes']
    nbsensors_vec = prob_params['nbsensors_vec']
    nbvariables = prob_params['nbvariables']
    graph_adj = prob_params['graph_adj']

    if "update_path" in prob_params:
        update_path = prob_params['update_path']
    else:
        # Random updating order.
        update_path = rng.permutation(range(nbnodes))
        prob_params['update_path'] = update_path

    compare_opt_flag = False
    plot_dynamic_flag = False

    if "X_star" in prob_params:
        X_star = prob_params['X_star']
    else:
        X_star = []
        compare_opt_flag = True
        plot_dynamic_flag = True

    if ("compare_opt" in prob_params) and (not compare_opt_flag):
        compare_opt = prob_params['compare_opt']
    else:
        compare_opt = False

    if ("plot_dynamic" in prob_params) and (not plot_dynamic_flag):
        plot_dynamic = prob_params['plot_dynamic']
    else:
        plot_dynamic = False

    if conv is None:
        tol_f_break = False
        tol_X_break = False
        nbiter = 200
        warnings.warn("Performing 200 iterations")
    elif(("nbiter" in conv and conv['nbiter'] > 0)
         or ("tol_f" in conv and conv['tol_f'] > 0)
         or ("tol_X" in conv['tol_X'] and conv['tol_X'] > 0)):

        if ("nbiter" not in conv or conv['nbiter'] <= 0):
            nbiter = 200
            warnings.warn("Performing at most 200 iterations")
        else:
            nbiter = conv['nbiter']

        if ("tol_f" not in conv or conv['tol_f'] <= 0):
            tol_f_break = False
        else:
            tol_f = conv['tol_f']
            tol_f_break = True

        if ("tol_X" not in conv or conv['tol_X'] <= 0):
            tol_X_break = False
        else:
            tol_X = conv['tol_X']
            tol_X_break = True
    else:
        tol_f_break = False
        tol_X_break = False
        nbiter = 200
        warnings.warn("Performing 200 iterations")

    if "X_init" in prob_params:
        X = prob_params['X_init']
    else:
        X = []
        for k in range(nbvariables):
            X_k = rng.standard_normal(size=(nbsensors, Q))
            X.append(X_k)

    X_old = X[:]

    if prob_eval is None:
        tol_f_break = False
    else:
        f = prob_eval(X,data)

    if plot_dynamic:
        plt.ion()
        fig, ax = plt.subplots()
        line1, = ax.plot(np.vstack(X)[:, 1], color='r')
        line2, = ax.plot(np.vstack(X_star)[:, 1], color='b')
        plt.axis([0, nbsensors, 1.2 * np.min(np.vstack(X_star)[:, 1]), 1.2 * np.max(np.vstack(X_star)[:, 1])])
        plt.show()

    i = 0

    norm_diff = []
    f_seq = []

    X_list = []

    while i < nbiter:
        # Select updating node.
        q = update_path[i % nbnodes]

        # Prune the network.
        # Find shortest path.
        neighbors, path = find_path(q, graph_adj)

        # Neighborhood clusters.
        clusters = find_clusters(neighbors, path)

        # Global - local transition matrix.
        Cq = []
        for k in range(nbvariables):
            Cq_k = build_Cq(X[k], q, prob_params, neighbors, clusters)
            Cq.append(Cq_k)

        # Compute the compressed data.
        data_compressed = []
        for k in range(nbvariables):
            data_compressed_k = compress(data[k], Cq[k])
            data_compressed.append(data_compressed_k)

        # Compute the local variable.
        # Solve the local problem with the algorithm for the global problem using the compressed data.
        X_tilde_old = []
        for k in range(nbvariables):
            Xq_old = block_q(X_old[k], q, nbsensors_vec)
            X_tilde_old_k = np.concatenate((Xq_old, np.tile(np.eye(Q), (len(neighbors), 1))), axis=0)
            X_tilde_old.append(X_tilde_old_k)

        X_tilde = prob_solver(prob_params, data_compressed, X_tilde_old, solver_params)

        # Select a solution among potential ones if the problem has a non-unique solution.
        if prob_select_sol is not None:
            X_tilde = prob_select_sol(X_tilde_old, X_tilde, prob_params, q)

        # Evaluate the objective.
        if prob_eval is not None:
            f_old = f
            f = prob_eval(X_tilde, data_compressed)
            f_seq.append(f)

        # Global variable.
        for k in range(nbvariables):
            X[k] = Cq[k] @ X_tilde[k]

        if i > 0:
            norm_diff.append(np.linalg.norm(np.vstack(X) - np.vstack(X_old), 'fro')**2
                             / np.vstack(X).size)

        if plot_dynamic:
            if prob_select_sol is not None:
                X_compare = prob_select_sol(X_star, X, prob_params, q)
            dynamic_plot(np.vstack(X_compare), np.vstack(X_star), line1, line2)

        X_old = X[:]

        i = i + 1

        X_list.append(X[:])

        if (tol_f_break and np.absolute(f - f_old) <= tol_f) \
                or (tol_X_break and np.linalg.norm(np.vstack(X) - np.vstack(X_old), 'fro')
                                                   <= tol_X):
            break

    if compare_opt:
        # Resolve uniqueness ambiguity on X_star for comparison
        if prob_select_sol is not None:
            X_star = prob_select_sol(X, X_star, prob_params, q)

        total_iterations = len(X_list)
        norm_err = [None] * total_iterations
        for k in range(total_iterations):
            norm_err[k] = (np.linalg.norm(np.vstack(X_list[k]) - np.vstack(X_star), 'fro') ** 2
                           / np.linalg.norm(np.vstack(X_star), 'fro') ** 2)

    if plot_dynamic:
        plt.ioff()
        #plt.show(block=False)
        plt.close()

    return X, norm_diff, norm_err, f_seq


def fdasf(prob_params, data, prob_aux_solver, prob_eval,
         conv=None, solver_params=None, prob_select_sol=None):
    """ Function running the F-DASF algorithm for a given problem.

    INPUTS :

    prob_params : Dictionary related to the problem parameters containing the following keys:

        - nbnodes : Number of nodes in the network.
        - nbsensors_vec : Vector containing the number of sensors for each node.
        - nbsensors : Sum of the number of sensors for each node (dimension of the network-wide signals).
            Is equal to sum(nbsensors_vec).
        - Q : Number of filters to use (dimension of projected space)
        - nbsamples : Number of time samples of the signals per iteration.
        - graph_adj : Adjacency (binary) matrix, with graph_adj[i,j]=1 if i and j are connected.
            Otherwise 0. graph_adj[i,i]=0.
        - update_path : (Optional) Vector of nodes representing the updating path followed by the algorithm.
            If not provided, a random path is created.
        - X_init : (Optional) Initial estimate for X.
        - X_star : (Optional) Optimal argument solving the problem (for comparison, e.g., to compute norm_err).
        - compare_opt : (Optional, binary) If "True" and X_star is given, compute norm_err. "False" by default.
        - plot_dynamic : (Optional, binary) If "True" X_star is given, plot dynamically the first column
            of X_star and the current estimate X. "False" by default.

    data : Dictionary related to the data containing the following keys:

        - Y_list : List containing matrices of size
              (nbsensors x nbsamples) corresponding to the
              stochastic signals.
        - B_list : List containing matrices or vectors with (nbsamples)
              rows corresponding to the constant parameters.
        - Gamma_list : List containing matrices of size
                  (nbsensors x nbsensors) corresponding to the
                  quadratic parameters.
        - Glob_Const_list : List containing the global constants which
                  are not filtered through X.

    prob_aux_solver : Function solving the auxiliary problem.

    prob_eval : Function evaluating the objective of the problem.

    conv:  (Optional) Dictionary related to the convergence and stopping criteria of the algorithm, containing
    the following keys:

        - nbiter : Maximum number of iterations.
        - tol_rho : Tolerance in objective: |rho^(i+1)-rho^(i)|>tol_rho
        - tol_X : Tolerance in arguments: ||X^(i+1)-X^(i)||_F>tol_X

    By default, the number of iterations is 200, unless specified otherwise.
    If other fields are given and valid, the first condition to be achieved
    stops the algorithm.

    solver_params: (Optional) Internal parameters for the solver for flexibility.

    prob_select_sol : (Optional) Function resolving the uniqueness ambiguity.

    OUTPUTS :

    X               : Estimation of the optimal variable.

    norm_diff       : Sequence of ||X^(i+1)-X^(i)||_F^2/(nbsensors*Q).

    norm_err        : Sequence of ||X^(i)-X_star||_F^2/||X_star||_F^2.

    rho_seq           : Sequence of objective values across iterations.
    """
    rng = np.random.default_rng()
    Q = prob_params['Q']
    nbsensors = prob_params['nbsensors']
    nbnodes = prob_params['nbnodes']
    nbsensors_vec = prob_params['nbsensors_vec']
    graph_adj = prob_params['graph_adj']

    if "update_path" in prob_params:
        update_path = prob_params['update_path']
    else:
        # Random updating order.
        update_path = rng.permutation(range(nbnodes))
        prob_params['update_path'] = update_path

    compare_opt_flag = False
    plot_dynamic_flag = False

    if "X_star" in prob_params:
        X_star = prob_params['X_star']
    else:
        X_star = []
        compare_opt_flag = True
        plot_dynamic_flag = True

    if ("compare_opt" in prob_params) and (not compare_opt_flag):
        compare_opt = prob_params['compare_opt']
    else:
        compare_opt = False

    if ("plot_dynamic" in prob_params) and (not plot_dynamic_flag):
        plot_dynamic = prob_params['plot_dynamic']
    else:
        plot_dynamic = False

    if conv is None:
        tol_rho_break = False
        tol_X_break = False
        nbiter = 200
        warnings.warn("Performing 200 iterations")
    elif(("nbiter" in conv and conv['nbiter'] > 0)
         or ("tol_rho" in conv and conv['tol_rho'] > 0)
         or ("tol_X" in conv['tol_X'] and conv['tol_X'] > 0)):

        if ("nbiter" not in conv or conv['nbiter'] <= 0):
            nbiter = 200
            warnings.warn("Performing at most 200 iterations")
        else:
            nbiter = conv['nbiter']

        if ("tol_rho" not in conv or conv['tol_rho'] <= 0):
            tol_rho_break = False
        else:
            tol_rho = conv['tol_f']
            tol_rho_break = True

        if ("tol_X" not in conv or conv['tol_X'] <= 0):
            tol_X_break = False
        else:
            tol_X = conv['tol_X']
            tol_X_break = True
    else:
        tol_rho_break = False
        tol_X_break = False
        nbiter = 200
        warnings.warn("Performing 200 iterations")

    if "X_init" in prob_params:
        X = prob_params['X_init']
    else:
        X = rng.standard_normal(size=(nbsensors, Q))

    X_old = X[:]

    if prob_eval is None:
        tol_rho_break = False
    else:
        rho = prob_eval(X,data)

    if plot_dynamic:
        plt.ion()
        fig, ax = plt.subplots()
        line1, = ax.plot(X[:, 1], color='r')
        line2, = ax.plot(X_star[:, 1], color='b')
        plt.axis([0, nbsensors, 1.2 * np.min(X_star[:, 1]), 1.2 * np.max(X_star[:, 1])])
        plt.show()

    i = 0

    norm_diff = []
    rho_seq = []

    X_list = []

    while i < nbiter:
        # Select updating node.
        q = update_path[i % nbnodes]

        # Prune the network.
        # Find shortest path.
        neighbors, path = find_path(q, graph_adj)

        # Neighborhood clusters.
        clusters = find_clusters(neighbors, path)

        # Global - local transition matrix.
        Cq = build_Cq(X, q, prob_params, neighbors, clusters)

        # Compute the compressed data.
        data_compressed = compress(data, Cq)

        # Compute the local variable.
        # Solve the local problem with the algorithm for the global problem using the compressed data.
        Xq_old = block_q(X_old, q, nbsensors_vec)
        X_tilde_old = np.concatenate((Xq_old, np.tile(np.eye(Q), (len(neighbors), 1))), axis=0)
        X_tilde = prob_aux_solver(prob_params, data_compressed, X_tilde_old, solver_params)

        # Select a solution among potential ones if the problem has a non-unique solution.
        if prob_select_sol is not None:
            X_tilde = prob_select_sol(X_tilde_old, X_tilde, prob_params, q)

        # Evaluate the objective.
        if prob_eval is not None:
            rho_old = rho
            rho = prob_eval(X_tilde, data_compressed)
            rho_seq.append(rho)

        # Global variable.
        X = Cq @ X_tilde

        if i > 0:
            norm_diff.append(np.linalg.norm(X - X_old, 'fro')**2 / X.size)

        if plot_dynamic:
            if prob_select_sol is not None:
                X_compare = prob_select_sol(X_star, X, nbsensors_vec, q)
            dynamic_plot(X_compare, X_star, line1, line2)

        X_old = X[:]

        i = i + 1

        X_list.append(X)

        if (tol_rho_break and np.absolute(rho - rho_old) <= tol_rho) \
                or (tol_X_break and np.linalg.norm(X - X_old, 'fro') <= tol_X):
            break

    if compare_opt:
        # Resolve uniqueness ambiguity on X_star for comparison
        if prob_select_sol is not None:
            X_star = prob_select_sol(X, X_star, prob_params, q)

        total_iterations = len(X_list)
        norm_err = [None] * total_iterations
        for k in range(total_iterations):
            norm_err[k] = (np.linalg.norm(X_list[k] - X_star, 'fro') ** 2
                        / np.linalg.norm(X_star, 'fro') ** 2)

    if plot_dynamic:
        plt.ioff()
        #plt.show(block=False)
        plt.close()

    return X, norm_diff, norm_err, rho_seq