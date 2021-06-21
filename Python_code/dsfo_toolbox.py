import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

mpl.use('macosx')


#mpl.use('Qt5Agg')
# mpl.use('TkAgg')

def dsfo(prob_params, data, prob_solver,
         conv=None, prob_select_sol=None, prob_eval=None):
    rng = np.random.default_rng()
    Q = prob_params['Q']
    nbsensors = prob_params['nbsensors']
    nbnodes = prob_params['nbnodes']
    nbsensors_vec = prob_params['nbsensors_vec']
    graph_adj = prob_params['graph_adj']

    if "update_path" in prob_params:
        update_path = prob_params['update_path']
    else:
        update_path = rng.permutation(range(nbnodes))
        prob_params['update_path'] = update_path

    if "X_star" in prob_params:
        X_star = prob_params['X_star']
    else:
        X_star = []

    if "compare_opt" in prob_params:
        compare_opt = prob_params['compare_opt']
    else:
        compare_opt = False

    if "plot_dynamic" in prob_params:
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

    X = rng.standard_normal(size=(nbsensors, Q))
    X_old = X

    if prob_eval is None:
        tol_f_break = False
    else:
        f = prob_eval(X,data)

    if len(X_star) and compare_opt and plot_dynamic:
        plt.ion()
        fig, ax = plt.subplots()
        line1, = ax.plot(X[:, 1], color='r')
        line2, = ax.plot(X_star[:, 1], color='b')
        plt.axis([0, nbsensors, 1.2 * np.min(X_star[:, 1]), 1.2 * np.max(X_star[:, 1])])
        plt.show()

    i = 0

    norm_diff = []
    norm_err = []
    f_seq = []

    while i < nbiter:
        q = update_path[i % nbnodes]

        neighbors, path = find_path(q, graph_adj)

        Nu = constr_Nu(neighbors, path)

        Cq = constr_Cq(X, q, prob_params, neighbors, Nu)

        data_compressed = compress(data, Cq)

        X_tilde = prob_solver(prob_params, data_compressed)

        if prob_select_sol is not None:
            Xq = X_tilde[0:nbsensors_vec[q], :]
            Xq_old = block_q(X_old, q, nbsensors_vec)
            X_tilde = prob_select_sol(Xq_old, Xq, X_tilde)

        if prob_eval is not None:
            f_old = f
            f = prob_eval(X_tilde, data_compressed)
            f_seq.append(f)

        X = Cq @ X_tilde

        if i > 0:
            norm_diff.append(np.linalg.norm(X - X_old, 'fro')**2 / X.size)

        if len(X_star) and compare_opt:
            if prob_select_sol is not None:
                Xq = X_tilde[0:nbsensors_vec[q], :]
                Xq_star = block_q(X_star, q, nbsensors_vec)
                X = prob_select_sol(Xq_star, Xq, X)
            norm_err.append(np.linalg.norm(X - X_star, 'fro')**2
                            / np.linalg.norm(X_star,'fro')**2)
            if plot_dynamic:
                dynamic_plot(X, X_star, line1, line2)

        X_old = X

        i = i + 1

        if (tol_f_break and np.absolute(f - f_old) <= tol_f) \
                or (tol_X_break and np.linalg.norm(X - X_old, 'fro') <= tol_X):
            break

    if len(X_star) and compare_opt and plot_dynamic:
        plt.ioff()
        #plt.show(block=False)

    return X, norm_diff, norm_err, f_seq


def find_path(q, graph_adj):
    dist, path = shortest_path(q, graph_adj)
    neighbors = [x for x in range(len(path)) if len(path[x]) == 2]
    neighbors.sort()

    return neighbors, path


def shortest_path(q, graph_adj):
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


def constr_Nu(neighbors, path):
    nbneighbors = len(neighbors)
    Nu = []
    for k in neighbors:
        Nu.append([x for x in range(len(path)) if k in path[x]])

    return Nu


def constr_Cq(X, q, prob_params, neighbors, Nu):
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
        for n in range(len(Nu[k])):
            Nu_k = Nu[k]
            l = Nu_k[n]
            X_curr = X[np.sum(nbsensors_vec[0:l]):np.sum(nbsensors_vec[0:l + 1]), :]
            Cq[np.sum(nbsensors_vec[0:l]):np.sum(nbsensors_vec[0:l + 1]),
            nbsensors_vec[q] + ind_k * Q: nbsensors_vec[q] + ind_k * Q + Q] = X_curr

    return Cq


def compress(data, Cq):
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
    M_q = nbsensors_vec[q]
    row_blk = np.cumsum(nbsensors_vec)
    row_blk = np.append(0, row_blk[0:-1])
    row_blk_q = row_blk[q]
    Xq = X[row_blk_q:row_blk_q + M_q, :]

    return Xq


def dynamic_plot(X, X_star, line1, line2):
    line1.set_ydata(X[:, 1])
    line2.set_ydata(X_star[:, 1])
    plt.draw()
    plt.pause(0.05)
