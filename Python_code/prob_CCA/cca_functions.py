import numpy as np
from scipy import linalg as LA


# This module implements the functions related to the CCA problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def cca_solver(prob_params, data):
    """Solve the CCA problem max_(X,W) E[trace(X.T @ y(t) @ v(t).T @ W)]
    s.t. E[X.T @ y(t) @ y(t).T @ X] = I, E[W.T @ v(t) @ v(t).T @ W] = I."""
    data_X = data[0]
    data_W = data[1]

    Y = data_X['Y_list'][0]
    V = data_W['Y_list'][0]

    Q = prob_params['Q']
    N = prob_params['nbsamples']

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    Rvv = V @ V.T / N
    Rvv = (Rvv + Rvv.T) / 2
    Ryv = Y @ V.T / N
    Rvy = Ryv.T

    inv_Rvv = np.linalg.inv(Rvv)
    inv_Rvv = (inv_Rvv + inv_Rvv.T) / 2
    A_X = Ryv @ inv_Rvv @ Rvy
    A_X = (A_X + A_X.T) / 2

    eigvals_X, eigvecs_X = LA.eigh(A_X, Ryy)
    indices_X = np.argsort(eigvals_X)[::-1]
    eigvals_X = eigvals_X[indices_X]
    eigvecs_X = eigvecs_X[:, indices_X]

    X = eigvecs_X[:, 0:Q]
    eigvecs_W = inv_Rvv @ Rvy @ eigvecs_X @ np.diag(1/np.sqrt(np.absolute(eigvals_X)))
    W = eigvecs_W[:, 0:Q]
    X_star = [X, W]

    return X_star


def cca_eval(X_list, data):
    """Evaluate the CCA objective E[trace(X.T @ y(t) @ v(t).T @ W)]."""
    data_X = data[0]
    data_W = data[1]

    Y = data_X['Y_list'][0]
    V = data_W['Y_list'][0]
    N = np.size(Y, 1)

    Ryv = Y @ V.T / N
    X = X_list[0]
    W = X_list[1]

    f = np.trace(X.T @ Ryv @ W)

    return f


def cca_select_sol(X_ref_multi, X_multi, prob_params, q):
    """Resolve the sign ambiguity for the CCA problem."""
    X = X_multi[0]
    W = X_multi[1]
    X_ref = X_ref_multi[0]
    Q = prob_params['Q']

    for l in range(Q):
        if np.linalg.norm(X_ref[:, l] - X[:, l]) > np.linalg.norm(-X_ref[:, l] - X[:, l]):
            X[:, l] = -X[:, l]
            W[:, l] = -W[:, l]

    X_multi[0] = X
    X_multi[1] = W

    return X_multi


def create_data(nbsensors, nbsamples):
    """Create data for the CCA problem."""
    rng = np.random.default_rng()

    noisepower = 0.1
    signalvar = 0.5
    nbsources = 10
    offset = 0.5
    lags = 3

    d = rng.normal(loc=0, scale=np.sqrt(signalvar), size=(nbsources, nbsamples + lags))
    A = rng.uniform(low=-offset, high=offset, size=(nbsensors, nbsources))
    noise = rng.normal(loc=0, scale=np.sqrt(noisepower), size=(nbsensors, nbsamples + lags))
    signal = A @ d + noise
    Y = signal[:, 0:nbsamples]
    V = signal[:, lags:None]

    return Y, V
