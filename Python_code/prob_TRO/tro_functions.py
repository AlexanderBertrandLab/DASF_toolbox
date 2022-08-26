import numpy as np
from scipy import linalg as LA


# This module implements the functions related to the TRO problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def tro_solver(prob_params, data):
    """Solve the TRO problem max E[||X.T @ y(t)||**2] / E[||X.T @ v(t)||**2] s.t. X.T @ Gamma @ X = I."""
    Y = data['Y_list'][0]
    V = data['Y_list'][1]
    Gamma = data['Gamma_list'][0]

    rng = np.random.default_rng()
    i = 0
    M = np.size(Y, 0)
    Q = prob_params['Q']
    X = rng.standard_normal(size=(M, Q))
    f = tro_eval(X, data)
    tol_f = 1e-6

    N = prob_params['nbsamples']

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    Rvv = V @ V.T / N
    Rvv = (Rvv + Rvv.T) / 2

    U_c, S_c, _ = LA.svd(Gamma)

    Y_t = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Y
    V_t = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ V

    Kyy = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Ryy @ U_c @ np.diag(np.sqrt(1 / S_c))
    Kvv = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Rvv @ U_c @ np.diag(np.sqrt(1 / S_c))

    while (i == 0) or (np.abs(f - f_old) > tol_f):
        eigvals, eigvecs = LA.eig(Kyy - f * Kvv)
        indices = np.argsort(eigvals)[::-1]

        X = eigvecs[:, indices[0:Q]]
        f_old = f
        Y_list_t = [Y_t, V_t]
        data_t = {'Y_list': Y_list_t}
        f = tro_eval(X, data_t)

        i = i + 1

    X_star = U_c @ np.diag(np.sqrt(1 / S_c)) @ X

    return X_star


def tro_eval(X, data):
    """Evaluate the TRO objective E[||X.T @ y(t)||**2] / E[||X.T @ v(t)||**2]."""
    Y = data['Y_list'][0]
    V = data['Y_list'][1]
    N = np.size(Y, 1)

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    Rvv = V @ V.T / N
    Rvv = (Rvv + Rvv.T) / 2

    f = np.trace(X.T @ Ryy @ X) / np.trace(X.T @ Rvv @ X)

    return f


def tro_select_sol(X_ref, X, prob_params, q):
    """Resolve the sign ambiguity for the TRO problem."""
    Q = prob_params['Q']

    for l in range(Q):
        if np.linalg.norm(X_ref[:, l] - X[:, l]) > np.linalg.norm(-X_ref[:, l] - X[:, l]):
            X[:, l] = -X[:, l]

    return X


def create_data(nbsensors, nbsamples):
    """Create data for the TRO problem."""
    rng = np.random.default_rng()

    noisepower = 0.1
    signalvar = 0.5
    nbsources = 5
    latent_dim = 10
    offset = 0.5

    d = rng.normal(loc=0, scale=np.sqrt(signalvar), size=(nbsources, nbsamples))
    s = rng.normal(loc=0, scale=np.sqrt(signalvar), size=(latent_dim - nbsources, nbsamples))
    A = rng.uniform(low=-offset, high=offset, size=(nbsensors, nbsources))
    B = rng.uniform(low=offset, high=offset, size=(nbsensors, latent_dim - nbsources))
    noise = rng.normal(loc=0, scale=np.sqrt(noisepower), size=(nbsensors, nbsamples))

    V = B @ s + noise
    Y = A @ d + V

    return Y, V
