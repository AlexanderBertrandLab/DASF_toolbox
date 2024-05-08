import numpy as np
from scipy import linalg as LA


# This module implements the functions related to the TRO problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def tro_solver(prob_params, data, X0, solver_params):
    """Solve the TRO problem max E[||X.T @ y(t)||**2] / E[||X.T @ v(t)||**2] s.t. X.T @ Gamma @ X = I."""
    Y = data['Y_list'][0]
    V = data['Y_list'][1]
    Gamma = data['Gamma_list'][0]

    rng = np.random.default_rng()
    i = 0
    M = np.size(Y, 0)
    Q = prob_params['Q']
    if X0 is None:
        X = rng.normal(size=(M, Q))
    else:
        X = X0
    f = tro_eval(X, data)

    if solver_params and ("tol_f" in solver_params):
        tol_f = solver_params['tol_f']
    else:
        tol_f = 1e-8

    if solver_params and ("tol_X" in solver_params):
        tol_X = solver_params['tol_X']
    else:
        tol_X = 1e-8

    if solver_params and ("maxiter" in solver_params):
        maxiter = solver_params['maxiter']
    else:
        maxiter = 300

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

    while i < maxiter:
        eigvals, eigvecs = LA.eig(Kyy - f * Kvv)
        indices = np.argsort(eigvals)[::-1]

        X_old = X
        X = eigvecs[:, indices[0:Q]]
        f_old = f
        Y_list_t = [Y_t, V_t]
        data_t = {'Y_list': Y_list_t}
        f = tro_eval(X, data_t)

        i += 1

        if (np.absolute(f - f_old) <= tol_f) \
                or (np.linalg.norm(X - X_old, 'fro') <= tol_X):
            break

    X_star = U_c @ np.diag(np.sqrt(1 / S_c)) @ X

    return X_star


def tro_aux_solver(prob_params, data, X0, solver_params):
    """Given X0, solve the auxiliary problem of TRO: max E[||X.T @ y(t)||**2] - rho(X0) * E[||X.T @ v(t)||**2] s.t. X.T @ Gamma @ X = I, where rho(X) == E[||X.T @ y(t)||**2] / E[||X.T @ v(t)||**2]"""
    Y = data['Y_list'][0]
    V = data['Y_list'][1]
    Gamma = data['Gamma_list'][0]

    Q = prob_params['Q']

    N = prob_params['nbsamples']

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    Rvv = V @ V.T / N
    Rvv = (Rvv + Rvv.T) / 2
    rho = tro_eval(X0, data)

    U_c, S_c, _ = LA.svd(Gamma)

    Kyy = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Ryy @ U_c @ np.diag(np.sqrt(1 / S_c))
    Kvv = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Rvv @ U_c @ np.diag(np.sqrt(1 / S_c))
    
    eigvals, eigvecs = LA.eig(Kyy - rho * Kvv)
    indices = np.argsort(eigvals)[::-1]

    X = eigvecs[:, indices[0:Q]]

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
