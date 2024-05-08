import numpy as np
from scipy import linalg as LA


# This module implements the functions related to the GEVD problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def gevd_solver(prob_params, data, X0, solver_params):
    """Solve the GEVD problem max E[||X.T @ y(t)||**2] s.t. E[X.T @ v(t) @ v(t).T @ X] = I."""
    Y = data['Y_list'][0]
    V = data['Y_list'][1]

    Q = prob_params['Q']
    N = prob_params['nbsamples']

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    Rvv = V @ V.T / N
    Rvv = (Rvv + Rvv.T) / 2

    eigvals, eigvecs = LA.eigh(Ryy, Rvv)
    indices = np.argsort(eigvals)[::-1]

    X_star = eigvecs[:, indices[0:Q]]

    return X_star


def gevd_eval(X, data):
    """Evaluate the GEVD objective E[||X.T @ y(t)||**2]."""
    Y = data['Y_list'][0]
    N = np.size(Y, 1)

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2

    f = np.trace(X.T @ Ryy @ X)

    return f


def gevd_select_sol(X_ref, X, prob_params, q):
    """Resolve the sign ambiguity for the GEVD problem."""
    Q = prob_params['Q']

    for l in range(Q):
        if np.linalg.norm(X_ref[:, l] - X[:, l]) > np.linalg.norm(-X_ref[:, l] - X[:, l]):
            X[:, l] = -X[:, l]

    return X


def create_data(nbsensors, nbsamples):
    """Create data for the GEVD problem."""
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
