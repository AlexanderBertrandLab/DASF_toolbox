import numpy as np
from scipy import linalg as LA


# This module implements the functions related to the LCMV problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def lcmv_solver(prob_params, data):
    """Solve the LCMV problem min E[||X.T @ y(t)||**2] s.t. X.T @ B = H."""
    Y = data['Y_list'][0]
    B = data['B_list'][0]
    H = data['Glob_Const_list'][0]

    N = prob_params['nbsamples']

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2

    X_star = np.linalg.inv(Ryy) @ B @ np.linalg.inv(B.T @ np.linalg.inv(Ryy) @ B) @ H

    return X_star


def lcmv_eval(X, data):
    """Evaluate the LCMV objective E[||X.T @ y(t)||**2]."""
    Y = data['Y_list'][0]
    N = np.size(Y, 1)

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2

    f = np.trace(X.T @ Ryy @ X)

    return f


def create_data(nbsensors, nbsamples, Q):
    """Create data for the LCMV problem."""
    rng = np.random.default_rng()

    Y, A = create_signal(nbsensors, nbsamples)
    B = A[:, 0:Q]
    H = rng.standard_normal(size=(Q,Q))

    return Y, B, H


def create_signal(nbsensors, nbsamples):
    """Create signals for the LCMV problem."""
    rng = np.random.default_rng()

    noisepower = 0.1
    signalvar = 0.5
    nbsources = 10
    offset = 0.5

    s = rng.normal(loc=0, scale=np.sqrt(signalvar), size=(nbsources, nbsamples))
    A = rng.uniform(low=-offset, high=offset, size=(nbsensors, nbsources))
    noise = rng.normal(loc=0, scale=np.sqrt(noisepower), size=(nbsensors, nbsamples))

    Y = A @ s + noise

    return Y, A
