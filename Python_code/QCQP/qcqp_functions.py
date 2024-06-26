import numpy as np
from numpy import linalg as LA
import scipy.optimize as opt
import warnings


# This module implements the functions related to the QCQP problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def qcqp_solver(prob_params, data, X0, solver_params):
    """Solve the QCQP problem min 0.5 * E[||X.T @ y(t)||**2] - trace(X.T @ B) s.t. trace(X.T @ Gamma @ X) <= alpha**2; X.T @ c = d."""
    Y = data['Y_list'][0]
    B = data['B_list'][0]
    c = data['B_list'][1]
    Gamma = data['Gamma_list'][0]
    alpha = data['Glob_Const_list'][0]
    d = data['Glob_Const_list'][1]

    U_c, S_c, _ = LA.svd(Gamma)

    sqrt_Gamma = (U_c @ np.diag(np.sqrt(S_c))).T

    if alpha ** 2 == LA.norm(d) ** 2 / LA.norm(LA.inv(sqrt_Gamma).T @ c) ** 2:
        X_star = LA.inv(Gamma) @ c @ d.T / LA.norm(sqrt_Gamma.T @ c)
    elif alpha ** 2 > LA.norm(d) ** 2 / LA.norm(LA.inv(sqrt_Gamma).T @ c) ** 2:
        if norm_fun(0,data) < 0:
            X_star = X_fun(0, data)
        else:
            mu_star = opt.fsolve(norm_fun,0,data)
            X_star = X_fun(mu_star, data)
    else:
        warnings.warn("Infeasible")

    return X_star


def X_fun(mu, data):
    Y = data['Y_list'][0]
    B = data['B_list'][0]
    c = data['B_list'][1]
    Gamma = data['Gamma_list'][0]
    d = data['Glob_Const_list'][1]
    N = np.size(Y, 1)
    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2

    M = Ryy + mu * Gamma
    M_inv = LA.inv(M)
    w = (B.T @ M_inv.T @ c - d) / (c.T @ M_inv @ c)
    X = M_inv @ (B - c @ w.T)

    return X

def norm_fun(mu,data):
    Gamma = data['Gamma_list'][0]
    alpha = data['Glob_Const_list'][0]
    X = X_fun(mu, data)
    norm = np.trace(X.T @ Gamma @ X) - alpha ** 2

    return norm


def qcqp_eval(X, data):
    """Evaluate the QCQP objective 0.5 * E[||X.T @ y(t)||**2] - trace(X.T @ B)."""
    Y = data['Y_list'][0]
    B = data['B_list'][0]
    N = np.size(Y, 1)

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2

    f = 0.5 * np.trace(X.T @ Ryy @ X) - np.trace(X.T @ B)

    return f


def create_data(nbsensors, nbsamples, Q):
    """Create data for the QCQP problem."""
    rng = np.random.default_rng()

    Y = create_signal(nbsensors, nbsamples)
    Ryy = Y @ Y.T / nbsamples
    Ryy = (Ryy + Ryy.T) / 2
    B = rng.standard_normal(size=(nbsensors, Q))
    c = rng.standard_normal(size=(nbsensors, 1))
    d = rng.standard_normal(size=(Q, 1))
    w = (B.T @ LA.inv(Ryy).T @ c - d) / (c.T @ LA.inv(Ryy) @ c)
    X = LA.inv(Ryy) @ (B - c @ w.T)

    toss = rng.integers(0, 1, endpoint=True)
    if toss == 0:
        alpha = rng.standard_normal()
        alpha = alpha ** 2
    else:
        alpha = rng.standard_normal()
        alpha = alpha ** 2
        alpha = np.sqrt(LA.norm(X, ord='fro') ** 2 + alpha ** 2)

    while alpha ** 2 < LA.norm(d) ** 2 / LA.norm(c) ** 2:
        c = rng.standard_normal(size=(nbsensors, 1))
        d = rng.standard_normal(size=(Q, 1))
        w = (B.T @ LA.inv(Ryy).T @ c - d) / (c.T @ LA.inv(Ryy) @ c)
        X = LA.inv(Ryy) @ (B - c @ w.T)
        toss = rng.integers(0, 1, endpoint=True)
        if toss == 0:
            alpha = rng.standard_normal()
            alpha = alpha ** 2
        else:
            alpha = rng.standard_normal()
            alpha = alpha ** 2
            alpha = np.sqrt(LA.norm(X, ord='fro') ** 2 + alpha ** 2)

    return Y, B, alpha, c, d

def create_signal(nbsensors, nbsamples):
    """Create signals for the QCQP problem."""
    rng = np.random.default_rng()

    signalvar = 0.5
    noisepower = 0.1
    nbsources = 10
    offset = 0.5

    s = rng.normal(loc=0, scale=np.sqrt(signalvar), size=(nbsources, nbsamples))
    A = rng.uniform(low=-offset, high=offset, size=(nbsensors, nbsources))
    noise = rng.normal(loc=0, scale=np.sqrt(noisepower), size=(nbsensors, nbsamples))

    Y = A @ s + noise

    return Y