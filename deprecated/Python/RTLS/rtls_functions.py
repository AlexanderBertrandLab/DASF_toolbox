import numpy as np
from scipy import linalg as LA
import scipy.optimize as opt


# This module implements the functions related to the RTLS problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be


def rtls_solver(prob_params, data, X0, solver_params):
    """Solve the RTLS problem max rho = E[|X.T @ y(t) - d(t)|**2] / (1 + X.T @ X) s.t. ||X.T @ L|| <= delta ** 2."""
    Y = data['Y_list'][0]
    L = data['B_list'][0]
    Gamma = data['Gamma_list'][0]
    d = data['Glob_Const_list'][0]
    delta = data['Glob_Const_list'][1]

    rng = np.random.default_rng()
    i = 0
    M = np.size(Y, 0)
    Q = prob_params['Q']
    X = rng.standard_normal(size=(M, Q))
    if X0 is None:
        X = rng.normal(size=(M, Q))
    else:
        X = X0
    f = rtls_eval(X, data)

    if solver_params and ("tol_f" in solver_params):
        tol_f = solver_params['tol_f']
    else:
        tol_f = 1e-8

    if solver_params and ("tol_X" in solver_params):
        tol_X = solver_params['tol_X']
    else:
        tol_X = 1e-10

    if solver_params and ("maxiter" in solver_params):
        maxiter = solver_params['maxiter']
    else:
        maxiter = 300

    N = prob_params['nbsamples']

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    ryd = np.sum(Y * d, axis=1) / N
    rdd = np.sum(d * d, axis=1) / N
    LL = L @ L.T
    LL = (LL + LL.T) / 2
    Gamma = (Gamma + Gamma.T) / 2

    
    while i < maxiter:
    
        X_old = X
        X_f = np.linalg.inv(Ryy - f * Gamma) @ ryd
        if X_f.T @ LL @ X_f < delta ** 2:
            X = X_f
        else:
            obj = lambda l: np.linalg.norm(L.T @ np.linalg.inv(Ryy - f * Gamma + l * LL) @ ryd) ** 2 - delta ** 2
            opt_sol = opt.root_scalar(obj, bracket=[0,1000], method='bisect', x0=0)
            l_star = opt_sol.root

            X = np.linalg.inv(Ryy - f * Gamma + l_star * LL) @ ryd

        f_old = f
        f = rtls_eval(X, data)

        if (np.linalg.norm(X - X_old) <= tol_X):
            break

        i += 1

    return np.expand_dims(X, axis=1)


def rtls_aux_solver(prob_params, data, X0, solver_params):
    """Solve the RTLS problem max rho = E[|X.T @ y(t) - d(t)|**2] / (1 + X.T @ X) s.t. ||X.T @ L|| <= delta ** 2."""
    Y = data['Y_list'][0]
    L = data['B_list'][0]
    Gamma = data['Gamma_list'][0]
    d = data['Glob_Const_list'][0]
    delta = data['Glob_Const_list'][1]

    rng = np.random.default_rng()
    i = 0
    M = np.size(Y, 0)
    Q = prob_params['Q']
    X = rng.standard_normal(size=(M, Q))
    if X0 is None:
        X = rng.normal(size=(M, Q))
    else:
        X = X0
    f = rtls_eval(X, data)

    if solver_params and ("tol_f" in solver_params):
        tol_f = solver_params['tol_f']
    else:
        tol_f = 1e-8

    if solver_params and ("tol_X" in solver_params):
        tol_X = solver_params['tol_X']
    else:
        tol_X = 1e-10

    if solver_params and ("maxiter" in solver_params):
        maxiter = solver_params['maxiter']
    else:
        maxiter = 300

    N = prob_params['nbsamples']

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    ryd = np.sum(Y * d, axis=1) / N
    rdd = np.sum(d * d, axis=1) / N
    LL = L @ L.T
    LL = (LL + LL.T) / 2
    Gamma = (Gamma + Gamma.T) / 2

    X_f = np.linalg.inv(Ryy - f * Gamma) @ ryd
    if X_f.T @ LL @ X_f < delta ** 2:
        X = X_f
    else:
        obj = lambda l: np.linalg.norm(L.T @ np.linalg.inv(Ryy - f * Gamma + l * LL) @ ryd) ** 2 - delta ** 2
        opt_sol = opt.root_scalar(obj, bracket=[0,1000], method='bisect', x0=0)
        l_star = opt_sol.root

        X = np.linalg.inv(Ryy - f * Gamma + l_star * LL) @ ryd

    f = rtls_eval(X, data)


    return np.expand_dims(X, axis=1)



def rtls_eval(X, data):
    """Evaluate the RTLS objective E[|X.T @ y(t) - d(t)|**2] / (1 + X.T @ Gamma @ X)."""
    Y = data['Y_list'][0]
    Gamma = data['Gamma_list'][0]
    d = data['Glob_Const_list'][0]
    N = np.size(Y, 1)

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    ryd = np.sum(Y * d, axis=1) / N
    rdd = np.sum(d * d, axis=1) / N

    f = (X.T @ Ryy @ X - 2 * X.T @ ryd + rdd) / (X.T @ Gamma @ X + 1)

    return f


def create_data(nbsensors, nbsamples, Q):
    """Create data for the RTLS problem."""
    rng = np.random.default_rng()

    noisevar = 0.2
    signalvar = 0.5
    nbsources = Q
    mixturevar = 0.3

    s = rng.normal(loc=0, scale=np.sqrt(signalvar), size=(nbsources, nbsamples))
    s = s - s.mean(axis=1, keepdims=True)
    s = s * np.sqrt(signalvar * np.ones((nbsources, 1)) / s.var(axis=1, keepdims=True))
    Pi_s = rng.normal(loc=0, scale=np.sqrt(mixturevar), size=(nbsensors, nbsources))
    noise = rng.normal(loc=0, scale=np.sqrt(noisevar), size=(nbsensors, nbsamples))
    noise = noise - noise.mean(axis=1, keepdims=True)
    noise = noise * np.sqrt(noisevar * np.ones((nbsensors, 1)) / noise.var(axis=1, keepdims=True))

    Y = Pi_s @ s + noise

    d_noisevar = 0.02
    d_noise = rng.normal(loc=0, scale=np.sqrt(d_noisevar), size=(nbsources, nbsamples))
    d_noise = d_noise - d_noise.mean(axis=1, keepdims=True)
    d_noise = d_noise * np.sqrt(d_noisevar * np.ones((nbsources, 1)) / d_noise.var(axis=1, keepdims=True))

    d = s + d_noise

    return Y, d
