import numpy as np
from numpy import linalg as LA
import pymanopt
from pymanopt.manifolds import Sphere
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
import autograd


# This module implements the functions related to the SCQP problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def scqp_solver(prob_params, data, X0, solver_params):
    """Solve the SCQP problem min 0.5 * E[||X.T @ y(t)||**2] + trace(X.T @ B) s.t. trace(X.T @ Gamma @ X)=1."""
    Y = data['Y_list'][0]
    B = data['B_list'][0]
    Gamma = data['Gamma_list'][0]
    nbsamples = prob_params['nbsamples']

    manifold = Sphere(np.size(B, 0), np.size(B, 1))

    Ryy = Y @ Y.T / nbsamples
    Ryy = (Ryy + Ryy.T) / 2

    Gamma = (Gamma + Gamma.T) / 2

    L = LA.cholesky(Gamma)
    Ryy_t = LA.inv(L) @ Ryy @ LA.inv(L).T
    Ryy_t = (Ryy_t + Ryy_t.T) / 2
    B_t = LA.inv(L) @ B

    @pymanopt.function.autograd(manifold)
    def cost(X):
        return 0.5 * autograd.numpy.trace(X.T @ Ryy_t @ X) + autograd.numpy.trace(X.T @ B_t)

    problem = Problem(manifold=manifold, cost=cost)
    problem.verbosity = 0

    solver = TrustRegions(verbosity=0)
    X_star = solver.run(problem).point
    X_star = LA.inv(L.T) @ X_star

    return X_star


def scqp_eval(X, data):
    """Evaluate the SCQP objective 0.5 * E[||X.T @ y(t)||**2] + trace(X.T @ B)."""
    Y = data['Y_list'][0]
    B = data['B_list'][0]
    N = np.size(Y, 1)

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2

    f = 0.5 * np.trace(X.T @ Ryy @ X) + np.trace(X.T @ B)

    return f


def create_data(nbsensors, nbsamples, Q):
    """Create data for the SCQP problem."""
    rng = np.random.default_rng()

    signalvar = 0.5
    noisepower = 0.1
    nbsources = 10
    offset = 0.5

    s = rng.normal(loc=0, scale=np.sqrt(signalvar), size=(nbsources, nbsamples))
    A = rng.uniform(low=-offset, high=offset, size=(nbsensors, nbsources))
    noise = rng.normal(loc=0, scale=np.sqrt(noisepower), size=(nbsensors, nbsamples))

    Y = A @ s + noise
    B = rng.standard_normal(size=(nbsensors, Q))

    return Y, B
