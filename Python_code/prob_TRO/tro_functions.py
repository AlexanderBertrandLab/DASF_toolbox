import numpy as np
from scipy import linalg as LA


# Solve the TRO problem max E[||X'*y(t)||^2]/E[||X'*v(t)||^2] s.t. X'*Gamma*X=I.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def tro_solver(prob_params, data):
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
        l_int, E_int = LA.eig(Kyy - f * Kvv)
        ind_int = np.argsort(l_int)[::-1]

        X = E_int[:, ind_int[0:Q]]
        f_old = f
        Y_list_t = [Y_t, V_t]
        data_t = {'Y_list': Y_list_t}
        f = tro_eval(X, data_t)

        i = i + 1

    X_star = U_c @ np.diag(np.sqrt(1 / S_c)) @ X

    return X_star


# Evaluate the TRO objective E[||X'*y(t)||^2]/E[||X'*v(t)||^2].
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def tro_eval(X, data):
    Y = data['Y_list'][0]
    V = data['Y_list'][1]
    N = np.size(Y, 1)

    Ryy = Y @ Y.T / N
    Ryy = (Ryy + Ryy.T) / 2
    Rvv = V @ V.T / N
    Rvv = (Rvv + Rvv.T) / 2

    f = np.trace(X.T @ Ryy @ X) / np.trace(X.T @ Rvv @ X)

    return f


# Resolve the sign ambiguity for the TRO problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def tro_select_sol(Xq_old, Xq, X):
    Q = np.size(Xq_old, 1)

    for l in range(Q):
        if np.sum((Xq_old[:, l] - Xq[:, l]) ** 2) > np.sum((-Xq_old[:, l] - Xq[:, l]) ** 2):
            X[:, l] = -X[:, l]

    return X


# Create data for the TRO problem.
#
# Author: Cem Musluoglu, KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems,
# Signal Processing and Data Analytics
# Correspondence: cemates.musluoglu@esat.kuleuven.be

def create_data(nbsensors, nbsamples):
    rng = np.random.default_rng()

    nbsources = 5
    latent_dim = 10

    d = rng.standard_normal(size=(nbsamples, nbsources))
    s = rng.standard_normal(size=(nbsamples, latent_dim - nbsources))
    A = rng.uniform(low=-0.5, high=0.5, size=(nbsources, nbsensors))
    B = rng.uniform(low=-0.5, high=0.5, size=(latent_dim - nbsources, nbsensors))
    noise = rng.standard_normal(size=(nbsamples, nbsensors))

    V = s @ B + noise
    Y = d @ A + V

    Y = Y.T
    V = V.T

    return Y, V
