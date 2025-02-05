import numpy as np


def make_symmetric(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2


def autocorrelation_matrix(data: np.ndarray) -> np.ndarray:
    matrix = data @ data.T / np.size(data, 1)
    return make_symmetric(matrix)


def cross_correlation_matrix(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    return data1 @ data2.T / np.size(data1, 1)


def covariance_matrix(data: np.ndarray) -> np.ndarray:
    matrix = (
        data @ data.T / np.size(data, 1)
        - np.mean(data, axis=1) @ np.mean(data, axis=1).T
    )
    return make_symmetric(matrix)


def cross_covariance_matrix(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    return (
        data1 @ data2.T / np.size(data1, 1)
        - np.mean(data1, axis=1) @ np.mean(data2, axis=1).T
    )
