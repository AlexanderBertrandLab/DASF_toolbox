import numpy as np


def make_symmetric(matrix: np.ndarray) -> np.ndarray:
    """
    Makes a matrix symmetric by averaging it with its transpose.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to make symmetric.

    Returns
    -------
    np.ndarray
        The symmetric matrix."""
    return (matrix + matrix.T) / 2


def autocorrelation_matrix(
    data: np.ndarray, normalizer: int | None = None
) -> np.ndarray:
    """
    Computes the autocorrelation matrix of the data.

    Parameters
    ----------
    data : np.ndarray
        The data to compute the autocorrelation matrix for.

    Returns
    -------
    np.ndarray
        The autocorrelation matrix.
    """
    if normalizer is None:
        normalizer = np.size(data, 1)
    matrix = data @ data.T / normalizer
    return make_symmetric(matrix)


def cross_correlation_matrix(
    data1: np.ndarray, data2: np.ndarray, normalizer: int | None = None
) -> np.ndarray:
    """
    Computes the cross-correlation matrix of the data.

    Parameters
    ----------
    data1 : np.ndarray
        The first data to compute the cross-correlation matrix for.
    data2 : np.ndarray
        The second data to compute the cross-correlation matrix for.

    Returns
    -------
    np.ndarray
        The cross-correlation matrix.
    """
    if normalizer is None:
        normalizer = np.size(data1, 1)
    return data1 @ data2.T / normalizer


def covariance_matrix(data: np.ndarray, normalizer: int | None = None) -> np.ndarray:
    """
    Computes the covariance matrix of the data.

    Parameters
    ---------
    data : np.ndarray
        The data to compute the covariance matrix for.

    Returns
    -------
    np.ndarray
        The covariance matrix.
    """
    if normalizer is None:
        normalizer = np.size(data, 1)
    matrix = (
        data @ data.T / normalizer - np.mean(data, axis=1) @ np.mean(data, axis=1).T
    )
    return make_symmetric(matrix)


def cross_covariance_matrix(
    data1: np.ndarray, data2: np.ndarray, normalizer: int | None = None
) -> np.ndarray:
    """
    Computes the cross-covariance matrix of the data.

    Parameters
    ----------
    data1 : np.ndarray
        The first data to compute the cross-covariance matrix for.
    data2 : np.ndarray
        The second data to compute the cross-covariance matrix for.

    Returns
    -------
    np.ndarray
        The cross-covariance matrix.
    """
    if normalizer is None:
        normalizer = np.size(data1, 1)
    return (
        data1 @ data2.T / normalizer - np.mean(data1, axis=1) @ np.mean(data2, axis=1).T
    )


def normalize(data: np.ndarray, scale: float = 1):
    """
    Normalize the data to be zero-mean and of specified variance.

    Parameters
    ----------
    data : np.ndarray
        The data to normalize.
    scale : float
        The scale to normalize the data to. Default is 1.

    Returns
    -------
    np.ndarray
        The normalized data.
    """
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_var = np.sqrt(np.expand_dims(np.var(data, axis=1), axis=1))
    data = data - data_mean
    data = scale * data / data_var
    return data
