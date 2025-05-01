import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem
from dasftoolbox.problem_settings import ConvergenceParameters

from dasftoolbox.utils import autocorrelation_matrix

from typing import Literal

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICAProblem(OptimizationProblem):
    """
    ICA problem class.

    Attributes
    ----------
    nb_filters : int
        Number of filters.
    convergence_parameters : ConvergenceParameters
        Convergence parameters.
    negentropy : Literal["logcosh", "exponential", "kurtosis"]
        Negentropy function to use, by default "logcosh".
    alpha: int
        Parameter for the negentropy function, by default 1.
    """

    def __init__(
        self,
        nb_filters: int,
        convergence_parameters: ConvergenceParameters,
        negentropy: Literal["logcosh", "exponential", "kurtosis"] = "logcosh",
        alpha: int = 1,
    ) -> None:
        super().__init__(
            nb_filters=nb_filters, convergence_parameters=convergence_parameters
        )
        self.alpha = alpha
        if negentropy == "logcosh":
            self.negentropy = self.LogCoshNegentropy(alpha)
        elif negentropy == "exponential":
            self.negentropy = self.ExponentialNegentropy(alpha)
        elif negentropy == "kurtosis":
            self.negentropy = self.KurtosisNegentropy(alpha)
        else:
            raise ValueError

    class NegentropyFunction:
        """
        Base class for negentropy functions.
        """

        def evaluate(self, y: np.ndarray) -> np.ndarray:
            """
            Evaluate the negentropy function.
            """
            raise NotImplementedError

        def gradient(self, y: np.ndarray) -> np.ndarray:
            """
            Calculate the gradient.
            """
            raise NotImplementedError

        def hessian(self, y: np.ndarray) -> np.ndarray:
            """
            Calculate the Hessian.
            """
            raise NotImplementedError

    class LogCoshNegentropy(NegentropyFunction):
        """
        Log cosh negentropy: :math:`\\frac{1}{\\alpha} \\log(\\cosh(\\alpha y))`.

        Attributes
        ----------
        alpha : float
            Parameter for the negentropy function, by default 1.
        """

        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def evaluate(self, y: np.ndarray) -> np.ndarray:
            """
            Evaluate the log cosh negentropy function per channel.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The negentropy value per channel, averaged over all samples.
            """
            return (np.log(np.cosh(self.alpha * y))).mean(axis=1) / self.alpha

        def gradient(self, y: np.ndarray) -> np.ndarray:
            """
            Calculate the first derivative of the log cosh negentropy per channel: :math:`\\tanh(\\alpha y)`.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The vector of first derivatives for each signal sample.
            """
            return np.tanh(self.alpha * y)

        def hessian(self, y: np.ndarray) -> np.ndarray:
            """
            Calculate the second derivative of the log cosh negentropy per channel: :math:`\\alpha (1-\\tanh(\\alpha y)^2)`.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The vector of second derivatives for each signal sample.
            """
            return self.alpha * (1 - np.tanh(self.alpha * y) ** 2)

    class ExponentialNegentropy(NegentropyFunction):
        """Exponential negentropy: :math:`-\\exp(-\\alpha y^2)`.

        Attributes
        ----------
        alpha : float
            Parameter for the negentropy function, by default 1.
        """

        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def evaluate(self, y: np.ndarray) -> np.ndarray:
            """
            Evaluate the exponential negentropy function per channel.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The negentropy value per channel, averaged over all samples.
            """
            return -(np.exp(-self.alpha * y**2)).mean(axis=1)

        def gradient(self, y: np.ndarray) -> np.ndarray:
            """
            Calculate the first derivative of the exponential negentropy per channel: :math:`2\\alpha y \\exp(-\\alpha y^2)`.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The vector of first derivatives for each signal sample.
            """
            return 2 * self.alpha * y * np.exp(-self.alpha * y**2)

        def hessian(self, y: np.ndarray) -> np.ndarray:
            """
            Calculate the second derivative of the exponential negentropy per channel: :math:`\\alpha (1-\\tanh(\\alpha y)^2)`.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The vector of second derivatives for each signal sample.
            """
            return (2 * self.alpha - 4 * self.alpha**2 * y**2) * np.exp(
                -self.alpha * y**2
            )

    class KurtosisNegentropy(NegentropyFunction):
        """Kurtosis negentropy :math:`-\\mathbb{E}[y^4]-\\alpha \\mathbb{E}[y^2]^2`.

        Attributes
        ----------
        alpha : float
            Parameter for the negentropy function, by default 1.
        """

        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def evaluate(self, y: np.ndarray) -> np.ndarray:
            """
            Evaluate the kurtosis negentropy function per channel.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The negentropy value per channel, averaged over all samples.
            """
            return (y**4).mean(axis=1) - self.alpha * ((y**2).mean(axis=1) ** 2)

        def gradient(self, y: np.ndarray) -> np.ndarray:
            """
            Calculate the first derivative of the kurtosis negentropy per channel: :math:`4 y^3 -2\\alpha y`.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The vector of first derivatives for each signal sample.
            """
            return 4 * y**3 - 2 * self.alpha * y

        def hessian(self, y: np.ndarray) -> np.ndarray:
            """
            Calculate the second derivative of the kurtosis negentropy per channel: :math:`12y^2-2\\alpha`.

            Parameters
            ----------
            y : np.ndarray
                The input signal.

            Returns
            -------
            np.ndarray
                The vector of second derivatives for each signal sample.
            """
            return 12 * y**2 - 2 * self.alpha

    def solve(
        self,
        problem_inputs: ProblemInputs,
        save_solution: bool = False,
        convergence_parameters=None,
        initial_estimate=None,
    ) -> np.ndarray:
        """
        Solve the ICA problem :math:`\max_X \sum_m \mathbb{E}[F(X_m^T \mathbf{y}(t))]` subject to :math:`\mathbb{E}[X^T \mathbf{y}(t)\mathbf{y}^T(t) X] = I`, where :math:`X_m` is the :math:`m`-th column of :math:`X` and :math:`F` is the negentropy function.

        Parameters
        ----------
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signal :math:`\mathbf{y}` and the matrices :math:`B` and  :math:`H`.
        save_solution : bool, optional
            Whether to save the solution or not, by default False
        convergence_parameters : None, optional
            Convergence parameters, by default None
        initial_estimate : None, optional
            Initial estimate, by default None

        Returns
        -------
        np.ndarray
            The solution to the ICA problem.
        """
        Y = problem_inputs.fused_signals[0]

        if initial_estimate is None:
            rng = self.rng if self.rng is not None else np.random.default_rng()
            X = rng.normal(
                size=(
                    np.size(Y, 0),
                    self.nb_filters,
                )
            )
            logger.warning("Initializing first estimate randomly")
        else:
            X = initial_estimate

        if convergence_parameters is None:
            if self.convergence_parameters is not None:
                convergence_parameters = self.convergence_parameters
            else:
                convergence_parameters = ConvergenceParameters()

        Ryy = autocorrelation_matrix(Y)

        eigvals, eigvecs = np.linalg.eig(Ryy)

        eigvals_t = np.diag(1 / np.sqrt(eigvals))

        Y_t = eigvecs @ eigvals_t @ eigvecs.T @ Y

        for col in range(self.nb_filters):
            x = X[:, col].copy()
            x = np.expand_dims(x, axis=1)
            x = x / np.linalg.norm(x)

            i = 0
            while i < convergence_parameters.max_iterations:
                filtered_signal = x.T @ Y_t

                d_negentropy = self.negentropy.gradient(filtered_signal)
                d2_negentropy = self.negentropy.hessian(filtered_signal)
                x_new = (Y_t * d_negentropy).mean(
                    axis=1
                ) - d2_negentropy.mean() * x.squeeze()
                x_new = np.expand_dims(x_new, axis=1)
                x_new = x_new - X[:, :col] @ X[:, :col].T @ x_new
                x_new = x_new / np.linalg.norm(x_new)

                x = x_new

                i += 1

                if (convergence_parameters.argument_tolerance is not None) and (
                    np.linalg.norm(x - x_new, "fro")
                    <= convergence_parameters.argument_tolerance
                ):
                    break

            X[:, col] = x.squeeze()

        X_star = eigvecs @ eigvals_t @ eigvecs.T @ X

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """
        Evaluate the ICA objective :math:`\\sum_m \\mathbb{E}[F(X_m^T \\mathbf{y}(t))]`, where :math:`X_m` is the :math:`m`-th column of :math:`X` and :math:`F` is the negentropy function.
        """
        Y = problem_inputs.fused_signals[0]
        f = self.negentropy.evaluate(X.T @ Y).sum()
        return f

    def resolve_ambiguity(
        self,
        X_reference: np.ndarray | list[np.ndarray],
        X_current: np.ndarray | list[np.ndarray],
        updating_node: int | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Resolve the sign ambiguity for the ICA problem by selecting the sign for each column of the current point so as to minimize the distance to the reference point.

        Parameters
        ----------
        X_reference : np.ndarray | list[np.ndarray]
            The reference point.
        X_current : np.ndarray | list[np.ndarray]
            The current point.
        updating_node : int | None
            The index of the updating node (for more flexibility), by default None.

        Returns
        -------
        np.ndarray | list[np.ndarray]
            A fixed solution of the ICA problem.
        """

        for col in range(self.nb_filters):
            if np.linalg.norm(X_reference[:, col] - X_current[:, col]) > np.linalg.norm(
                -X_reference[:, col] - X_current[:, col]
            ):
                X_current[:, col] = -X_current[:, col]

        return X_current
