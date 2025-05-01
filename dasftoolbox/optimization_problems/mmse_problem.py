import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem

from dasftoolbox.utils import (
    autocorrelation_matrix,
    cross_correlation_matrix,
)


class MMSEProblem(OptimizationProblem):
    """
    MMSE problem class.
    """
    def __init__(self, nb_filters: int) -> None:
        super().__init__(nb_filters=nb_filters)

    def solve(
        self,
        problem_inputs: ProblemInputs,
        save_solution: bool = False,
        convergence_parameters=None,
        initial_estimate=None,
    ) -> np.ndarray:
        """
        Solve the MMSE problem
        :math:`\min_X\; \mathbb{E}[\|\mathbf{d}(t) - X^T \mathbf{y}(t)\|^2]`.

        Parameters
        ----------
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signal :math:`\mathbf{y}` and the target signal :math:`\mathbf{d}`.
        save_solution : bool, optional
            Whether to save the solution or not, by default False
        convergence_parameters : None, optional
            Convergence parameters, by default None
        initial_estimate : None, optional
            Initial estimate, by default None

        Returns
        -------
        np.ndarray
            The solution to the MMSE problem.
        """
        Y = problem_inputs.fused_signals[0]
        D = problem_inputs.global_parameters[0]

        Ryy = autocorrelation_matrix(Y)
        Ryd = cross_correlation_matrix(Y, D)

        X_star = np.linalg.inv(Ryy) @ Ryd

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        r"""
        Evaluate the MMSE objective
        :math:`\mathbb{E}[\|\mathbf{d}(t) - X^T \mathbf{y}(t)\|^2]`.

        Parameters
        ----------
        X : np.ndarray
            The point to evaluate.
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signal :math:`\mathbf{y}` and the target signal :math:`\mathbf{d}`.
        
        Returns
        -------
        float
            The value of the objective function at the point X.
        """
        Y = problem_inputs.fused_signals[0]
        D = problem_inputs.global_parameters[0]

        Ryy = autocorrelation_matrix(Y)
        Rdd = autocorrelation_matrix(D)
        Ryd = cross_correlation_matrix(Y, D)

        f = np.trace(X.T @ Ryy @ X) - 2 * np.trace(X.T @ Ryd) + np.trace(Rdd)

        return f
