import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem

from dasftoolbox.utils import (
    autocorrelation_matrix,
    cross_correlation_matrix,
)


class MMSEProblem(OptimizationProblem):
    def __init__(self, nb_filters: int) -> None:
        super().__init__(nb_filters=nb_filters)

    def solve(
        self,
        problem_inputs: ProblemInputs,
        save_solution: bool = False,
        convergence_parameters=None,
        initial_estimate=None,
    ) -> np.ndarray:
        """Solve the MMSE problem min E[||d(t) - X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_signals[0]
        D = problem_inputs.global_parameters[0]

        Ryy = autocorrelation_matrix(Y)
        Ryd = cross_correlation_matrix(Y, D)

        X_star = np.linalg.inv(Ryy) @ Ryd

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the MMSE objective E[||d(t) - X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_signals[0]
        D = problem_inputs.global_parameters[0]

        Ryy = autocorrelation_matrix(Y)
        Rdd = autocorrelation_matrix(D)
        Ryd = cross_correlation_matrix(Y, D)

        f = np.trace(X.T @ Ryy @ X) - 2 * np.trace(X.T @ Ryd) + np.trace(Rdd)

        return f
