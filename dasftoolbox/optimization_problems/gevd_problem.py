import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem

from dasftoolbox.utils import autocorrelation_matrix

import scipy


class GEVDProblem(OptimizationProblem):
    def __init__(
        self,
        nb_filters: int,
    ) -> None:
        super().__init__(nb_filters=nb_filters)

    def solve(
        self,
        problem_inputs: ProblemInputs,
        save_solution: bool = False,
        convergence_parameters=None,
        initial_estimate=None,
    ) -> np.ndarray:
        """Solve the GEVD problem max E[||X.T @ y(t)||**2] s.t. E[X.T @ v(t) @ v(t).T @ X] = I."""
        Y = problem_inputs.fused_signals[0]
        V = problem_inputs.fused_signals[1]

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)

        eigvals, eigvecs = scipy.linalg.eigh(Ryy, Rvv)
        indices = np.argsort(eigvals)[::-1]

        X_star = eigvecs[:, indices[0 : self.nb_filters]]

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the GEVD objective E[||X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_signals[0]

        Ryy = autocorrelation_matrix(Y)

        f = np.trace(X.T @ Ryy @ X)

        return f

    def resolve_ambiguity(
        self,
        X_reference: np.ndarray | list[np.ndarray],
        X_current: np.ndarray | list[np.ndarray],
        updating_node: int | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Resolve the sign ambiguity for the GEVD problem."""

        for col in range(self.nb_filters):
            if np.linalg.norm(X_reference[:, col] - X_current[:, col]) > np.linalg.norm(
                -X_reference[:, col] - X_current[:, col]
            ):
                X_current[:, col] = -X_current[:, col]

        return X_current
