import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem

from dasftoolbox.utils import (
    autocorrelation_matrix,
    cross_correlation_matrix,
    make_symmetric,
)

import scipy


class CCAProblem(OptimizationProblem):
    def __init__(
        self,
        nb_filters: int,
    ) -> None:
        super().__init__(nb_filters=nb_filters, nb_variables=2)

    def solve(
        self,
        problem_inputs: list[ProblemInputs],
        save_solution: bool = False,
        convergence_parameters=None,
        initial_estimate=None,
    ) -> list[np.ndarray]:
        """Solve the CCA problem max_(X,W) E[trace(X.T @ y(t) @ v(t).T @ W)]
        s.t. E[X.T @ y(t) @ y(t).T @ X] = I, E[W.T @ v(t) @ v(t).T @ W] = I."""
        inputs_X = problem_inputs[0]
        inputs_W = problem_inputs[1]

        Y = inputs_X.fused_signals[0]
        V = inputs_W.fused_signals[0]

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)
        Ryv = cross_correlation_matrix(Y, V)
        Rvy = Ryv.T

        inv_Rvv = np.linalg.inv(Rvv)
        inv_Rvv = make_symmetric(inv_Rvv)
        A_X = Ryv @ inv_Rvv @ Rvy
        A_X = make_symmetric(A_X)

        eigvals_X, eigvecs_X = scipy.linalg.eigh(A_X, Ryy)
        indices_X = np.argsort(eigvals_X)[::-1]
        eigvals_X = eigvals_X[indices_X]
        eigvecs_X = eigvecs_X[:, indices_X]

        X = eigvecs_X[:, 0 : self.nb_filters]
        eigvecs_W = (
            inv_Rvv @ Rvy @ eigvecs_X @ np.diag(1 / np.sqrt(np.absolute(eigvals_X)))
        )
        W = eigvecs_W[:, 0 : self.nb_filters]
        X_star = [X, W]

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(
        self, X: list[np.ndarray], problem_inputs: list[ProblemInputs]
    ) -> float:
        """Evaluate the CCA objective E[trace(X.T @ y(t) @ v(t).T @ W)]."""
        inputs_X = problem_inputs[0]
        inputs_W = problem_inputs[1]

        Y = inputs_X.fused_signals[0]
        V = inputs_W.fused_signals[0]

        Ryv = cross_correlation_matrix(Y, V)

        f = np.trace(X[0].T @ Ryv @ X[1])

        return f

    def resolve_ambiguity(
        self,
        X_reference: list[np.ndarray],
        X_current: list[np.ndarray],
        updating_node: int | None = None,
    ) -> list[np.ndarray]:
        """Resolve the sign ambiguity for the CCA problem."""
        X = X_current[0]
        W = X_current[1]
        X_ref = X_reference[0]

        for col in range(self.nb_filters):
            if np.linalg.norm(X_ref[:, col] - X[:, col]) > np.linalg.norm(
                -X_ref[:, col] - X[:, col]
            ):
                X[:, col] = -X[:, col]
                W[:, col] = -W[:, col]

        return [X, W]
