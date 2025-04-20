import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem

from dasftoolbox.utils import autocorrelation_matrix

import scipy.optimize as opt
import warnings


class QCQPProblem(OptimizationProblem):
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
        """Solve the QCQP problem min 0.5 * E[||X.T @ y(t)||**2] - trace(X.T @ B) s.t. trace(X.T @ Gamma @ X) <= alpha**2; X.T @ c = d."""

        def X_fun(mu: float, problem_inputs: ProblemInputs) -> np.ndarray:
            Y = problem_inputs.fused_signals[0]
            B = problem_inputs.fused_constants[0]
            c = problem_inputs.fused_constants[1]
            Gamma = problem_inputs.fused_quadratics[0]
            d = problem_inputs.global_parameters[1]

            Ryy = autocorrelation_matrix(Y)

            M = Ryy + mu * Gamma
            M_inv = np.linalg.inv(M)
            w = (B.T @ M_inv.T @ c - d) / (c.T @ M_inv @ c)
            X = M_inv @ (B - c @ w.T)

            return X

        def norm_fun(mu: float, problem_inputs: ProblemInputs) -> float:
            Gamma = problem_inputs.fused_quadratics[0]
            alpha = problem_inputs.global_parameters[0]
            X = X_fun(mu, problem_inputs)
            norm = np.trace(X.T @ Gamma @ X) - alpha**2

            return norm

        c = problem_inputs.fused_constants[1]
        Gamma = problem_inputs.fused_quadratics[0]
        alpha = problem_inputs.global_parameters[0]
        d = problem_inputs.global_parameters[1]

        U_c, S_c, _ = np.linalg.svd(Gamma)

        sqrt_Gamma = (U_c @ np.diag(np.sqrt(S_c))).T

        if (
            alpha**2
            == np.linalg.norm(d) ** 2
            / np.linalg.norm(np.linalg.inv(sqrt_Gamma).T @ c) ** 2
        ):
            X_star = np.linalg.inv(Gamma) @ c @ d.T / np.linalg.norm(sqrt_Gamma.T @ c)
        elif (
            alpha**2
            > np.linalg.norm(d) ** 2
            / np.linalg.norm(np.linalg.inv(sqrt_Gamma).T @ c) ** 2
        ):
            if norm_fun(mu=0, problem_inputs=problem_inputs) < 0:
                X_star = X_fun(0, problem_inputs)
            else:
                mu_star = opt.fsolve(norm_fun, 0, problem_inputs)
                X_star = X_fun(mu=mu_star, problem_inputs=problem_inputs)
        else:
            warnings.warn("Infeasible problem")

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the QCQP objective 0.5 * E[||X.T @ y(t)||**2] - trace(X.T @ B)."""
        Y = problem_inputs.fused_signals[0]
        B = problem_inputs.fused_constants[0]

        Ryy = autocorrelation_matrix(Y)

        f = 0.5 * np.trace(X.T @ Ryy @ X) - np.trace(X.T @ B)

        return f
