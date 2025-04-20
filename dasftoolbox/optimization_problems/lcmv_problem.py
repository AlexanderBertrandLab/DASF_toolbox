import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem

from dasftoolbox.utils import autocorrelation_matrix


class LCMVProblem(OptimizationProblem):
    def __init__(self, nb_filters: int) -> None:
        super().__init__(nb_filters=nb_filters)

    def solve(
        self,
        problem_inputs: ProblemInputs,
        save_solution: bool = False,
        convergence_parameters=None,
        initial_estimate=None,
    ) -> np.ndarray:
        """Solve the LCMV problem min E[||X.T @ y(t)||**2] s.t. X.T @ B = H."""
        Y = problem_inputs.fused_signals[0]
        B = problem_inputs.fused_constants[0]
        H = problem_inputs.global_parameters[0]

        Ryy = autocorrelation_matrix(Y)

        X_star = (
            np.linalg.inv(Ryy) @ B @ np.linalg.inv(B.T @ np.linalg.inv(Ryy) @ B) @ H.T
        )

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """Evaluate the LCMV objective E[||X.T @ y(t)||**2]."""
        Y = problem_inputs.fused_signals[0]

        Ryy = autocorrelation_matrix(Y)

        f = np.trace(X.T @ Ryy @ X)

        return f
