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
        """
        Solve the LCMV problem
        :math:`\min_X\; \mathbb{E}[\| X^T \mathbf{y}(t)\|^2]` subject to :math:`X^TB=H`.

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
            The solution to the LCMV problem.
        """
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
        """
        Evaluate the LCMV objective :math:`\mathbb{E}[|\X^T \mathbf{y}(t)\|^2]`.

        Parameters
        ----------
        X : np.ndarray
            The point to evaluate.
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signal :math:`\mathbf{y}` and the matrices :math:`B` and  :math:`H`.
        
        Returns
        -------
        float
            The value of the objective function at the point X.
        """
        Y = problem_inputs.fused_signals[0]

        Ryy = autocorrelation_matrix(Y)

        f = np.trace(X.T @ Ryy @ X)

        return f
