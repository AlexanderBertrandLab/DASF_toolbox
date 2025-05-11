import warnings

import numpy as np
import scipy.optimize as opt

from dasftoolbox.optimization_problems.optimization_problem import (
    ConstraintType,
    OptimizationProblem,
)
from dasftoolbox.problem_settings import ConvergenceParameters, ProblemInputs
from dasftoolbox.utils import autocorrelation_matrix


class QCQPProblem(OptimizationProblem):
    """
    QCQP problem class.

    :math:`\min_X\; \\frac{1}{2}\mathbb{E}[\| X^T \mathbf{y}(t)\|^2] - \\text{trace}(X^T B)` subject to :math:`\\text{trace}(X^T \Gamma  X) \\leq \\alpha^2,\; X^T \mathbf{c} = \mathbf{d}.`

    Attributes
    ----------
    nb_filters : int
        Number of filters.
    """

    def __init__(self, nb_filters: int, **kwargs) -> None:
        super().__init__(nb_filters=nb_filters, **kwargs)

    def solve(
        self,
        problem_inputs: ProblemInputs,
        save_solution: bool = False,
        convergence_parameters: ConvergenceParameters | None = None,
        initial_estimate: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the QCQP problem.

        The solver is derived from the KKT conditions of the problem.

        Parameters
        ----------
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signal :math:`\mathbf{y}` and the matrices and vectors :math:`B`, :math:`\Gamma`, :math:`\mathbf{c}` and :math:`\mathbf{d}`.
        save_solution : bool, optional
            Whether to save the solution or not, by default False
        convergence_parameters : ConvergenceParameters | None, optional
            Convergence parameters, by default None
        initial_estimate : np.ndarray | None, optional
            Initial estimate, by default None

        Returns
        -------
        np.ndarray
            The solution to the QCQP problem.
        """

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
        """
        Evaluate the QCQP objective :math:`\\frac{1}{2}\mathbb{E}[\| X^T \mathbf{y}(t)\|^2] - \\text{trace}(X^T B)`.

        Parameters
        ----------
        X : np.ndarray
            The point to evaluate.
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signal :math:`\mathbf{y}` and the matrices and vectors :math:`B`, :math:`\Gamma`, :math:`\mathbf{c}` and :math:`\mathbf{d}`.

        Returns
        -------
        float
            The value of the objective function at point `X`.
        """
        Y = problem_inputs.fused_signals[0]
        B = problem_inputs.fused_constants[0]

        Ryy = autocorrelation_matrix(Y)

        f = 0.5 * np.trace(X.T @ Ryy @ X) - np.trace(X.T @ B)

        return f

    def get_problem_constraints(self, problem_inputs: ProblemInputs) -> ConstraintType:
        c = problem_inputs.fused_constants[1]
        Gamma = problem_inputs.fused_quadratics[0]
        alpha = problem_inputs.global_parameters[0]
        d = problem_inputs.global_parameters[1]

        def equality_constraint(X: np.ndarray) -> np.ndarray:
            return X.T @ c - d

        def inequality_constraint(X: np.ndarray) -> np.ndarray:
            return np.trace(X.T @ Gamma @ X) - alpha**2

        return equality_constraint, inequality_constraint

    get_problem_constraints.__doc__ = (
        OptimizationProblem.get_problem_constraints.__doc__
    )
