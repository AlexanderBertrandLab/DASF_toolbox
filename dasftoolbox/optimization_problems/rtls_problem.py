import logging

import numpy as np
import scipy.optimize as opt

from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem
from dasftoolbox.problem_settings import ConvergenceParameters, ProblemInputs
from dasftoolbox.utils import autocorrelation_matrix, make_symmetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTLSProblem(OptimizationProblem):
    """
    RTLS problem class.

    :math:`\max_X\; \\frac{\mathbb{E}[\|X^T \mathbf{y}(t)-\mathbf{d}(t)\|^2]}{1+X^T\Gamma X}` subject to :math:`\|X^T L\|^2 \\leq \delta^2.`

    Attributes
    ----------
    nb_filters : int
        Number of filters.
    convergence_parameters : ConvergenceParameters
        Convergence parameters of the solver.
    initial_estimate : np.ndarray | None
        Initial estimate.
    """

    def __init__(
        self,
        nb_filters: int,
        convergence_parameters: ConvergenceParameters,
        initial_estimate: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            nb_filters=nb_filters,
            convergence_parameters=convergence_parameters,
            initial_estimate=initial_estimate,
        )

    def solve(
        self,
        problem_inputs: ProblemInputs,
        save_solution: bool = False,
        convergence_parameters: ConvergenceParameters | None = None,
        initial_estimate: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the RTLS problem.

        The solver implements an iterative algorithm based on the Dinkelbach method, solving a convex quadratic problem at each iteration.

        Parameters
        ----------
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signal :math:`\mathbf{y}`, the observed signal :math:`\mathbf{d}` and the constants :math:`\Gamma` in the field `fused_quadratics`, :math:`L` and :math:`\delta`.
        save_solution : bool, optional
            Whether to save the solution or not, by default False
        convergence_parameters : ConvergenceParameters | None, optional
            Convergence parameters, by default None
        initial_estimate : np.ndarray | None, optional
            Initial estimate, by default None

        Returns
        -------
        np.ndarray
            The solution to the RTLS problem.
        """
        Y = problem_inputs.fused_signals[0]
        L = problem_inputs.fused_constants[0]
        Gamma = problem_inputs.fused_quadratics[0]
        d = problem_inputs.global_parameters[0]
        delta = problem_inputs.global_parameters[1]

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
        i = 0

        f = self.evaluate_objective(X, problem_inputs)

        convergence_parameters = (
            self.convergence_parameters
            if self.convergence_parameters
            else ConvergenceParameters()
        )

        Ryy = autocorrelation_matrix(Y)
        ryd = np.sum(Y * d, axis=1) / np.size(Y, 1)
        LL = L @ L.T
        LL = make_symmetric(LL)
        Gamma = make_symmetric(Gamma)

        def objective_function(
            mult: float,
            L: np.ndarray,
            Ryy: np.ndarray,
            f: float,
            Gamma: np.ndarray,
            ryd: np.ndarray,
            delta: float,
        ) -> float:
            return (
                np.linalg.norm(
                    L.T @ np.linalg.inv(Ryy - f * Gamma + mult * (L @ L.T)) @ ryd
                )
                ** 2
                - delta**2
            )

        while i < convergence_parameters.max_iterations:
            X_old = X
            X_f = np.linalg.inv(Ryy - f * Gamma) @ ryd
            if X_f.T @ LL @ X_f < delta**2:
                X = X_f
            else:
                bracket = [0, 1000]
                initial_guess = 0

                opt_sol = opt.root_scalar(
                    objective_function,
                    args=(L, Ryy, f, Gamma, ryd, delta),
                    bracket=bracket,
                    method="bisect",
                    x0=initial_guess,
                )
                mult_star = opt_sol.root

                X = np.linalg.inv(Ryy - f * Gamma + mult_star * LL) @ ryd

            f_old = f
            f = self.evaluate_objective(X, problem_inputs)

            if (convergence_parameters.objective_tolerance is not None) and (
                np.absolute(f - f_old) <= convergence_parameters.objective_tolerance
            ):
                break
            if (convergence_parameters.argument_tolerance is not None) and (
                np.linalg.norm(X - X_old, "fro")
                <= convergence_parameters.argument_tolerance
            ):
                break

            i += 1

        X_star = np.expand_dims(X, axis=1)

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """
        Evaluate the RTLS objective :math:`\\frac{\mathbb{E}[\|X^T \mathbf{y}(t)-\mathbf{d}(t)\|^2]}{1+X^T\Gamma X}`.

        Parameters
        ----------
        X : np.ndarray
            The point to evaluate.
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signal :math:`\mathbf{y}`, the observed signal :math:`\mathbf{d}` and the constants :math:`\Gamma` in the field `fused_quadratics`, :math:`L` and :math:`\delta`.

        Returns
        -------
        float
            The value of the objective function at point `X`.
        """
        Y = problem_inputs.fused_signals[0]
        Gamma = problem_inputs.fused_quadratics[0]
        d = problem_inputs.global_parameters[0]

        Ryy = autocorrelation_matrix(Y)
        ryd = np.sum(Y * d, axis=1) / np.size(Y, 1)
        rdd = np.sum(d * d, axis=1) / np.size(d, 1)

        f = (X.T @ Ryy @ X - 2 * X.T @ ryd + rdd) / (X.T @ Gamma @ X + 1)

        return f
