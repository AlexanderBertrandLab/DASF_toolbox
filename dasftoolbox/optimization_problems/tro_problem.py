import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.optimization_problems.optimization_problem import OptimizationProblem
from dasftoolbox.problem_settings import ConvergenceParameters

from dasftoolbox.utils import autocorrelation_matrix

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TROProblem(OptimizationProblem):
    """
    TRO problem class.

    Attributes
    ----------
    nb_filters : int
        Number of filters.
    convergence_parameters : ConvergenceParameters
        Convergence parameters of the solver.
    """

    def __init__(
        self, nb_filters: int, convergence_parameters: ConvergenceParameters
    ) -> None:
        super().__init__(
            nb_filters=nb_filters, convergence_parameters=convergence_parameters
        )

    def solve(
        self,
        problem_inputs: ProblemInputs,
        save_solution: bool = False,
        convergence_parameters: ConvergenceParameters | None = None,
        initial_estimate: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the TRO problem :math:`\max_X\; \\frac{\mathbb{E}[\|X^T \mathbf{y}(t)\|^2]}{\mathbb{E}[\|X^T \mathbf{v}(t)\|^2]}` subject to :math:`X^T \Gamma X = I`.

        Parameters
        ----------
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signals :math:`\mathbf{y}`, :math:`\mathbf{v}` and the :math:`\Gamma` matrix in the field `fused_quadratics`.
        save_solution : bool, optional
            Whether to save the solution or not, by default False
        convergence_parameters : ConvergenceParameters | None, optional
            Convergence parameters, by default None
        initial_estimate : np.ndarray | None, optional
            Initial estimate, by default None

        Returns
        -------
        np.ndarray
            The solution to the TRO problem.
        """
        Y = problem_inputs.fused_signals[0]
        V = problem_inputs.fused_signals[1]
        Gamma = problem_inputs.fused_quadratics[0]

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
        f = self.evaluate_objective(X, problem_inputs)

        if convergence_parameters is None:
            if self.convergence_parameters is not None:
                convergence_parameters = self.convergence_parameters
            else:
                convergence_parameters = ConvergenceParameters()

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)

        U_c, S_c, _ = np.linalg.svd(Gamma)

        Y_t = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Y
        V_t = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ V

        Kyy = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Ryy @ U_c @ np.diag(np.sqrt(1 / S_c))
        Kvv = np.diag(np.sqrt(1 / S_c)) @ U_c.T @ Rvv @ U_c @ np.diag(np.sqrt(1 / S_c))

        i = 0
        while i < convergence_parameters.max_iterations:
            eigvals, eigvecs = np.linalg.eig(Kyy - f * Kvv)
            indices = np.argsort(eigvals)[::-1]

            X_old = X
            X = eigvecs[:, indices[0 : self.nb_filters]]
            f_old = f
            Y_list_t = [Y_t, V_t]
            problem_inputs_t = ProblemInputs(fused_signals=Y_list_t)
            f = self.evaluate_objective(X, problem_inputs_t)

            i += 1

            if (convergence_parameters.objective_tolerance is not None) and (
                np.absolute(f - f_old) <= convergence_parameters.objective_tolerance
            ):
                break
            if (convergence_parameters.argument_tolerance is not None) and (
                np.linalg.norm(X - X_old, "fro")
                <= convergence_parameters.argument_tolerance
            ):
                break

        X_star = U_c @ np.diag(np.sqrt(1 / S_c)) @ X

        if save_solution:
            self._X_star = X_star

        return X_star

    def evaluate_objective(self, X: np.ndarray, problem_inputs: ProblemInputs) -> float:
        """
        Evaluate the TRO objective :math:`\\frac{\mathbb{E}[\|X^T \mathbf{y}(t)\|^2]}{\mathbb{E}[\|X^T \mathbf{v}(t)\|^2]}`.

        Parameters
        ----------
        X : np.ndarray
            The point to evaluate.
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signals :math:`\mathbf{y}`, :math:`\mathbf{v}` and the :math:`\Gamma` matrix in the field `fused_quadratics`.

        Returns
        -------
        float
            The value of the objective function at point `X`.
        """

        Y = problem_inputs.fused_signals[0]
        V = problem_inputs.fused_signals[1]

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)

        f = np.trace(X.T @ Ryy @ X) / np.trace(X.T @ Rvv @ X)

        return f

    def resolve_ambiguity(
        self,
        X_reference: np.ndarray | list[np.ndarray],
        X_current: np.ndarray | list[np.ndarray],
        updating_node: int | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Resolve the sign ambiguity for the TRO problem by selecting the sign for each column of the current point so as to minimize the distance to the reference point.

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
            A fixed solution of the TRO problem.
        """

        for col in range(self.nb_filters):
            if np.linalg.norm(X_reference[:, col] - X_current[:, col]) > np.linalg.norm(
                -X_reference[:, col] - X_current[:, col]
            ):
                X_current[:, col] = -X_current[:, col]

        return X_current
