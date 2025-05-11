import numpy as np
import scipy

from dasftoolbox.optimization_problems.optimization_problem import (
    ConstraintType,
    OptimizationProblem,
)
from dasftoolbox.problem_settings import ConvergenceParameters, ProblemInputs
from dasftoolbox.utils import autocorrelation_matrix


class GEVDProblem(OptimizationProblem):
    """
    GEVD problem class.

    :math:`\max_X\; \mathbb{E}[\|X^T \mathbf{y}(t)\|^2]` subject to :math:`\mathbb{E}[X^T \mathbf{v}(t) \mathbf{v}^T(t) X] = I.`

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
        Solve the GEVD problem.

        The solver returns the :math:`Q` eigenvectors corresponding to the :math:`Q` largest generalized eigenvalues of the matrix pair :math:`(R_{\mathbf{yy}}, R_{\mathbf{vv}})`, where :math:`R_{\mathbf{yy}}` and :math:`R_{\mathbf{vv}}` correspond to the autocorrelation matrix of the signals :math:`\mathbf{y}` and :math:`\mathbf{v}`, respectively.

        Parameters
        ----------
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signals :math:`\mathbf{y}` and :math:`\mathbf{v}`.
        save_solution : bool, optional
            Whether to save the solution or not, by default False
        convergence_parameters : ConvergenceParameters | None, optional
            Convergence parameters, by default None
        initial_estimate : np.ndarray | None, optional
            Initial estimate, by default None

        Returns
        -------
        np.ndarray
            The solution to the GEVD problem.
        """
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
        """
        Evaluate the GEVD objective :math:`\mathbb{E}[\|X^T \mathbf{y}(t)\|^2]`.

        Parameters
        ----------
        X : np.ndarray
            The point to evaluate.
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signals :math:`\mathbf{y}` and :math:`\mathbf{v}`.

        Returns
        -------
        float
            The value of the objective function at point `X`.
        """
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
        """
        Resolve the sign ambiguity for the GEVD problem by selecting the sign for each column of the current point so as to minimize the distance to the reference point.

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
            A fixed solution of the GEVD problem.
        """

        for col in range(self.nb_filters):
            if np.linalg.norm(X_reference[:, col] - X_current[:, col]) > np.linalg.norm(
                -X_reference[:, col] - X_current[:, col]
            ):
                X_current[:, col] = -X_current[:, col]

        return X_current

    def get_problem_constraints(self, problem_inputs: ProblemInputs) -> ConstraintType:
        V = problem_inputs.fused_signals[1]
        Rvv = autocorrelation_matrix(V)

        def equality_constraint(X: np.ndarray) -> np.ndarray:
            return X.T @ Rvv @ X - np.eye(self.nb_filters)

        return equality_constraint, None

    get_problem_constraints.__doc__ = (
        OptimizationProblem.get_problem_constraints.__doc__
    )
