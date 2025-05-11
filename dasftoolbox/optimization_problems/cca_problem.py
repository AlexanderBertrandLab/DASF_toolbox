import numpy as np
import scipy

from dasftoolbox.optimization_problems.optimization_problem import (
    ConstraintType,
    OptimizationProblem,
)
from dasftoolbox.problem_settings import ConvergenceParameters, ProblemInputs
from dasftoolbox.utils import (
    autocorrelation_matrix,
    cross_correlation_matrix,
    make_symmetric,
)


class CCAProblem(OptimizationProblem):
    """
    CCA problem class.

    :math:`\max_{X,W}\; \mathbb{E}[\\text{trace}(X^T \mathbf{y}(t) \mathbf{v}^T(t) W)]` subject to :math:`\mathbb{E}[X^T \mathbf{y}(t) \mathbf{y}^T(t) X] = I,\; \mathbb{E}[W^T \mathbf{v}(t) \mathbf{v}^T(t) W] = I.`

    Attributes
    ----------
    nb_filters : int
        Number of filters.
    """

    def __init__(self, nb_filters: int, **kwargs) -> None:
        super().__init__(nb_filters=nb_filters, nb_variables=2, **kwargs)

    def solve(
        self,
        problem_inputs: list[ProblemInputs],
        save_solution: bool = False,
        convergence_parameters: ConvergenceParameters | None = None,
        initial_estimate: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """
        Solve the CCA problem.

        The solver returns :math:`Q` eigenvectors corresponding to the :math:`Q` largest generalized eigenvalues of the matrix pair :math:`(R_{\mathbf{yv}}R_{\mathbf{vv}}^{-1}R_{\mathbf{vy}}, R_{\mathbf{yy}})`, where :math:`R_{\mathbf{yy}}` and :math:`R_{\mathbf{vv}}` correspond to the autocorrelation matrix of the observed signal :math:`\mathbf{y}` and :math:`\mathbf{y}`, respectively, while :math:`R_{\mathbf{yv}}=R_{\mathbf{vy}}^T` corresponds to the cross-correlation matrix between the observed signals :math:`\mathbf{y}` and :math:`\mathbf{v}`.

        Parameters
        ----------
        problem_inputs : list[ProblemInputs]
            The problem inputs containing the observed signals :math:`\mathbf{y}` for the first variable and :math:`\mathbf{v}` for the second variable.
        save_solution : bool, optional
            Whether to save the solution or not, by default False
        convergence_parameters : ConvergenceParameters | None, optional
            Convergence parameters, by default None
        initial_estimate : list[np.ndarray] | None, optional
            Initial estimate, by default None

        Returns
        -------
        list[np.ndarray]
            The solution to the CCA problem.
        """
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
        """
        Evaluate the CCA objective :math:`\mathbb{E}[\\text{trace}(X^T \mathbf{y}(t) \mathbf{v}^T(t) W)]`.

        Parameters
        ----------
        X : list[np.ndarray]
            The point to evaluate.
        problem_inputs : ProblemInputs
            The problem inputs containing the observed signals :math:`\mathbf{y}` and :math:`\mathbf{v}`.

        Returns
        -------
        float
            The value of the objective function at point `X` consisting of a list of the two variables :math:`X` and :math:`W`.
        """
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
        """
        Resolve the sign ambiguity for the CCA problem by selecting the sign for each column of the current point so as to minimize the distance to the reference point.

        Parameters
        ----------
        X_reference : list[np.ndarray]
            The reference point.
        X_current : list[np.ndarray]
            The current point.
        updating_node : int | None
            The index of the updating node (for more flexibility), by default None.

        Returns
        -------
        list[np.ndarray]
            A fixed solution of the CCA problem.
        """
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

    def get_problem_constraints(
        self, problem_inputs: list[ProblemInputs]
    ) -> ConstraintType:
        inputs_X = problem_inputs[0]
        inputs_W = problem_inputs[1]

        Y = inputs_X.fused_signals[0]
        V = inputs_W.fused_signals[0]

        Ryy = autocorrelation_matrix(Y)
        Rvv = autocorrelation_matrix(V)

        def equality_constraints(X: list[np.ndarray]) -> list[np.ndarray]:
            return [
                X[0].T @ Ryy @ X[0] - np.eye(self.nb_filters),
                X[1].T @ Rvv @ X[1] - np.eye(self.nb_filters),
            ]

        return equality_constraints, None

    get_problem_constraints.__doc__ = (
        OptimizationProblem.get_problem_constraints.__doc__
    )
