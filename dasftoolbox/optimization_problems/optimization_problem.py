import logging
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np

from dasftoolbox.problem_settings import (
    ConvergenceParameters,
    ProblemInputs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ConstraintType = (
    Callable[[np.ndarray | list[np.ndarray]], np.ndarray]
    | list[Callable[[np.ndarray | list[np.ndarray]], np.ndarray]]
)
"""
ConstraintType
    A callable or list of callables representing constraints.

    Each callable takes a NumPy array or list of arrays and returns a NumPy array.
    This is used to represent the equality and inequality constraints of the optimization problem.
"""


class OptimizationProblem(ABC):
    """Base class for optimization problems in the DASF format.

    Attributes
    ----------
    nb_filters : int
        Number of filters in the problem.
    convergence_parameters : ConvergenceParameters | None, optional
        Convergence parameters for the optimization problem. Defaults to None.
    initial_estimate : np.ndarray | list[np.ndarray] | None, optional
        Initial estimate for the optimization problem. Defaults to None.
    rng : np.random.Generator | None, optional
        Random number generator. Defaults to None.
    nb_variables : int, optional
        Number of variables in the problem. Defaults to 1.
    """

    def __init__(
        self,
        nb_filters: int,
        convergence_parameters: ConvergenceParameters | None = None,
        initial_estimate: np.ndarray | list[np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
        nb_variables: int = 1,
        **kwargs,
    ) -> None:
        self.nb_filters = nb_filters
        self.convergence_parameters = convergence_parameters
        self.initial_estimate = initial_estimate
        self.rng = rng
        self.nb_variables = nb_variables
        self._X_star = None

        self._init_args = dict(
            nb_filters=nb_filters,
            convergence_parameters=convergence_parameters,
            initial_estimate=initial_estimate,
            rng=rng,
            nb_variables=nb_variables,
            **kwargs,
        )

    @abstractmethod
    def solve(
        self,
        problem_inputs: ProblemInputs | list[ProblemInputs],
        save_solution: bool = False,
        convergence_parameters: ConvergenceParameters | None = None,
        initial_estimate: np.ndarray | list[np.ndarray] | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Method to solve the optimization problem.

        Parameters
        ----------
        problem_inputs : ProblemInputs or list of ProblemInputs
            Problem inputs for the optimization problem.
        save_solution : bool, optional
            Whether to save the optimal solution. Defaults to False.
        convergence_parameters : ConvergenceParameters or None, optional
            Convergence parameters for the optimization problem. Defaults to None.
        initial_estimate : np.ndarray, list of np.ndarray or None, optional
            Initial estimate for the optimization problem. Defaults to None.

        Returns
        -------
        np.ndarray or list of np.ndarray
            Solution to the optimization problem.
        """
        pass

    @abstractmethod
    def evaluate_objective(
        self,
        X: np.ndarray | list[np.ndarray],
        problem_inputs: ProblemInputs | list[ProblemInputs],
    ) -> float:
        """
        Method to evaluate the objective function of the optimization problem.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Point to evaluate.
        problem_inputs : ProblemInputs or list of ProblemInputs
            Problem inputs.

        Returns
        -------
        float
            Objective function value.
        """
        pass

    def resolve_ambiguity(
        self,
        X_reference: np.ndarray | list[np.ndarray],
        X_current: np.ndarray | list[np.ndarray],
        updating_node: int | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Method to resolve the ambiguity in the optimization problem.

        Parameters
        ----------
        X_reference : np.ndarray or list of np.ndarray
            Reference solution.
        X_current : np.ndarray or list of np.ndarray
            Current solution.
        updating_node : int, optional
            Updating node, for more flexibility. Defaults to None.

        Returns
        -------
        np.ndarray or list of np.ndarray
            Resolved solution.
        """
        return X_current

    @abstractmethod
    def get_problem_constraints(
        self,
        problem_inputs: ProblemInputs | list[ProblemInputs],
    ) -> Tuple[ConstraintType | None, ConstraintType | None] | None:
        """
        Return the constraints of the optimization problem. By convention, every ineuality constraint is given by :math:`h(X)\leq 0`.

        If the problem is unconstrained, the method should return None.

        Parameters
        ----------
        problem_inputs : ProblemInputs or list of ProblemInputs
            The inputs of the problem.

        Returns
        -------
        constraints : tuple of (ConstraintType or None, ConstraintType or None) or None
            If constraints are defined, returns a tuple:
            - First element: equality constraints (or None if not present).
            - Second element: inequality constraints (or None if not present).
            If no constraints are defined, returns None.
        """
        pass

    def X_satisfies_constraints(
        self,
        X: np.ndarray | list[np.ndarray],
        problem_inputs: ProblemInputs | list[ProblemInputs],
    ) -> bool:
        """
        Verify whether the constraints of the problem are satisfied at a given point.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Point at which to check the constraints.

        Returns
        -------
        bool
            Boolean which is `True` if all constraints are satisfied or if the problem is unbounded, and `false` otherwise.
        """
        tolerance = 1e-8
        all_constraints = self.get_problem_constraints(problem_inputs=problem_inputs)
        if all_constraints is None:
            return True
        equality_constraints, inequality_constraints = all_constraints
        if equality_constraints is None and inequality_constraints is None:
            raise ValueError(
                "Both equality and inequality constraints are `None`. If the problem is unconstrained, please return `None` from `get_problem_constraints`."
            )
        elif equality_constraints is None and inequality_constraints is not None:
            valid_constraints = (
                [inequality_constraints]
                if isinstance(inequality_constraints, Callable)
                else inequality_constraints
            )
            return np.all(
                [np.all(constr(X) <= tolerance) for constr in valid_constraints]
            )
        elif equality_constraints is not None and inequality_constraints is None:
            valid_constraints = (
                [equality_constraints]
                if isinstance(equality_constraints, Callable)
                else equality_constraints
            )
            return np.all(
                [
                    np.allclose(constr(X), 0.0, atol=tolerance)
                    for constr in valid_constraints
                ]
            )
        else:
            valid_equality_constraints = (
                [equality_constraints]
                if isinstance(equality_constraints, Callable)
                else equality_constraints
            )
            check_equality = np.all(
                [
                    np.allclose(constr(X), 0.0, atol=tolerance)
                    for constr in valid_equality_constraints
                ]
            )
            valid_inequality_constraints = (
                [inequality_constraints]
                if isinstance(inequality_constraints, Callable)
                else inequality_constraints
            )
            check_inequality = np.all(
                [
                    np.all(constr(X) <= tolerance)
                    for constr in valid_inequality_constraints
                ]
            )

            return check_equality and check_inequality

    @property
    def X_star(self):
        """
        Property to get the optimal solution of the optimization problem.

        Returns
        -------
        np.ndarray or list of np.ndarray
            Optimal solution.
        """
        if self._X_star is None:
            logger.warning("The problem has not been solved yet.")
        return self._X_star

    def copy(self):
        """
        Return a copy of the class instance.
        """
        return self.__class__(**self._init_args)

    def reset_X_star(self) -> None:
        """
        Reset the value of :math:`X^*` saved in the class.
        """
        self._X_star = None
