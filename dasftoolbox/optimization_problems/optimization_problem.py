import numpy as np
from dasftoolbox.problem_settings import (
    ProblemInputs,
    ConvergenceParameters,
)
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationProblem(ABC):
    def __init__(
        self,
        nb_filters: int,
        convergence_parameters: ConvergenceParameters | None = None,
        initial_estimate: np.ndarray | list[np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
        nb_variables: int = 1,
    ) -> None:
        """Base class for optimization problems in the DASF format.
        Args:
            nb_filters (int): Number of filters in the problem.
            convergence_parameters (ConvergenceParameters, optional): Convergence parameters for the optimization. Defaults to None.
            initial_estimate (np.ndarray | list[np.ndarray], optional): Initial estimate for the optimization. Defaults to None.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            nb_variables (int, optional): Number of variables in the problem. Defaults to 1.
        """
        self.nb_filters = nb_filters
        self.convergence_parameters = convergence_parameters
        self.initial_estimate = initial_estimate
        self.rng = rng
        self.nb_variables = nb_variables
        self._X_star = None

    @abstractmethod
    def solve(
        self,
        problem_inputs: ProblemInputs | list[ProblemInputs],
        save_solution: bool = False,
        convergence_parameters: ConvergenceParameters | None = None,
        initial_estimate: np.ndarray | list[np.ndarray] | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Method to solve the optimization problem.
        Args:
            problem_inputs (ProblemInputs | list[ProblemInputs]): Problem inputs for the optimization.
            save_solution (bool, optional): Whether to save the solution. Defaults to False.
            convergence_parameters (ConvergenceParameters | None, optional): Convergence parameters for the optimization. Defaults to None.
            initial_estimate (np.ndarray | list[np.ndarray] | None, optional): Initial estimate for the optimization. Defaults to None.
        Returns:
            np.ndarray | list[np.ndarray]: Solution to the optimization problem.
        """
        pass

    @abstractmethod
    def evaluate_objective(
        self,
        X: np.ndarray | list[np.ndarray],
        problem_inputs: ProblemInputs | list[ProblemInputs],
    ) -> float:
        """Method to evaluate the objective function of the optimization problem.
        Args:
            X (np.ndarray | list[np.ndarray]): Point to evaluate.
            problem_inputs (ProblemInputs | list[ProblemInputs]): Problem inputs.
        Returns:
            float: Objective function value.
        """
        pass

    def resolve_ambiguity(
        self,
        X_reference: np.ndarray | list[np.ndarray],
        X_current: np.ndarray | list[np.ndarray],
        updating_node: int | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Method to resolve the ambiguity in the optimization problem.
        Args:
            X_reference (np.ndarray | list[np.ndarray]): Reference solution.
            X_current (np.ndarray | list[np.ndarray]): Current solution.
            updating_node (int, optional): Updating node, for more flexibility. Defaults to None.
        Returns:
            np.ndarray | list[np.ndarray]: Resolved solution.
        """
        return X_current

    @property
    def X_star(self):
        """Property to get the optimal solution of the optimization problem.
        Returns:
            np.ndarray | list[np.ndarray]: Optimal solution.
        """
        if self._X_star is None:
            logger.warning("The problem has not been solved yet.")
        return self._X_star
