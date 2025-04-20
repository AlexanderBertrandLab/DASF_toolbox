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
        pass

    @abstractmethod
    def evaluate_objective(
        self,
        X: np.ndarray | list[np.ndarray],
        problem_inputs: ProblemInputs | list[ProblemInputs],
    ) -> float:
        pass

    def resolve_ambiguity(
        self,
        X_reference: np.ndarray | list[np.ndarray],
        X_current: np.ndarray | list[np.ndarray],
        updating_node: int | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        return X_current

    @property
    def X_star(self):
        if self._X_star is None:
            logger.warning("The problem has not been solved yet.")
        return self._X_star
