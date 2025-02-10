import numpy as np
from problem_settings import ProblemInputs
from utils import normalize
from abc import ABC, abstractmethod


class DataRetriever:
    def __init__(self, *args, **kwargs) -> None:
        return None

    @abstractmethod
    def get_current_window(self, window_id: int) -> ProblemInputs:
        pass


class MMSEDataRetriever(DataRetriever):
    def __init__(
        self,
        nb_samples: int,
        nb_sensors: int,
        nb_sources: int,
        nb_windows: int,
        rng: np.random.Generator,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        mixture_var: float = 0.5,
        diff_var: float = 1,
    ) -> None:
        self.D = rng.normal(
            loc=0,
            scale=np.sqrt(signal_var),
            size=(nb_sources, nb_samples),
        )
        self.A_0 = rng.normal(
            loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sources)
        )
        self.Delta = rng.normal(
            loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sources)
        )
        self.Delta = (
            self.Delta
            * np.linalg.norm(self.A_0, "fro")
            * diff_var
            / np.linalg.norm(self.Delta, "fro")
        )
        self.noise = rng.normal(
            loc=0,
            scale=np.sqrt(noise_var),
            size=(nb_sensors, nb_samples),
        )

        self.weights = self.weight_function(nb_windows)

    def get_current_window(self, window_id):
        Y_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.D + self.noise
        Y_window = normalize(Y_window)
        mmse_inputs = ProblemInputs(
            fused_signals=[Y_window], global_parameters=[self.D]
        )
        return mmse_inputs

    def weight_function(self, nb_windows):
        segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
        segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
        segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

        weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights
