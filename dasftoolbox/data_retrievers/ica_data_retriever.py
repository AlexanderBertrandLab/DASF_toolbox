import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.utils import normalize

from dasftoolbox.data_retrievers.data_retriever import (
    DataRetriever,
    DataWindowParameters,
)

from scipy import signal


class ICADataRetriever(DataRetriever):
    def __init__(
        self,
        data_window_params: DataWindowParameters,
        nb_sensors: int,
        nb_windows: int,
        rng: np.random.Generator,
        signal_var: float = 1,
        mixture_var: float = 0.5,
        diff_var: float = 0.1,
    ) -> None:
        self.data_window_params = data_window_params
        nb_samples = data_window_params.window_length
        coef = 1 / np.linspace(start=1, stop=nb_sensors, num=nb_sensors)
        coef = np.expand_dims(coef, axis=1)
        self.D = coef * (rng.random(size=(nb_sensors, nb_samples)) - 0.5) + (
            1 - coef
        ) * rng.normal(loc=0, scale=signal_var, size=(nb_sensors, nb_samples))

        self.D[0, :] = np.sin(np.linspace(-np.pi, np.pi, nb_samples) * 9)
        self.D[1, :] = signal.square(
            np.linspace(-5 * np.pi, 5 * np.pi, nb_samples) * 5, duty=0.5
        )
        self.A_0 = rng.normal(
            loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sensors)
        )
        self.Delta = rng.normal(
            loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sensors)
        )
        self.Delta = (
            self.Delta
            * np.linalg.norm(self.A_0, "fro")
            * diff_var
            / np.linalg.norm(self.Delta, "fro")
        )

        self.weights = self.weight_function(nb_windows)

    def get_current_window(self, window_id: int) -> ProblemInputs:
        Y_window = (self.A_0 + self.Delta * self.weights[window_id]) @ self.D
        Y_window = normalize(Y_window)
        ica_inputs = ProblemInputs(fused_signals=[Y_window])
        return ica_inputs

    def weight_function(self, nb_windows: int) -> np.ndarray:
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights
