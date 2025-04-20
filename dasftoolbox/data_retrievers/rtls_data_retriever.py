import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.utils import normalize

from dasftoolbox.data_retrievers.data_retriever import (
    DataRetriever,
    DataWindowParameters,
)


class RTLSDataRetriever(DataRetriever):
    def __init__(
        self,
        data_window_params: DataWindowParameters,
        nb_sensors: int,
        nb_sources: int,
        nb_windows: int,
        rng: np.random.Generator,
        signal_var: float = 1,
        noise_var: float = 0.1,
        noise_var_d: float = 0.02,
        mixture_var: float = 0.5,
        diff_var: float = 0.1,
    ) -> None:
        self.data_window_params = data_window_params
        nb_samples = data_window_params.window_length
        self.nb_sensors = nb_sensors

        self.S = rng.normal(
            loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples)
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

        d_noise = rng.normal(
            loc=0, scale=np.sqrt(noise_var_d), size=(nb_sources, nb_samples)
        )

        self.D = self.S + d_noise

        self.L = np.diag(1 + rng.normal(loc=0, scale=0.1, size=(nb_sensors,)))

        self.delta = 1

        self.weights = self.weight_function(nb_windows)

    def get_current_window(self, window_id: int) -> ProblemInputs:
        Y_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.S + self.noise
        Y_window = normalize(Y_window)

        rtls_inputs = ProblemInputs(
            fused_signals=[Y_window],
            fused_constants=[self.L],
            fused_quadratics=[np.eye(self.nb_sensors)],
            global_parameters=[self.D, self.delta],
        )

        return rtls_inputs

    def weight_function(self, nb_windows: int) -> np.ndarray:
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights
