import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.utils import normalize, autocorrelation_matrix

from dasftoolbox.data_retrievers.data_retriever import (
    DataRetriever,
    DataWindowParameters,
)


class QCQPDataRetriever(DataRetriever):
    def __init__(
        self,
        data_window_params: DataWindowParameters,
        nb_filters: int,
        nb_sensors: int,
        nb_windows: int,
        rng: np.random.Generator,
        nb_sources: int = 10,
        signal_var: float = 1,
        noise_var: float = 0.1,
        mixture_var: float = 0.5,
        diff_var: float = 0.1,
    ) -> None:
        self.data_window_params = data_window_params
        self.nb_filters = nb_filters
        nb_samples = data_window_params.window_length
        self.nb_sensors = nb_sensors

        self.D = rng.normal(
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
        Y = self.A_0 @ self.D + self.noise
        Ryy = autocorrelation_matrix(Y)
        self.B = rng.standard_normal(size=(nb_sensors, self.nb_filters))

        def generate_valid_samples(
            rng: np.random.Generator,
            nb_sensors: int,
            nb_filters: int,
            Ryy: np.ndarray,
            B: np.ndarray,
            max_attempts: int = 1000,
        ):
            attempts = 0
            while attempts < max_attempts:
                c = rng.standard_normal(size=(nb_sensors, 1))
                d = rng.standard_normal(size=(nb_filters, 1))
                w = (B.T @ np.linalg.inv(Ryy).T @ c - d) / (
                    c.T @ np.linalg.inv(Ryy) @ c
                )
                X = np.linalg.inv(Ryy) @ (B - c @ w.T)

                toss = rng.integers(0, 1, endpoint=True)
                alpha = rng.standard_normal()
                alpha = alpha**2
                if toss != 0:
                    alpha = np.sqrt(np.linalg.norm(X, ord="fro") ** 2 + alpha**2)

                if alpha**2 >= np.linalg.norm(d) ** 2 / np.linalg.norm(c) ** 2:
                    return c, d, w, alpha

                attempts += 1

            raise ValueError("Failed to generate valid samples after maximum attempts")

        try:
            self.c, self.d, self.w, self.alpha = generate_valid_samples(
                rng=rng,
                nb_sensors=nb_sensors,
                nb_filters=self.nb_filters,
                Ryy=Ryy,
                B=self.B,
            )
        except ValueError as e:
            print(e)

        self.weights = self.weight_function(nb_windows)

    def get_data_window(self, window_id: int) -> ProblemInputs:
        Y_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.D + self.noise
        Y_window = normalize(Y_window)

        qcqp_inputs = ProblemInputs(
            fused_signals=[Y_window],
            fused_constants=[self.B, self.c],
            fused_quadratics=[np.eye(self.nb_sensors)],
            global_parameters=[self.alpha, self.d],
        )

        return qcqp_inputs

    get_data_window.__doc__ = DataRetriever.get_data_window.__doc__

    def weight_function(self, nb_windows: int) -> np.ndarray:
        """
        Weight function :math:`w` for the non-stationarity of the signals. Here, a piecewise linear function is used.

        Parameters
        ----------
        nb_windows : int
            Number of windows of data.
        """
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights
