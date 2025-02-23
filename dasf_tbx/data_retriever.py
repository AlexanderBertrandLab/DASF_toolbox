import numpy as np
from dasf_tbx.problem_settings import ProblemInputs
from dasf_tbx.utils import normalize, autocorrelation_matrix
from abc import ABC, abstractmethod
from scipy import signal
from dataclasses import dataclass


@dataclass
class DataWindowParameters:
    window_length: int
    nb_window_reuse: int = 1
    sliding_window_offset: int | None = None

    def __post_init__(self) -> None:
        if self.sliding_window_offset is None:
            self.sliding_window_offset = self.window_length


def get_stationary_setting(window_length: int, iterations: int) -> DataWindowParameters:
    return DataWindowParameters(
        window_length=window_length, nb_window_reuse=iterations, sliding_window_offset=0
    )


class DataRetriever:
    def __init__(
        self, data_window_params: DataWindowParameters, *args, **kwargs
    ) -> None:
        return None

    @abstractmethod
    def get_current_window(self, window_id: int) -> ProblemInputs | list[ProblemInputs]:
        pass


class MMSEDataRetriever(DataRetriever):
    def __init__(
        self,
        data_window_params: DataWindowParameters,
        nb_sensors: int,
        nb_sources: int,
        nb_windows: int,
        rng: np.random.Generator,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        mixture_var: float = 0.5,
        diff_var: float = 1,
    ) -> None:
        self.data_window_params = data_window_params
        nb_samples = data_window_params.window_length
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

    def get_current_window(self, window_id: int) -> ProblemInputs:
        Y_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.D + self.noise
        Y_window = normalize(Y_window)
        mmse_inputs = ProblemInputs(
            fused_signals=[Y_window], global_parameters=[self.D]
        )
        return mmse_inputs

    def weight_function(self, nb_windows: int) -> np.ndarray:
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights


class LCMVDataRetriever(DataRetriever):
    def __init__(
        self,
        data_window_params: DataWindowParameters,
        nb_sensors: int,
        nb_sources: int,
        nb_windows: int,
        rng: np.random.Generator,
        nb_filters: int,
        nb_steering: int | None = None,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        mixture_var: float = 0.5,
        diff_var: float = 1,
    ) -> None:
        self.data_window_params = data_window_params
        nb_samples = data_window_params.window_length
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
        self.nb_filters = nb_filters
        self.nb_steering = nb_steering if nb_steering is not None else nb_filters
        self.B = self.A_0[:, 0:nb_steering]
        self.H = rng.standard_normal(size=(self.nb_filters, self.nb_steering))

        self.weights = self.weight_function(nb_windows)

    def get_current_window(self, window_id: int) -> ProblemInputs:
        Y_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.D + self.noise
        Y_window = normalize(Y_window)
        lcmv_inputs = ProblemInputs(
            fused_signals=[Y_window],
            fused_constants=[self.B],
            global_parameters=[self.H],
        )
        return lcmv_inputs

    def weight_function(self, nb_windows: int) -> np.ndarray:
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights


class GEVDDataRetriever(DataRetriever):
    def __init__(
        self,
        data_window_params: DataWindowParameters,
        nb_sensors: int,
        nb_sources: int,
        nb_windows: int,
        rng: np.random.Generator,
        latent_dim: int | None = None,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        mixture_var: float = 0.5,
        diff_var: float = 1,
    ) -> None:
        self.data_window_params = data_window_params
        nb_samples = data_window_params.window_length
        self.latent_dim = latent_dim if latent_dim is not None else 2 * nb_sources
        self.D1 = rng.normal(
            loc=0,
            scale=np.sqrt(signal_var),
            size=(nb_sources, nb_samples),
        )
        self.D2 = rng.normal(
            loc=0,
            scale=np.sqrt(signal_var),
            size=(self.latent_dim - nb_sources, nb_samples),
        )
        self.A_0 = rng.normal(
            loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sources)
        )
        self.B_0 = rng.normal(
            loc=0,
            scale=np.sqrt(mixture_var),
            size=(nb_sensors, self.latent_dim - nb_sources),
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

    def get_current_window(self, window_id: int) -> ProblemInputs:
        V_window = self.B_0 @ self.D1 + self.noise
        V_window = normalize(V_window)
        Y_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.D2 + V_window
        Y_window = normalize(Y_window)
        gevd_inputs = ProblemInputs(fused_signals=[Y_window, V_window])
        return gevd_inputs

    def weight_function(self, nb_windows: int) -> np.ndarray:
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights


class TRODataRetriever(DataRetriever):
    def __init__(
        self,
        data_window_params: DataWindowParameters,
        nb_sensors: int,
        nb_sources: int,
        nb_windows: int,
        rng: np.random.Generator,
        latent_dim: int | None = None,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        mixture_var: float = 0.5,
        diff_var: float = 1,
    ) -> None:
        self.data_window_params = data_window_params
        nb_samples = data_window_params.window_length
        self.latent_dim = latent_dim if latent_dim is not None else 2 * nb_sources
        self.D1 = rng.normal(
            loc=0,
            scale=np.sqrt(signal_var),
            size=(nb_sources, nb_samples),
        )
        self.D2 = rng.normal(
            loc=0,
            scale=np.sqrt(signal_var),
            size=(self.latent_dim - nb_sources, nb_samples),
        )
        self.A_0 = rng.normal(
            loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sources)
        )
        self.B_0 = rng.normal(
            loc=0,
            scale=np.sqrt(mixture_var),
            size=(nb_sensors, self.latent_dim - nb_sources),
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

        self.Gamma = np.eye(nb_sensors)

        self.weights = self.weight_function(nb_windows)

    def get_current_window(self, window_id: int) -> ProblemInputs:
        V_window = self.B_0 @ self.D1 + self.noise
        V_window = normalize(V_window)
        Y_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.D2 + V_window
        Y_window = normalize(Y_window)
        tro_inputs = ProblemInputs(
            fused_signals=[Y_window, V_window], fused_quadratics=[self.Gamma]
        )
        return tro_inputs

    def weight_function(self, nb_windows: int) -> np.ndarray:
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights


class CCADataRetriever(DataRetriever):
    def __init__(
        self,
        data_window_params: DataWindowParameters,
        nb_sensors: int,
        nb_sources: int,
        nb_windows: int,
        rng: np.random.Generator,
        lags: int = 5,
        signal_var: float = 0.5,
        noise_var: float = 0.1,
        mixture_var: float = 0.5,
        diff_var: float = 1,
    ) -> None:
        self.data_window_params = data_window_params
        self.nb_samples = data_window_params.window_length
        self.lags = lags
        self.D = rng.normal(
            loc=0,
            scale=np.sqrt(signal_var),
            size=(nb_sources, self.nb_samples + lags),
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
            size=(nb_sensors, self.nb_samples + lags),
        )

        self.weights = self.weight_function(nb_windows)

    def get_current_window(self, window_id: int) -> list[ProblemInputs]:
        total_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.D + self.noise
        Y_window = total_window[:, 0 : self.nb_samples]
        V_window = total_window[:, self.lags : None]
        V_window = normalize(V_window)
        Y_window = normalize(Y_window)
        cca_inputs_X = ProblemInputs(fused_signals=[Y_window])
        cca_inputs_W = ProblemInputs(fused_signals=[V_window])

        return [cca_inputs_X, cca_inputs_W]

    def weight_function(self, nb_windows: int) -> np.ndarray:
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights


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

    def get_current_window(self, window_id: int) -> ProblemInputs:
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

    def weight_function(self, nb_windows: int) -> np.ndarray:
        if nb_windows < 10:
            weights = np.zeros(nb_windows)
        else:
            segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
            segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
            segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)

            weights = np.concatenate([segment_1, segment_2, segment_3])
        return weights


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
