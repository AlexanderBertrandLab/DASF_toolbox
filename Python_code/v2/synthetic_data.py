import numpy as np
from problem_settings import ProblemInputs
from utils import normalize


def mmse_generate_synthetic_inputs(
    nb_samples: int,
    nb_sensors: int,
    rng: np.random.Generator,
    nb_sources: int,
    signal_var: float = 0.5,
    noise_var: float = 0.1,
    mixture_var: float = 0.5,
) -> ProblemInputs:
    """Generate synthetic inputs for the MMSE problem."""

    D = rng.normal(loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples))
    A = rng.normal(loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sources))
    noise = rng.normal(loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples))

    Y = A @ D + noise
    Y = normalize(Y)

    mmse_inputs = ProblemInputs(fused_data=[Y], global_parameters=[D])

    return mmse_inputs


def mmse_generate_non_stationary_inputs(
    nb_samples_per_window: int,
    nb_sensors: int,
    rng: np.random.Generator,
    nb_sources: int,
    nb_windows: int,
    signal_var: float = 0.5,
    noise_var: float = 0.1,
    mixture_var: float = 0.5,
    diff_var: float = 0.01,
) -> ProblemInputs:
    D = rng.normal(
        loc=0, scale=np.sqrt(signal_var), size=(nb_sources, nb_samples_per_window)
    )
    A_0 = rng.normal(loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sources))
    Delta = rng.normal(loc=0, scale=np.sqrt(mixture_var), size=(nb_sensors, nb_sources))
    Delta = Delta * np.linalg.norm(A_0, "fro") * diff_var / np.linalg.norm(Delta, "fro")
    noise = rng.normal(
        loc=0, scale=np.sqrt(noise_var), size=(nb_sensors, nb_samples_per_window)
    )

    Y = normalize(A_0 @ D + noise)
    weights = weight_function(nb_windows)
    for window in range(1, nb_windows):
        Y_window = (A_0 + Delta * weights[window]) @ D + noise
        Y_window = normalize(Y_window)
        Y = np.concatenate((Y, Y_window), axis=1)

    mmse_inputs = ProblemInputs(fused_data=[Y], global_parameters=[D])

    return mmse_inputs


def weight_function(nb_windows):
    segment_1 = np.linspace(0, 1, int(5 * nb_windows / 10), endpoint=False)
    segment_2 = np.linspace(0, 1, int(3 * nb_windows / 10), endpoint=False)
    segment_3 = np.linspace(0, 1, int(2 * nb_windows / 10), endpoint=False)
    segment_4 = np.linspace(0, 1, int(1 * nb_windows / 10), endpoint=False)

    weights = np.concatenate([segment_1, segment_2, segment_3, segment_4])
    return weights
