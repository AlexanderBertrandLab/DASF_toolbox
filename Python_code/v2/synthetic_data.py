import numpy as np
from problem_settings import ProblemInputs


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

    mmse_inputs = ProblemInputs(fused_data=[Y], global_constants=[D])

    return mmse_inputs
