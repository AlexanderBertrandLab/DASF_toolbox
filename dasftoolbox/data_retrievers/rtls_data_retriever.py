import numpy as np

from dasftoolbox.data_retrievers.data_retriever import (
    DataRetriever,
    DataWindowParameters,
)
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.utils import normalize


class RTLSDataRetriever(DataRetriever):
    """
    RTLS data retriever class.

    Simulates a setting where noisy mixture of sources are observed by the nodes of the network.

    Formally, the signals generated are given by :math:`\mathbf{y}(t)=A(t)\cdot\mathbf{s}(t)+\mathbf{n}(t)`, where :math:`\mathbf{s}\in\mathbb{R}^L` corresponds to the source signal, :math:`\mathbf{n}\in\mathbb{R}^M` to the noise and :math:`A\in\mathbb{R}^{M\times L}` to the mixture matrix. The non-stationarity of :math:`\mathbf{y}` follows from the dependence of :math:`A` on time, where :math:`A(t)=A_0+\Delta\cdot w(t)`, with :math:`w` representing a weight function varying in time.

    Additionally, the target signal :math:`\mathbf{d}\in\mathbb{R}^L` is also noisy, where :math:`\mathbf{d}(t)=\mathbf{s}(t)+\mathbf{n}_d(t)`.

    The signal :math:`\mathbf{y}` is normalized to have unit norm and zero mean.

    The constants of the problem are chosen as :math:`\delta=1` and :math:`L\in\mahbb{R}^{L\\times L}` is a diagonal matrix with diagonal entries following a Gaussian distribution centered around 1.


    Attributes
    ----------
    data_window_params : DataWindowParameters
        Class instance storing the parameters that define a window of data.
    nb_sources : int
        Number of sources. Represents the number of true number of sources that generate the data. Equals to :math:`L`, the dimension of :math:`\mathbf{d}`. By default 10.
    nb_sensors : int
        Number of sensors in the network. Equals to :math:`M`, the dimension of :math:`\mathbf{y}` and :math:`\mathbf{v}`.
    nb_windows : int
        Number of windows of data.
    rng : np.random.Generator
        Random number generator for reproducibility.
    signal_var : float
        Variance of the signals of interest, i.e., :math:`\mathbf{d}`. By default 1.
    noise_var : float
        Variance of the noise, i.e., :math:`\mathbf{n}`. By default 0.1.
    noise_var_D : float
        Variance of the noise on :math:`\mathbf{d}`, i.e., :math:`\mathbf{n}`. By default 0.02.
    mixture_var : float
        Variance of the elements of mixture matrix :math:`A_0`. By default 0.5.
    diff_var : float
        Norm of :math:`\Delta`. By default 0.1.
    """

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

    def get_data_window(self, window_id: int) -> ProblemInputs:
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
