import numpy as np
from dasftoolbox.problem_settings import ProblemInputs
from dasftoolbox.utils import normalize

from dasftoolbox.data_retrievers.data_retriever import (
    DataRetriever,
    DataWindowParameters,
)


class GEVDDataRetriever(DataRetriever):
    """
    GEVD data retriever class.

    Simulates a setting where noisy mixture of sources are observed by the nodes of the network.

    Formally, the signals generated are given by :math:`\mathbf{v}(t)=B\cdot\mathbf{d}_1(t)+\mathbf{n}(t)` and :math:`\mathbf{y}(t)=A(t)\cdot\mathbf{d}_2(t)+\mathbf{v}(t)`, where :math:`\mathbf{d}_1\in\mathbb{R}^L` and `math:`\mathbf{d}_2\in\mathbb{R}^{P-L}` correspond to the source signals, :math:`\mathbf{n}\in\mathbb{R}^M` to the noise and :math:`A\in\mathbb{R}^{M\times L}` and :math:`B\in\mathbb{R}^{M\times (P-L)}` to the mixture matrices. The non-stationarity of :math:`\mathbf{y}` follows from the dependence of :math:`A` on time, where :math:`A(t)=A_0+\Delta\cdot w(t)`, with :math:`w` representing a weight function varying in time.

    The signals :math:`\mathbf{y}` and :math:`\mathbf{v}` are normalized to have unit norm and zero mean.

    Attributes
    ----------
    data_window_params : DataWindowParameters
        Class instance storing the parameters that define a window of data.
    nb_sensors : int
        Number of sensors in the network. Equals to :math:`M`, the dimension of :math:`\mathbf{y}` and :math:`\mathbf{v}`.
    nb_sources : int
        Number of sources. Represents the number of true number of sources that generate the data. Equals to :math:`L`, the dimension of :math:`\mathbf{d}_1`.
    nb_windows : int
        Number of windows of data.
    rng : np.random.Generator
        Random number generator for reproducibility.
    latent_dim : int | None
        Latent dimension :math:`P` of the problem. If None, will be fixed to :math:`2L`. By default None.
    signal_var : float
        Variance of the signals of interest, i.e., :math:`\mathbf{d}_1` and :math:`\mathbf{d}_2`. By default 0.5.
    noise_var : float
        Variance of the noise, i.e., :math:`\mathbf{n}`. By default 0.1.
    mixture_var : float
        Variance of the elements of mixture matrices :math:`A_0` and :math:`B`. By default 0.5.
    diff_var : float
        Norm of :math:`\Delta`. By default 1.
    """

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

    def get_data_window(self, window_id: int) -> ProblemInputs:
        V_window = self.B_0 @ self.D1 + self.noise
        V_window = normalize(V_window)
        Y_window = (
            self.A_0 + self.Delta * self.weights[window_id]
        ) @ self.D2 + V_window
        Y_window = normalize(Y_window)
        gevd_inputs = ProblemInputs(fused_signals=[Y_window, V_window])
        return gevd_inputs

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
