from dasftoolbox.problem_settings import ProblemInputs
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataWindowParameters:
    """
    Class storing the parameters defining a window of data.

    Attributes
    ----------
    window_length : int
        Length of the window of data.
    nb_window_reuse : int
        Number of times each window of data is reused.
    sliding_window_offset : int | None
        Offset of the sliding window. If None, it is set to the window length.
    """

    window_length: int
    nb_window_reuse: int = 1
    sliding_window_offset: int | None = None

    def __post_init__(self) -> None:
        if self.sliding_window_offset is None:
            self.sliding_window_offset = self.window_length

    def get_window_sample_interval(self, window_id: int) -> Tuple[int, int]:
        start = window_id * self.sliding_window_offset
        return start, start + self.window_length


def get_stationary_setting(window_length: int, iterations: int) -> DataWindowParameters:
    """
    Get the parameters to simulate a stationary setting by setting the sliding window offset to 0.

    Parameters
    ----------
    window_length : int
        Length of the window of data.
    iterations : int
        Number of iterations desired for the simulation.
    Returns
    -------
    DataWindowParameters
        Parameters to simulate a stationary setting.
    """
    return DataWindowParameters(
        window_length=window_length, nb_window_reuse=iterations, sliding_window_offset=0
    )


class DataRetriever:
    """
    Base class for data retrievers in the DASF format.

    Attributes
    ----------
    data_window_params : DataWindowParameters
        Parameters defining the window of data.
    """

    def __init__(
        self, data_window_params: DataWindowParameters, *args, **kwargs
    ) -> None:
        return None

    @abstractmethod
    def get_data_window(self, window_id: int) -> ProblemInputs | list[ProblemInputs]:
        """
        Get the window of data for the specified window ID.

        Parameters
        ----------
        window_id : int
            ID of the window of data to retrieve.
        Returns
        -------
        ProblemInputs | list[ProblemInputs]
            Window of data for the specified window ID.
        """
        pass
