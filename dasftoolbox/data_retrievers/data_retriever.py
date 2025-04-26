from dasftoolbox.problem_settings import ProblemInputs
from abc import abstractmethod
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
    """Get the parameters to simulate a stationary setting by setting the sliding window offset to 0."""
    return DataWindowParameters(
        window_length=window_length, nb_window_reuse=iterations, sliding_window_offset=0
    )


class DataRetriever:
    def __init__(
        self, data_window_params: DataWindowParameters, *args, **kwargs
    ) -> None:
        return None

    @abstractmethod
    def get_data_window(self, window_id: int) -> ProblemInputs | list[ProblemInputs]:
        """Get the window of data for the specified window ID."""
        pass
