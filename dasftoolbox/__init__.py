from . import dasf
from .data_retrievers import data_retriever
from .optimization_problems import optimization_problem
from . import problem_settings
from . import utils
from .dasf import (
    DASF as DASF,
    DASFMultiVar as DASFMultiVar,
    DynamicPlotParameters as DynamicPlotParameters,
)
from .problem_settings import (
    ProblemInputs as ProblemInputs,
    ConvergenceParameters as ConvergenceParameters,
    NetworkGraph as NetworkGraph,
)
from .data_retrievers.data_retriever import (
    DataWindowParameters as DataWindowParameters,
    DataRetriever as DataRetriever,
    get_stationary_setting as get_stationary_setting,
)
from .optimization_problems.optimization_problem import (
    OptimizationProblem as OptimizationProblem,
)


try:
    from importlib.metadata import version
except ImportError:
    from pkg_resources import get_distribution as version

try:
    __version__ = version("dasftoolbox")
except Exception:
    __version__ = "0.0.0"
