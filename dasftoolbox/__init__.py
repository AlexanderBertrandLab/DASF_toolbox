from . import dasf, problem_settings, utils
from .dasf import (
    DASF as DASF,
)
from .dasf import (
    DASFMultiVar as DASFMultiVar,
)
from .dasf import (
    DynamicPlotParameters as DynamicPlotParameters,
)
from .data_retrievers import data_retriever
from .data_retrievers.data_retriever import (
    DataRetriever as DataRetriever,
)
from .data_retrievers.data_retriever import (
    DataWindowParameters as DataWindowParameters,
)
from .data_retrievers.data_retriever import (
    get_stationary_setting as get_stationary_setting,
)
from .optimization_problems import optimization_problem
from .optimization_problems.optimization_problem import ConstraintType as ConstraintType
from .optimization_problems.optimization_problem import (
    OptimizationProblem as OptimizationProblem,
)
from .problem_settings import (
    ConvergenceParameters as ConvergenceParameters,
)
from .problem_settings import (
    NetworkGraph as NetworkGraph,
)
from .problem_settings import (
    ProblemInputs as ProblemInputs,
)

try:
    from importlib.metadata import version
except ImportError:
    from pkg_resources import get_distribution as version

try:
    __version__ = version("dasftoolbox")
except Exception:
    __version__ = "0.0.0"
