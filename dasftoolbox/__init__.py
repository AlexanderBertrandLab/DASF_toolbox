from . import dasf
from .data_retrievers import data_retriever
from .optimization_problems import optimization_problem
from . import problem_settings
from . import utils

try:
    from importlib.metadata import version
except ImportError:
    from pkg_resources import get_distribution as version

try:
    __version__ = version("dasftoolbox")
except Exception:
    __version__ = "0.0.0"
