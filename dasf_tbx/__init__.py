from . import dasf
from . import data_retriever
from . import optimization_problems
from . import problem_settings
from . import utils

try:
    from importlib.metadata import version
except ImportError:
    from pkg_resources import get_distribution as version

try:
    __version__ = version("dasf_tbx")
except Exception:
    __version__ = "0.0.0"
