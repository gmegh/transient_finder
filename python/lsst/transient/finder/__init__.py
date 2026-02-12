# This package's namespace is intentionally kept empty, as users typically
# only need one or two Tasks from it.
from .find_transients import *
from .find_dark_sources import *
from .consolidate_dark_source_catalogs import *
from .plot_roundest_dark_sources import *
from .export_roundest_dark_sources import *
from .find_dark_rings_wavelet import *

try:
    from .version import *
except ImportError:
    __version__ = "?"
