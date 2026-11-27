"""PA-FORGE: configurable 2D photoacoustic research software.

Experimental measured-data reconstruction is exploratory; generated outputs
are always synthetic and must not be misrepresented as measured acquisitions.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pa-forge")
except PackageNotFoundError:
    # This is a source checkout not installed as a Python distribution.
    __version__ = "0+uninstalled"
