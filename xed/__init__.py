"""
Xed — Cardiff Introductory Data Science course utilities.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("xed")
except PackageNotFoundError:
    __version__ = "dev"
