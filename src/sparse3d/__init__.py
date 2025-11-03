# Standard library
from importlib.metadata import PackageNotFoundError, version  # noqa


def get_version():
    try:
        return version("sparse3d")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()

from .sparse3d import ROISparse3D, Sparse3D, stack  # noqa: F401, E402
