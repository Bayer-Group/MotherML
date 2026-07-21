"""
A description of the package
"""

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "mother-ml"

try:
    # Installed distributions expose the version via metadata generated from
    # pyproject.toml at build time (bundled into the wheel's .dist-info).
    __version__: str = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:
    # Fallback for running from an uninstalled source checkout: read the
    # version directly from pyproject.toml.
    import tomllib
    from pathlib import Path

    _pyproject = Path(__file__).resolve().parent.parent.parent.joinpath("pyproject.toml")
    try:
        with _pyproject.open("rb") as _f:
            __version__ = tomllib.load(_f)["project"]["version"]
    except (OSError, KeyError):
        __version__ = "0.0.0"


class MotherModelException(Exception):
    pass


class MotherUnsupportedFeature(Exception):
    pass


class MotherPreprocessingException(Exception):
    pass
