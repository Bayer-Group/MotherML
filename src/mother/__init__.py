"""
A description of the package
"""

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "mother-ml"

try:
    # Installed distributions expose the version via metadata generated from
    # pyproject.toml at build time (bundled into the wheel's .dist-info).
    __version__ = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:
    # Fallback for running from an uninstalled source checkout: read the
    # version directly from pyproject.toml. Parsed with a regex to avoid a
    # dependency on tomllib (Python 3.11+) / tomli on Python 3.10.
    import re
    from pathlib import Path

    _pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    try:
        _match = re.search(
            r'^version\s*=\s*"([^"]+)"',
            _pyproject.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        __version__ = _match.group(1) if _match else "0.0.0"
    except OSError:
        __version__ = "0.0.0"


class MotherModelException(Exception):
    pass


class MotherUnsupportedFeature(Exception):
    pass


class MotherPreprocessingException(Exception):
    pass
