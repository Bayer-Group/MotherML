"""
A description of the package
"""

from pathlib import Path

from single_source import VersionNotFoundError, get_version

# from mother.feature_generation.core import FeatureGenerator

path_to_pyproject_dir = Path(__file__).parent.parent
try:
    __version__ = get_version(__name__, path_to_pyproject_dir, fail=True)
except VersionNotFoundError:
    pass


class MotherModelException(Exception):
    pass


class MotherUnsupportedFeature(Exception):
    pass


class MotherPreprocessingException(Exception):
    pass
