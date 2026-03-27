import json
import logging
import typing
from pathlib import Path
from types import FunctionType

import dotenv
import yaml
from joblib import Memory
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mother.cv import config as cv_conf
from mother.feature_generation.config import FeatureGenerationConfig
from mother.ml import config as ml_conf
from mother.optimization import config as opt_conf
from mother.preprocessing.config import PreprocessingConfig

module_logger = logging.getLogger(__name__)

config_file: Path = Path(__file__).parent.joinpath("data", "mother_config.yaml")


def obj_to_importstr(obj: typing.Callable) -> str:
    """
    given a callable callable object this will return the
    import string to. From the string the object can be
    initiated again using importlib. This is useful for
    defining a function or class in a json serializable manner

    Args:
        obj: typing.Callable
    Returns:
        str: import string

    Example:
        >>> obj_from_importstr(pathlib.Path)
        'pathlib.Path'
    """
    try:
        mod = obj.__module__
    except Exception:
        raise ValueError(f"{str(obj)} doesnt have a __module__ attribute.")
    try:
        nm = obj.__name__
    except Exception:
        raise ValueError(f"{str(obj)} doesnt have a __name__ attribute. (might be a functool.partial?)")

    return mod + "." + nm


class InputConfig(BaseSettings):
    file: str = Field(description="Input file path")
    separator: str = Field(default=",", description="Input file field separator")
    structure_col: str = Field(description="Column name that contains the molecule structure")
    target_columns: typing.List[str] = Field(description="Column names that contain the target values")
    group_col: typing.Optional[str] = Field(
        default=None,
        description="""Column name that should contain the groups (clusters) or should be used to generate groups
        based on CV settings""",
    )


class PipelineConfig(BaseSettings):
    memory: typing.Optional[typing.Union[str, Memory]] = Field(default=None, description="Use memory for caching")
    verbose: bool = Field(default=False, description="Verbose output")
    transform: typing.Literal["pandas", "default"] = Field(default="pandas", description="Output format")
    n_jobs: typing.Optional[int] = Field(default=None, ge=1, description="Number of jobs to run in parallel")
    remainder: typing.Literal["drop", "passthrough"] = Field(
        default="passthrough", description="Remainder handling of the pipeline."
    )
    verbose_feature_names_out: bool = Field(
        default=False, description="Verbose feature names output for ColumnTransformer"
    )


class MotherSettings(BaseSettings):
    input: InputConfig = Field(description="General input settings to read data")
    pipeline: PipelineConfig = Field(description="SciKit learn Pipeline settings")

    # usable for loss function?!
    # loss_function: ImportString[Callable[[Any], Any]] = "math.cos"

    # load preprocessing config
    preprocessing: PreprocessingConfig = Field(description="Preprocessing settings")

    # load feature generation settings
    feature_generation: FeatureGenerationConfig = Field(description="Feature generation settings")

    # load CV Settings
    cv: typing.Optional[cv_conf.CVSettings] = Field(default=None, description="Cross-validation settings")

    # load model Settings
    model: ml_conf.ModelConfig = Field(description="Model settings")

    # load hyperparameter tuning settings
    tuning: typing.Optional[opt_conf.MotherTunerConfig] = Field(
        default=None, description="Hyperparameter tuning settings"
    )

    model_config = SettingsConfigDict(
        env_prefix="mother_",
        case_sensitive=False,
        use_enum_values=False,
        extra="ignore",  # set to forbid?!?
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
        json_encoders={
            FunctionType: lambda v: obj_to_importstr(v),
        },
    )

    @classmethod
    def create(cls, file_path: typing.Optional[typing.Union[Path, str]] = None) -> "MotherSettings":
        """
        Create a MotherSettings instance from a default YAML file and writes to given path.

        Args:
            file_path (typing.Optional[typing.Union[Path, str]], optional): The path to the YAML file. Defaults to None.

        Returns:
            MotherSettings: An instance of MotherSettings with defaults.

        Logs:
            Logs the creation of the YAML file if `file_path` is provided.
        """
        ms: MotherSettings = cls.load_from_yaml(file_path=config_file)
        if file_path:
            module_logger.info("Creating yaml file at %s", file_path)
            ms.dump_to_yaml(file_path=file_path)
        return ms

    @classmethod
    def load_from_yaml(cls, file_path: typing.Union[Path, str]) -> "MotherSettings":
        """
        Load MotherSettings from a YAML file.

        Args:
            file_path (typing.Union[Path, str]): The path to the YAML configuration file. Defaults to config_file.

        Returns:
            MotherSettings: An instance of MotherSettings populated with data from the YAML file.

        Raises:
            ValueError: If the provided file does not exist.
        """
        path: Path = Path(file_path)  # Ensure file_path is a Path object
        if not path.exists():
            raise ValueError(f"Provided file '{path}' does not exist")
        module_logger.debug("Loading MotherSettings from %s", path)
        with open(path, "r") as content:
            return MotherSettings(**yaml.safe_load(content))

    def dump_to_yaml(self, file_path: typing.Union[Path, str]) -> None:
        """
        Dumps the current model to a YAML file at the specified path.

        Args:
            file_path (Path): The file path where the YAML content will be saved.

        Raises:
            OSError: If the file cannot be created or written to.

        Notes:
            - If the parent directory of the specified path does not exist, it will be created.
            - If a file already exists at the specified path, it will be overwritten.
        """
        path = Path(file_path)  # Ensure path is a Path object
        if not path.parent.exists():
            module_logger.debug("Creating parent directory %s", path.parent)
            path.parent.mkdir(parents=True)
        if path.exists():
            module_logger.warning("Overwriting existing file %s", path)
        with open(path, "w") as content:
            yaml.dump(self.model_dump(mode="json"), content)

    def __str__(self) -> str:
        return json.dumps(self.model_dump(), indent=4)

    def __repr__(self) -> str:
        return self.__str__()
