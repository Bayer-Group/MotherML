import typing

import dotenv
from optuna.study import StudyDirection
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn import metrics as skl_metrics


class MotherTunerConfig(BaseSettings):
    scorer: typing.Optional[str] = Field(
        description="A scorer name returned by sklearn.metrics.get_scorer_names()",
    )
    direction: typing.Union[str, StudyDirection] = Field(
        default="maximize", description="Optimization direction: maximize or minimize"
    )
    early_stopping_optuna: bool = Field(
        default=False, description="Only set to True if torch is installed to stop tuning without improvement"
    )
    n_trials_optuna: int = Field(default=50, description="Number of trials for Optuna optimization")
    n_threads_optuna: int = Field(default=1, description="Number of threads for Optuna optimization")
    n_startup_trials: int = Field(
        default=12, description="The number of random sampling iterations before model-based optimization"
    )
    # target_type: props.TargetType = Field(description="Target type: single_target or multi_target")
    seed: int = Field(default=42, description="The random seed used for Optuna")

    @field_validator("scorer")
    def validate_scorer(v: str) -> str:
        if v.lower() not in skl_metrics.get_scorer_names():
            raise ValueError(f"Invalid value for scorer: {v}")
        return v.lower()

    @field_validator("early_stopping_optuna")
    def verify_torch_is_installed(cls, v: bool) -> bool:
        if v:
            try:
                import torch  # noqa: F401
            except ImportError:
                raise ImportError(
                    """Torch is required for early termination of Optuna optimization.
                    To enable early termination please install torch:
                    pip install mother[torch] or uv add mother[torch]"""
                )
        return v

    model_config = SettingsConfigDict(
        env_prefix="mother_preprocessing_",
        case_sensitive=False,
        use_enum_values=False,
        extra="ignore",  # set to forbid?!?
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
        str_to_upper=False,
    )
