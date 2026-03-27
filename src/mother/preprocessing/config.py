import dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mother.preprocessing.utils import StandardizationFlag


class PreprocessingConfig(BaseSettings):
    flags: list[str] = Field(
        default=["STANDARDIZE", "NEUTRALIZE", "DESALT"],
        description="The standardization flags to be used for preprocessing.",
    )

    @field_validator("flags")
    @classmethod
    def flag_only_allows_certain_string(cls, v: list[str]) -> list[str]:  # pylint: disable=no-self-argument
        """
        Validate that the flags only contain certain characters.

        Args:
            v (list[str]): The input flags.

        Returns:
            list[str]: The validated flags.

        Raises:
            ValueError: If an unsupported flag is found.
        """
        for key in v:
            if key not in [str(v) for v in StandardizationFlag]:
                raise ValueError(f"'{key}' not supported as flag")
        return v

    model_config = SettingsConfigDict(
        env_prefix="mother_preprocessing_",
        case_sensitive=False,
        use_enum_values=False,
        extra="ignore",
        env_file=dotenv.find_dotenv(filename=".env"),
        validate_assignment=True,
        env_file_encoding="utf-8",
        str_to_upper=True,
    )
