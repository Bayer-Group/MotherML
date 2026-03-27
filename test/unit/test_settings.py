import typing
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mother.feature_generation import config as fg_conf
from mother.ml import config as ml_conf
from mother.preprocessing import config as prep_conv
from mother.settings import MotherSettings

project_dir: Path = Path(__file__).parent.parent.parent


@pytest.fixture
def fp_config() -> typing.Any:
    return yaml.safe_load("""
feature_generation:
  maccs: True
  fingerprints:
    - MorganFP:
        radius: 2
        fpSize: 1024
        include_chirality: False
    - AtomPairFP:
        fpSize: 256
  chemical_descriptors:
    descriptor_prefix: "rdkit_"
    omit_prefixes: ["fr_"]
    descriptor_list: ["MolWt","NumAromaticRings","NumHAcceptors","NumHDonors","NumHeteroatoms","NumRotatableBonds"]
""")


class TestFeatureGenSettings:
    def test_fp_config_from_dict(self, fp_config) -> None:
        settings: fg_conf.FeatureGenerationConfig = fg_conf.FeatureGenerationConfig(**fp_config["feature_generation"])
        assert isinstance(settings, fg_conf.FeatureGenerationConfig)

    def test_invalid_fp_type(self, fp_config) -> None:
        fp_settings: typing.Dict[str, typing.Any] = fp_config["feature_generation"]
        fp_settings["fingerprints"].append({"InvalidFPType": {"fpSize": 256}})
        with pytest.raises(ValidationError):
            fg_conf.FeatureGenerationConfig(**fp_settings)

    def test_fp_config(self) -> None:
        fp_settings: typing.Dict[str, typing.Any] = {
            "maccs": False,
            "fingerprints": [{"TopologicalTorsionFP": {"fpSize": 218}}],
        }
        settings: fg_conf.FeatureGenerationConfig = fg_conf.FeatureGenerationConfig(**fp_settings)
        assert isinstance(settings, fg_conf.FeatureGenerationConfig)


class TestPreprocessingSettings:
    def test_flags(self, settings) -> None:
        settings.preprocessing = prep_conv.PreprocessingConfig(flags=["STANDARDIZE"])
        assert settings.preprocessing.flags == ["STANDARDIZE"]

    def test_preprocessingSettings_raise(self) -> None:
        with pytest.raises(ValidationError):
            prep_conv.PreprocessingConfig(flags=["STANDARDIZATION"])


class TestModelConfig:
    def test_generic(self):
        ml_settings: ml_conf.ModelConfig = ml_conf.ModelConfig(
            target_type="single_target",
            model_type="regression",
            feature_selection_type="permutation",
            feature_selection_flags=["DROP_DUPLICATES"],  # type: ignore
            algorithm="catboost",
            parameters={"iterations": 100},
        )
        assert ml_settings is not None

    def test_catboost_model(self, catboost_settings) -> None:
        ml_settings: ml_conf.ModelConfig = ml_conf.ModelConfig(**catboost_settings)
        assert ml_settings is not None

    def test_lasso_model(self) -> None:
        ml_settings: ml_conf.ModelConfig = ml_conf.ModelConfig(
            target_type="single_target",
            model_type="classification_binary",
            feature_selection_type="permutation",
            feature_selection_flags=["DROP_DUPLICATES"],  # type: ignore
            algorithm="lasso",
            parameters={"C": 1.0, "solver": "liblinear", "max_iter": 1000},
        )
        assert ml_settings is not None

    def test_random_forest_model(self) -> None:
        ml_settings: ml_conf.ModelConfig = ml_conf.ModelConfig(
            target_type="single_target",
            model_type="regression",
            feature_selection_type="permutation",
            feature_selection_flags=["DROP_DUPLICATES"],  # type: ignore
            algorithm="random_forest",
            parameters={"n_estimators": 300, "max_depth": 10},
        )
        assert ml_settings is not None


class TestMotherSettings:
    def test_generic(self, settings: MotherSettings) -> None:
        assert isinstance(settings.feature_generation, fg_conf.FeatureGenerationConfig)
        assert isinstance(settings.preprocessing, prep_conv.PreprocessingConfig)

    def test_dump_to_yaml(self, tmp_path) -> None:
        # Create a sample MotherSettings object
        settings = MotherSettings.create()

        config_path: Path = Path(tmp_path).joinpath("mother_config.yaml")
        # Dump to a temporary file
        settings.dump_to_yaml(config_path)

        # Load the content back
        loaded_content = MotherSettings.load_from_yaml(config_path)

        # Verify the content is the same
        assert loaded_content == settings
