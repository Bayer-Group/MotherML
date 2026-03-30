import itertools
import typing
from functools import wraps
from pathlib import Path
from typing import Callable, Type

import numpy as np
import pandas as pd
import pytest
from rdkit.Chem import MolFromSmiles
from sklearn import get_config as skl_get_config
from sklearn import set_config as skl_set_config
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

import mother.cv as cv_module
import mother.feature_generation as fg
import mother.ml as ml
import mother.ml.properties as props
from mother import pipeline_utils
from mother.feature_generation import config as fg_config
from mother.feature_generation.config import (
    ChemicalDescriptorsParams,
    FeatureGenerationConfig,
    FingerprintParams,
    MaccsFingerprintsParams,
)
from mother.ml import PipelineWithHyperparameterRooting
from mother.ml.models.m_catboost import (
    CatboostClassifierMother,
    CatboostRankerMother,
    CatboostRegressorMother,
)
from mother.optimization.core import MotherTuner
from mother.preprocessing.config import PreprocessingConfig
from mother.preprocessing.core import SmilesToMolTransformer, StandardizerTransformer
from mother.settings import MotherSettings

project_dir = Path(__file__).parent.parent.parent


@pytest.fixture
def preserve_metadata_routing():
    """
    Fixture to preserve and restore the metadata routing configuration.

    This fixture saves the current state of sklearn's metadata routing configuration
    before the test runs and restores it after the test completes, ensuring that
    tests don't affect each other's metadata routing state.
    """
    # Save the current metadata routing state
    original_state: bool = bool(skl_get_config().get("enable_metadata_routing", False))

    yield

    # Restore the original metadata routing state after the test
    skl_set_config(enable_metadata_routing=original_state)


@pytest.fixture
def settings() -> MotherSettings:
    settings = MotherSettings.load_from_yaml(project_dir.joinpath("src", "mother", "data", "mother_config.yaml"))
    return settings


@pytest.fixture
def catboost_settings(settings: MotherSettings) -> typing.Any:
    return settings.model.model_dump()


@pytest.fixture
def input_file(settings: MotherSettings) -> Path:
    input_file = project_dir.joinpath(settings.input.file)
    assert input_file.exists()
    return input_file


@pytest.fixture
def data(input_file: Path, settings: MotherSettings) -> pd.DataFrame:
    data = pd.read_csv(input_file, sep=settings.input.separator)
    assert len(data) > 0
    return data.head(10)


@pytest.fixture
def preprocessor(settings: MotherSettings) -> BaseEstimator:
    prep_config: PreprocessingConfig = settings.preprocessing
    return Pipeline(
        [
            ("smiles_standardizer", StandardizerTransformer(flags=prep_config.flags)),
            ("smiles_to_mol", SmilesToMolTransformer(molecule_col="molecule")),
        ],
        memory=None,
    ).set_output(transform="pandas")


@pytest.fixture
def mol_data(preprocessor, data: pd.DataFrame, settings: MotherSettings) -> pd.DataFrame:
    structure_data: pd.Series = data[settings.input.structure_col]
    return preprocessor.fit_transform(structure_data)


@pytest.fixture
def feature_generator(settings: MotherSettings) -> BaseEstimator:
    fg_conf: FeatureGenerationConfig = settings.feature_generation

    transformer_list = []
    if fg_conf.maccs:
        maccs_params: MaccsFingerprintsParams = MaccsFingerprintsParams()
        transformer_list.append(("Maccs", fg.MaccsFingerprints(**maccs_params.model_dump())))
    if fg_conf.chemical_descriptors:
        physchem_params: ChemicalDescriptorsParams = ChemicalDescriptorsParams(
            **fg_conf.chemical_descriptors.model_dump()
        )
        transformer_list.append(("Desc", fg.ChemicalDescriptors(**physchem_params.model_dump())))
    for fp in fg_conf.fingerprints:
        fp_type: str = next(iter(fp.keys()))
        params_class: Type[FingerprintParams] = fg_config.get_params_for_fp_type(fp_type)
        params: FingerprintParams = params_class(**fp[fp_type])
        transformer_list.append((fp_type, fg.FingerprintsGeneric(fp_type=fp_type, parameters=params.model_dump())))
    return FeatureUnion(transformer_list=transformer_list).set_output(transform="pandas")


@pytest.fixture
def features(feature_generator: BaseEstimator, mol_data: pd.DataFrame, settings: MotherSettings) -> pd.DataFrame:
    features = feature_generator.fit_transform(mol_data["molecule"])  # type: ignore
    assert features.filter(like="Morgan__", axis=1).empty
    assert len(features.filter(like="AtomPair__", axis=1)) > 0
    assert len(features.filter(like="Maccs__", axis=1)) > 0
    assert len(features.filter(like="Desc__", axis=1)) > 0
    return features


@pytest.fixture
def groups(settings: MotherSettings, mol_data) -> pd.DataFrame:
    assert settings.cv
    cv_conf: cv_module.GenericCVModel = settings.cv.get_cv_settings()
    groups: pd.DataFrame
    if settings.cv.cv_type == cv_module.CVtype.TANIMOTO_GROUPING:
        groups_engine: BaseEstimator = cv_module.TanimotoGroupingFromMols(**cv_conf.model_dump())
        groups = groups_engine.set_output(transform="pandas").fit_transform(mol_data["molecule"])  # type: ignore
    elif settings.cv.cv_type == cv_module.CVtype.TIME_SERIES:
        ...
    assert len(groups) == len(mol_data)
    assert isinstance(groups, pd.DataFrame)
    return groups


@pytest.fixture
def model(settings: MotherSettings, cv: RepeatedKFold) -> BaseEstimator:
    assert settings.model.model_type == "regression" and settings
    return ml.PipelineWithHyperparameterRooting(
        [
            ("feature_selector", pipeline_utils.get_feature_selection_pipeline(settings=settings, cv=cv)),
            (
                "ml_model",
                ml.CatboostRegressorMother(target_type=settings.model.target_type, **settings.model.parameters),
            ),
        ]
    ).set_output(transform="pandas")


@pytest.fixture(params=[True, False])
def cv_strategy(request):
    if request.param:
        feature_selection_cv = KFold(
            n_splits=2,
            shuffle=True,
            random_state=42,
        )
        return feature_selection_cv
    else:
        return None


@pytest.fixture
def cv() -> RepeatedKFold:
    return RepeatedKFold(n_splits=2, n_repeats=1, random_state=42)


@pytest.fixture()
def cv_cross_validation(request) -> KFold:
    use_groups = getattr(request, "param", False)
    return GroupKFold(n_splits=2) if use_groups else KFold(n_splits=2)


@pytest.fixture
def tuner(settings: MotherSettings) -> MotherTuner:
    assert settings.tuning
    return MotherTuner(**settings.tuning.model_dump())


#####################################################################


@pytest.fixture(params=["default", "pandas", None])  # skip "polars"
def transform_result(request) -> list[str]:
    return request.param


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame(index=range(100), columns=range(1000))


def mock_compounds_df_from_smiles(
    smiles: typing.List[str],
) -> pd.DataFrame:
    ids = [f"BCS_{i}" for i in range(len(smiles))]

    compounds = pd.DataFrame(
        {
            "ID": ids,
            "Smiles": smiles,
        }
    )

    return compounds


@pytest.fixture
def invalid_compounds() -> pd.DataFrame:
    invalid_smiles = [
        r"CHC",  # Invalid atom label
        r"C[H]C",  # Nonstandard atom label escaped but hydrogen
        r"Cc1ccc2c(c1)-n1-c(=O)/c=c\c(=O)-n-2-c2cc(C)ccc2-1",  # rdkit can't kekulize mol and returns None
    ]
    return mock_compounds_df_from_smiles(invalid_smiles)


@pytest.fixture
def mixed_compounds() -> pd.DataFrame:
    mixed_smiles = [
        r"N#N",
        r"CHC",  # Invalid atom label
        r"",  # empty molecule (skipped and results in missing data frame index)
        r"CN=C=O",
        r"C[H]C",  # Nonstandard atom label escaped but hydrogen
        r"O=S(=O)([O-])[O-].[Cu+2]",  # Copper Sulfate
        r"COc1cc(C=O)ccc1O",  # Vanillin
    ]
    return mock_compounds_df_from_smiles(mixed_smiles)


@pytest.fixture
def valid_smiles() -> list[str]:
    return [
        r"N#N",
        r"CN=C=O",
        r"O=S(=O)([O-])[O-].[Cu+2]",  # Copper Sulfate
        r"COc1cc(C=O)ccc1O",  # Vanillin
        r"COc1ccc2[nH]cc(CCNC(C)=O)c2c1",  # Melatonin
        r"CCc1ccc2c3[nH]c4ccccc4c3cc[n+]2c1.[Cl-]",  # Flavopereirin
        r"CN1CCC[C@H]1c1cccnc1",  # Nicotine
        r"CCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO",  # Oenanthotoxin
        r"C=C/C=C\CC1=C(C)[C@@H](OC(=O)[C@@H]2[C@@H](/C=C(\C)C(=O)OC)C2(C)C)CC1=O",  # Pyrethrin
        r"COc1cc2c(c3oc(=O)c4c(c13)CCC4=O)[C@@H]1C=CO[C@@H]1O2",  # Aflatoxin B1
    ]


@pytest.fixture
def invalid_smiles() -> list[str]:
    return [
        r"COO",
        r"CHC",  # Invalid atom label
        r"C[H]C",  # Nonstandard atom label escaped but hydrogen
        r"Cc1ccc2c(c1)-n1-c(=O)/c=c\c(=O)-n-2-c2cc(C)ccc2-1",  # rdkit can't kekulize mol and returns None
    ]


@pytest.fixture
def missing_smiles() -> list[typing.Optional[str]]:
    return [
        r"CCO",
        None,
        r"CCC",
    ]


@pytest.fixture
def mols():
    compounds = [
        "O",  # Water (H2O)
        "CCO",  # Ethanol (C2H5OH)
        "CC(=O)O",  # Acetic Acid (C2H4O2)
        "C(C1C(C(C(C(O1)O)O)O)O)O)O",  # Glucose (C6H12O6)
        "c1ccccc1",  # Benzene (C6H6)
        "CC(=O)Oc1ccccc(O)c1C(=O)O",  # Aspirin (C9H8O4)
        "CN1C=NC2=C(N1C(=O)N(C(=O)N2C)C)C(=O)N(C)C",  # Caffeine (C8H10N4O2)
        "CC(C)C(C(C)C(C(C)C)C)C(C)C(C)C(C)C(C)C(C)(C)C(C)C",  # Cholesterol (C27H46O)
        "CC(C)CC1=CC=C(C=C1)C(=O)O",  # Ibuprofen (C13H18O2)
        "CC(C1=CC2=C(C=C1)C(=CN2)C(=O)O)O",  # Serotonin (C10H12N2O)
    ]
    mols = [MolFromSmiles(compound) for compound in compounds]
    return np.array(mols)


@pytest.fixture
def valid_compounds(valid_smiles) -> pd.DataFrame:
    return mock_compounds_df_from_smiles(valid_smiles)


@pytest.fixture
def isotope_smiles() -> list:
    n_atoms = 5
    smiles = []
    for permutation in itertools.permutations(range(10, 10 + n_atoms + 1), n_atoms):
        smiles.append("C" + "".join([f"[{i}CH2]" for i in permutation]) + "C")

    return smiles


@pytest.fixture
def isotope_compounds(isotope_smiles) -> pd.DataFrame:
    return mock_compounds_df_from_smiles(isotope_smiles)


@pytest.fixture
def freesolv_csv_compounds_missing(freesolv_csv_compounds) -> pd.DataFrame:
    return pd.concat(
        [
            freesolv_csv_compounds,
            pd.DataFrame(
                {
                    "iupac": ["emptyData", "emptyBoth", "emptySmiles"],
                    "smiles": ["CCC=O", "CC=O", None],
                    "expt": [None, None, -4],
                    "calc": [0.8, None, -3.9],
                }
            ),
        ]
    )


@pytest.fixture()
def freesolv_binary_classification_compounds(freesolv_csv_compounds) -> pd.DataFrame:
    compounds = freesolv_csv_compounds.copy()
    for target in ["calc", "expt"]:
        compounds[f"{target}_class"] = 1
        compounds.loc[compounds[target] <= 0, f"{target}_class"] = 0
    return compounds


@pytest.fixture()
def freesolv_classification_compounds(freesolv_csv_compounds) -> pd.DataFrame:
    compounds = freesolv_csv_compounds.copy()
    for target in ["calc", "expt"]:
        quantiles = np.quantile(compounds[target], [0.33, 0.66])
        class_1 = (compounds[target] >= quantiles[0]) & (compounds[target] < quantiles[1])
        class_2 = compounds[target] >= quantiles[1]

        col_name: str = f"{target}_class"
        compounds[col_name] = 0
        compounds.loc[class_1, col_name] = 1
        compounds.loc[class_2, col_name] = 2
    return compounds


@pytest.fixture(
    params=[
        "single-target-regression",
        "multi-target-regression",
        "binary-classification",
        "multi-class-classification",
        "single-target-ranking",
    ]
)
def output_type(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["with_categorical", "without_categorical"])
def categoric_feature(request):
    return request.param


def ensure_metadata_routing(func: Callable) -> Callable:
    """
    Decorator to ensure metadata routing is enabled before executing a function.

    This decorator checks if sklearn's metadata routing is enabled and activates it
    if necessary. It's particularly useful for initializing ranking models that require
    metadata routing for passing additional parameters like group_id.

    Parameters
    ----------
    func : Callable
        The function to be decorated (typically __init__ of a ranking model)

    Returns
    -------
    Callable
        The wrapped function with metadata routing ensured
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "ranking" in kwargs.get("output_type", ""):
            use_metadata_routing: bool = bool(skl_get_config().get("enable_metadata_routing", False))
            if not use_metadata_routing:
                skl_set_config(enable_metadata_routing=True)  # NOSONAR
        return func(*args, **kwargs)

    return wrapper


@pytest.fixture()
@ensure_metadata_routing
def modeling_data(
    categoric_feature: bool, output_type: str, preserve_metadata_routing
) -> typing.Tuple[pd.DataFrame, str, pd.DataFrame, list, np.ndarray, np.ndarray]:
    np.random.seed(42)

    features = pd.DataFrame({f"feature_{i}": np.random.rand(300) for i in range(1, 30)})

    target = features.values.sum(axis=1) / 30 + np.random.normal(
        0,
        0.1,
        300,
    )
    target = pd.DataFrame(target, columns=["target"])

    # add a ranking group column, and sort by that (needs to be in order for ranking)
    features["ranking_groups"] = np.random.randint(0, 10, size=300)
    features = features.sort_values(by=["ranking_groups"], ascending=True)
    ranking_groups = pd.Categorical(features["ranking_groups"]).codes
    features = features.drop("ranking_groups", axis=1)

    multi_target = output_type == "multi-target-regression"
    if multi_target:
        target = pd.concat([target, target], axis=1)
        target.columns = [f"target_{i}" for i in range(len(target.columns))]

    categorical_features = []
    if categoric_feature:
        mask = features["feature_1"] < 0.5
        features["feature_1"] = features["feature_1"].astype(str)
        features.loc[mask, "feature_1"] = "a"
        features.loc[~mask, "feature_1"] = "b"
        categorical_features.append("feature_1")

    categoric_output = "classification" in output_type
    if categoric_output:
        target_categorical = pd.DataFrame(index=target.index, columns=target.columns)
        mask = target < 0.5
        target_categorical[mask] = "foo"
        target_categorical[~mask] = "bar"
        if "multi-class" in output_type:
            mask2 = target > 0.65
            target_categorical[mask2] = "baz"

        target = target_categorical

    # cv-groups
    cv_groups = np.random.randint(5, size=(len(target),))

    return features, output_type, target, categorical_features, ranking_groups, cv_groups


@pytest.fixture()
def ml_model(
    modeling_data,
) -> CatboostRegressorMother | CatboostClassifierMother | CatboostRankerMother:
    _, output_type, target, categorical_features, ranking_groups, _ = modeling_data

    target_type: props.TargetType = "single_target" if target.shape[1] == 1 else "multi_target"

    if output_type == "binary-classification":
        return CatboostClassifierMother(
            target_type=target_type, cat_features=categorical_features, model_type="classification_binary", num_trees=10
        )
    elif output_type == "multi-class-classification":
        return CatboostClassifierMother(
            target_type=target_type,
            cat_features=categorical_features,
            model_type="classification_multiclass",
            num_trees=10,
        )
    elif "ranking" in output_type:
        return CatboostRankerMother(
            target_type=target_type,
            cat_features=categorical_features,
            num_trees=10,
        ).set_fit_request(group_id="group_id")
    else:
        return CatboostRegressorMother(target_type=target_type, cat_features=categorical_features, num_trees=10)


@pytest.fixture()
def feature_selector(
    modeling_data: typing.Tuple[pd.DataFrame, str, pd.DataFrame, list], cv: RepeatedKFold
) -> ColumnTransformer:
    _, output_type, _, categorical_features, ranking_groups, _ = modeling_data
    model_settings = {
        "feature_selection_flags": ["DROP_CORRELATED", "DROP_CONSTANT", "DROP_DUPLICATES", "DROP_UNIMPORTANT"],
        "categorical_features": categorical_features,
        "feature_selection_threshold": 0,
        "correlation_threshold": 0.9,
        "algorithm": "catboost",
        "feature_selection_type": "catboost",
    }
    pipeline_settings = {
        "remainder": "drop" if len(categorical_features) == 0 else "passthrough",
        "verbose_feature_names_out": False,
    }
    if "regression" in output_type:
        model_settings["model_type"] = "regression"
        model_settings["target_type"] = "single_target" if "single-target" in output_type else "multi_target"
    elif "classification" in output_type:
        model_settings["feature_selection_type"] = "permutation"
        model_settings["model_type"] = (
            "classification_binary" if "binary" in output_type else "classification_multiclass"
        )
        model_settings["target_type"] = "single_target"
    elif "ranking" in output_type:
        model_settings["feature_selection_type"] = "permutation"
        model_settings["model_type"] = "regression"
        model_settings["target_type"] = "single_target"
    return pipeline_utils.get_feature_selection_pipeline(
        settings=model_settings, pipeline_settings=pipeline_settings, cv=cv
    ).set_output(transform="pandas")


@pytest.fixture()
def model_pipeline(ml_model, feature_selector: ColumnTransformer) -> PipelineWithHyperparameterRooting:
    model_pipeline = PipelineWithHyperparameterRooting(
        [("feature_selector", feature_selector.set_output(transform="pandas")), ("model", ml_model)]
    )

    return model_pipeline


@pytest.fixture()
def grouping_params() -> cv_module.TanimotoSimilarityParams:
    """Returns TanimotoSimilarityParams with a low similarity threshold for testing."""
    return cv_module.TanimotoSimilarityParams(similarity_threshold=0.1)
