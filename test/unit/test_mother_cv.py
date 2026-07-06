import numpy as np
import pandas as pd
import pytest
from optuna.samplers import RandomSampler
from optuna.trial import Trial
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from mother.ml import (
    CatboostClassifierMother,
    CatboostRegressorMother,
    get_available_algorithms,
)
from mother.ml.core import AbstractMotherPipeline, PipelineWithHyperparameterRooting
from mother.ml.models.m_lasso import LassoRegressorMother
from mother.ml.models.m_randomForest import (
    RandomForestClassifierMother,
    RandomForestRegressorMother,
)
from mother.optimization.core import MotherTuner
from mother.pipeline_utils import get_feature_selection_pipeline, mother_cv


@pytest.fixture()
def scorer_regression(request):
    return make_scorer(mean_squared_error)


@pytest.fixture()
def scorer_classification(request):
    return make_scorer(accuracy_score)


@pytest.fixture(params=[True, False])
def sampler(request):
    if request.param:
        return None  # default sampler
    return RandomSampler()


@pytest.fixture(params=[True, False])
def tuner_classification(scorer_classification, sampler, request):
    if request.param:
        return MotherTuner(scorer=scorer_classification, sampler=sampler, n_trials_optuna=5)
    else:
        None


@pytest.fixture(params=[True, False])
def tuner_regression(scorer_regression, sampler, request):
    if request.param:
        return MotherTuner(scorer=scorer_regression, sampler=sampler, n_trials_optuna=5)
    else:
        None


@pytest.fixture()
def cv() -> KFold:
    return KFold(n_splits=2, shuffle=True, random_state=42)


@pytest.fixture(params=[True, False])
def use_multitarget(request):
    return request.param


@pytest.fixture(params=[True, False])
def return_estimators(request):
    return request.param


@pytest.fixture()
def regression_pipeline(request) -> Pipeline:
    class LassoPipeline(Pipeline, BaseEstimator, AbstractMotherPipeline):
        """
        MOTHER pipeline for a LASSO regression
        """

        def __init__(self):
            super().__init__(steps=self.create_steps())

        def create_steps(
            self,
        ) -> list:
            steps = [
                ("s", MinMaxScaler()),
                ("m", LassoRegressorMother(alpha=1e-3, random_state=42)),
            ]

            return steps

        def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
            return {prefix + "m__alpha": trial.suggest_float(prefix + "m__alpha", 1e-6, 1e1, log=True)}

        def default_parameters(self, prefix: str = "") -> dict:
            return {prefix + "m__alpha": 1e-3}

    return LassoPipeline()


@pytest.fixture()
def classification_pipeline() -> Pipeline:
    class LogisticPipeline(Pipeline, BaseEstimator, AbstractMotherPipeline):
        """
        MOTHER pipeline for a LASSO regression
        """

        def __init__(self):
            super().__init__(steps=self.create_steps())

        def create_steps(
            self,
        ) -> list:
            steps = [
                ("s", MinMaxScaler()),
                ("m", LogisticRegression(C=1e-3, random_state=42)),
            ]

            return steps

        def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
            return {prefix + "m__C": trial.suggest_float(prefix + "m__C", 1e-6, 1e1, log=True)}

        def default_parameters(self, prefix: str = "") -> dict:
            return {prefix + "m__C": 1e-3}

    return LogisticPipeline()


@pytest.fixture()
def classification_pipeline_multitarget(request) -> Pipeline:
    class MultitargetClassificationPipeline(Pipeline, BaseEstimator, AbstractMotherPipeline):
        """
        MOTHER pipeline for a catboost multitarget binary regression
        """

        def __init__(self):
            super().__init__(steps=self.create_steps())

        def create_steps(
            self,
        ) -> list:
            steps = [
                ("s", MinMaxScaler()),
                ("m", CatboostClassifierMother(learning_rate=1e-3, iterations=2, loss_function="MultiLogloss")),
            ]

            return steps

        def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
            return {prefix + "m__learning_rate": trial.suggest_float(prefix + "m__learning_rate", 1e-6, 1e1, log=True)}

        def default_parameters(self, prefix: str = "") -> dict:
            return {prefix + "m__learning_rate": 1e-3}

    return MultitargetClassificationPipeline()


@pytest.fixture()
def regression_pipeline_mother_hyperparam_rooting(request) -> PipelineWithHyperparameterRooting:
    class RegressionMotherFeatureSelectionPipeline(PipelineWithHyperparameterRooting):
        """
        MOTHER pipeline for a catboost regression with uncertainty
        following a MOTHER feature selection
        """

        def __init__(self):
            super().__init__(steps=self.create_steps())

        def create_steps(
            self,
        ) -> list:
            # mother feature selection pipeline
            feature_selector = get_feature_selection_pipeline(
                settings={
                    "feature_selection_flags": ["DROP_CORRELATED"],
                    "correlation_threshold": 0.9,
                    "algorithm": "catboost",
                    "feature_selection_type": "catboost",
                    "model_type": "regression",
                    "target_type": "single_target",
                },
                pipeline_settings={
                    "remainder": "passthrough",
                    "verbose_feature_names_out": False,
                },
            ).set_output(transform="pandas")

            steps = [
                ("s", feature_selector),
                (
                    "m",
                    RandomForestRegressorMother(n_estimators=10),
                ),  # test predict_uncertainty
            ]

            return steps

        def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
            return {prefix + "m__learning_rate": trial.suggest_float(prefix + "m__learning_rate", 1e-6, 1e1, log=True)}

        def default_parameters(self, prefix: str = "") -> dict:
            return {prefix + "m__learning_rate": 1e-3}

    return RegressionMotherFeatureSelectionPipeline()


@pytest.fixture()
def synthetic_data_classification() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_features = 100

    X, y = make_blobs(n_samples=100, centers=2, n_features=n_features, random_state=42)
    rng = np.random.default_rng(seed=42)
    random_index = rng.choice(np.arange(10_000, 10_000 + X.shape[0]), size=X.shape[0], replace=False)

    X = pd.DataFrame(X, index=random_index)

    y_multitask = pd.DataFrame({"target_one": y, "target_two": y}, index=random_index)
    y = pd.DataFrame(y, columns=["target"], index=random_index)

    return X, y, y_multitask


@pytest.fixture(params=get_available_algorithms())
def all_classification_algorithms(request) -> BaseEstimator:
    algorithm = request.param

    if algorithm == "catboost":
        model = CatboostClassifierMother(target_type="single_target")
    elif algorithm == "randomforest":
        model = RandomForestClassifierMother()
    elif algorithm == "tabpfn":
        from mother.ml.models.m_tabpfn import TabPFNClassifierMother

        model = TabPFNClassifierMother()
    elif algorithm == "lasso":
        from mother.ml.models.m_lasso import LassoClassifierBinaryMother

        model = LassoClassifierBinaryMother()
    return model


@pytest.fixture(params=get_available_algorithms())
def all_regression_algorithms(request) -> BaseEstimator:
    algorithm = request.param

    if algorithm == "catboost":
        model = CatboostRegressorMother(target_type="single_target")
    elif algorithm == "randomforest":
        model = RandomForestRegressorMother()
    elif algorithm == "tabpfn":
        from mother.ml.models.m_tabpfn import TabPFNRegressorMother

        model = TabPFNRegressorMother()
    elif algorithm == "lasso":
        model = LassoRegressorMother()
    return model


@pytest.fixture()
def synthetic_data_regression() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_features = 100

    X_array, y_array = make_blobs(n_samples=100, centers=2, n_features=n_features, random_state=42)
    rng = np.random.default_rng(seed=42)
    random_index = rng.choice(np.arange(10_000, 10_000 + X_array.shape[0]), size=X_array.shape[0], replace=False)
    X: pd.DataFrame = pd.DataFrame(X_array, index=random_index)
    y_multitask: pd.DataFrame = pd.DataFrame({"targetOne": y_array, "targetTwo": y_array}, index=random_index)
    y: pd.DataFrame = pd.DataFrame(y_array, columns=["target"], index=random_index)

    return X, y, y_multitask


def test_optimize_and_train_mother_cv_regression(
    tuner_regression,
    regression_pipeline,
    synthetic_data_regression,
    cv,
    use_multitarget,
    return_estimators,
    tmp_path,
    monkeypatch,
):
    # The LassoPipeline is not suited for multi-target regression, so we skip this test case.
    if "Lasso" in regression_pipeline.__class__.__name__ and use_multitarget:
        pytest.skip("LassoRegressorMother does not support multi-target regression.")

    if return_estimators:
        monkeypatch.chdir(tmp_path)

    X, y, y_multitask = synthetic_data_regression

    # Create a random number generator
    rng = np.random.default_rng(seed=42)

    # Generate the group vector using the random number generator
    groups: pd.DataFrame = pd.DataFrame(
        {"Groups": rng.choice(range(1, 6), size=X.shape[0], replace=True)},
        index=X.index,
    )

    result = mother_cv(
        regression_pipeline,
        tuner=tuner_regression,
        inner_cv=cv,
        groups=groups,
        cv=cv,
        X=X,
        y=y if not use_multitarget else y_multitask,
        return_estimators=return_estimators,
    )

    if return_estimators:
        cv_table, estimators = result
    else:
        cv_table = result

    assert cv_table.shape[0] == X.shape[0]
    assert np.sum(cv_table.columns.str.count("_proba").values) == 0
    assert len(np.unique(cv_table["iteration"])) == 2
    assert cv_table["test_index"].is_monotonic_increasing

    if return_estimators:
        assert "estimators" in estimators
        assert "prediction_prefix" in estimators
        assert "target_columns" in estimators
        assert len(estimators["estimators"]) == cv.get_n_splits()

    if use_multitarget:
        assert {"targetOne", "targetTwo", "cv_group", "iteration", "test_index"}.issubset(cv_table.columns)
        expected_pred_cols = {"pred_target_0_pred", "pred_target_0_mean_predictions"}
        assert any(col in cv_table.columns for col in expected_pred_cols), (
            f"No prediction columns found. Expected one of {expected_pred_cols}, got {list(cv_table.columns)}"
        )

    else:
        assert np.array_equal(np.asarray(cv_table["target"].values), np.ravel(y))
        assert {"target", "cv_group", "iteration", "test_index"}.issubset(cv_table.columns)


def test_optimize_and_train_mother_cv_hyperparam_rooting(
    tuner_regression,
    regression_pipeline_mother_hyperparam_rooting,
    synthetic_data_regression,
    cv,
    return_estimators,
    tmp_path,
    monkeypatch,
):
    if return_estimators:
        monkeypatch.chdir(tmp_path)

    X, y, _ = synthetic_data_regression

    # Create a random number generator
    rng = np.random.default_rng(seed=42)

    # Generate the group vector using the random number generator
    groups: pd.DataFrame = pd.DataFrame(
        {"Groups": rng.choice(range(1, 6), size=X.shape[0], replace=True)}, index=X.index
    )

    result = mother_cv(
        regression_pipeline_mother_hyperparam_rooting,
        tuner=tuner_regression,
        inner_cv=cv,
        groups=groups,
        cv=cv,
        X=X,
        y=y,
        return_estimators=return_estimators,
    )

    if return_estimators:
        cv_table, estimators = result
    else:
        cv_table = result

    print(cv_table)

    assert cv_table.shape[0] == X.shape[0]
    assert np.sum(cv_table.columns.str.count("_proba").values) == 0
    assert len(np.unique(cv_table["iteration"])) == 2
    assert cv_table["test_index"].is_monotonic_increasing

    assert np.array_equal(cv_table["target"].values, np.ravel(y))
    assert {"target", "cv_group", "iteration", "test_index"}.issubset(cv_table.columns)

    if return_estimators:
        assert "estimators" in estimators
        assert "prediction_prefix" in estimators
        assert "target_columns" in estimators
        assert len(estimators["estimators"]) == cv.get_n_splits()


def test_optimize_and_train_mother_cv_classification(
    tuner_classification,
    classification_pipeline,
    synthetic_data_classification,
    cv,
    return_estimators,
    tmp_path,
    monkeypatch,
):
    if return_estimators:
        monkeypatch.chdir(tmp_path)

    X, y, _ = synthetic_data_classification

    result = mother_cv(
        classification_pipeline,
        tuner=tuner_classification,
        inner_cv=cv,
        cv=cv,
        X=X,
        y=y,
        return_estimators=return_estimators,
    )

    if return_estimators:
        cv_table, estimators = result
    else:
        cv_table = result

    assert cv_table.shape[0] == X.shape[0]
    assert np.sum(cv_table.columns.str.count("_proba").values) == 2
    assert cv_table["test_index"].is_monotonic_increasing

    if return_estimators:
        assert "estimators" in estimators
        assert "prediction_prefix" in estimators
        assert "target_columns" in estimators
        assert len(estimators["estimators"]) == cv.get_n_splits()


def test_optimize_and_train_mother_cv_multiclassification(
    tuner_classification,
    classification_pipeline_multitarget,
    synthetic_data_classification,
    cv,
    return_estimators,
    tmp_path,
    monkeypatch,
):
    if return_estimators:
        monkeypatch.chdir(tmp_path)

    X, _, y_multitarget = synthetic_data_classification

    result = mother_cv(
        classification_pipeline_multitarget,
        tuner=tuner_classification,
        inner_cv=cv,
        cv=cv,
        X=X,
        y=y_multitarget,
        return_estimators=return_estimators,
    )

    if return_estimators:
        cv_table, estimators = result
    else:
        cv_table = result

    assert cv_table.shape[0] == X.shape[0]
    assert np.sum(cv_table.columns.str.count("_proba").values) == 2
    assert cv_table["test_index"].is_monotonic_increasing

    if return_estimators:
        assert len(estimators["estimators"]) == cv.get_n_splits()


@pytest.mark.slow
def test_mother_cv_all_classification_algorithms(
    tuner_classification,
    all_classification_algorithms,
    synthetic_data_classification,
    cv,
    return_estimators,
    tmp_path,
    monkeypatch,
):
    """
    Test if mother_cv runs with all available classification algorithms
    """
    if return_estimators:
        monkeypatch.chdir(tmp_path)

    X, y, _ = synthetic_data_classification

    result = mother_cv(
        all_classification_algorithms,
        tuner=tuner_classification,
        inner_cv=cv,
        cv=cv,
        X=X,
        y=y,
        return_estimators=return_estimators,
    )

    if return_estimators:
        cv_table, estimators = result
        assert "estimators" in estimators
        assert "prediction_prefix" in estimators
        assert "target_columns" in estimators
    else:
        cv_table = result
        assert cv_table.shape[0] == X.shape[0]
        assert np.sum(cv_table.columns.str.count("_proba").values) == 2
        assert cv_table["test_index"].is_monotonic_increasing


@pytest.mark.slow
def test_mother_cv_all_regression_algorithms(
    tuner_regression,
    all_regression_algorithms,
    synthetic_data_regression,
    cv,
    return_estimators,
    tmp_path,
    monkeypatch,
):
    """
    Test if mother_cv runs with all available regression algorithms
    """
    if return_estimators:
        monkeypatch.chdir(tmp_path)

    X, y, _ = synthetic_data_regression

    result = mother_cv(
        all_regression_algorithms,
        tuner=tuner_regression,
        inner_cv=cv,
        cv=cv,
        X=X,
        y=y,
        return_estimators=return_estimators,
    )

    if return_estimators:
        cv_table, estimators = result
        assert "estimators" in estimators
        assert "prediction_prefix" in estimators
        assert "target_columns" in estimators
    else:
        cv_table = result
        assert cv_table.shape[0] == X.shape[0]
        assert np.sum(cv_table.columns.str.count("_proba").values) == 0
        assert cv_table["test_index"].is_monotonic_increasing


@pytest.mark.parametrize(
    "pipeline_fixture,data_fixture",
    [
        ("regression_pipeline", "synthetic_data_regression"),
        ("classification_pipeline", "synthetic_data_classification"),
    ],
)
def test_mother_cv_raises_error_on_mismatched_group_index(
    pipeline_fixture,
    data_fixture,
    cv,
    request,
    return_estimators,
):
    """
    Test that mother_cv raises ValueError when groups have a different index than X
    for both classification and regression cases
    """
    pipeline = request.getfixturevalue(pipeline_fixture)
    synthetic_data = request.getfixturevalue(data_fixture)
    X, y, _ = synthetic_data

    rng = np.random.default_rng(seed=42)
    groups_mismatched: pd.DataFrame = pd.DataFrame({"Groups": rng.choice(range(1, 6), size=X.shape[0], replace=True)})

    with pytest.raises(ValueError, match="groups must have the same index as X"):
        mother_cv(
            pipeline,
            inner_cv=cv,
            cv=cv,
            X=X,
            y=y,
            groups=groups_mismatched,
            return_estimators=return_estimators,
        )


def test_mother_cv_raises_error_on_invalid_estimator_type(synthetic_data_regression, cv, return_estimators):
    X, y, _ = synthetic_data_regression

    with pytest.raises(
        ValueError,
        match="Estimator must be a PipelineWithHyperparameterRooting or AbstractMotherPipeline",
    ):
        mother_cv(
            object(),
            cv=cv,
            X=X,
            y=y,
            return_estimators=return_estimators,
        )


def test_mother_cv_return_estimators_as_tuple(
    regression_pipeline,
    synthetic_data_regression,
    cv,
    tmp_path,
    monkeypatch,
):
    X, y, _ = synthetic_data_regression
    monkeypatch.chdir(tmp_path)

    result = mother_cv(
        regression_pipeline,
        cv=cv,
        X=X,
        y=y,
        return_estimators=True,
    )

    cv_table, estimators = result

    assert isinstance(cv_table, pd.DataFrame)
    assert isinstance(estimators, dict)
    assert "estimators" in estimators
    assert "prediction_prefix" in estimators
    assert "target_columns" in estimators
    assert isinstance(estimators["estimators"], list)
    assert len(estimators["estimators"]) == cv.get_n_splits()
    assert estimators["prediction_prefix"] == "pred_"
    assert estimators["target_columns"] == ["target"]
