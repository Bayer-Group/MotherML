import numpy as np
import pandas as pd
import pytest
from optuna.samplers import RandomSampler
from optuna.trial import Trial
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from mother.ml import CatboostClassifierMother, avg_ndcg_score
from mother.ml.core import (
    AbstractMotherPipeline,
    PipelineWithHyperparameterRooting,
)
from mother.optimization.core import MotherTuner


# define a simple model to run the tests with
@pytest.fixture()
def lasso_pipeline() -> Pipeline:
    """will return a simple Lasso model hyperparam steps"""

    class LassoPipeline(Pipeline, BaseEstimator, AbstractMotherPipeline):
        """
        MOTHER pipeline for a LASSO regression
        """

        def __init__(self):
            super().__init__(steps=self.create_steps())

        def create_steps(
            self,
        ) -> list[tuple[str, BaseEstimator]]:
            steps = [
                ("s", MinMaxScaler()),
                ("m", Lasso(alpha=1e-3, max_iter=3000)),
            ]

            return steps

        def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
            return {prefix + "m__alpha": trial.suggest_float(prefix + "m__alpha", 1e-6, 1e1, log=True)}

        def default_parameters(self, prefix: str = "") -> dict:
            return {prefix + "m__alpha": 1e-3}

    return LassoPipeline()


@pytest.fixture()
def scorer(request, modeling_data):
    if "ranking" not in modeling_data[1]:
        return make_scorer(mean_squared_error, greater_is_better=False)

    def score_func(y, y_pred, group_id):
        return avg_ndcg_score(y, y_pred, group_id, k=3)

    return make_scorer(score_func, greater_is_better=True).set_score_request(group_id="group_id")


@pytest.fixture(params=[True, False], ids=["with_sampler", "without_sampler"])
def sampler(request):
    if request.param:
        return None  # default sampler
    return RandomSampler()


@pytest.fixture()
def tuner(scorer, sampler):
    return MotherTuner(scorer=scorer, sampler=sampler, n_trials_optuna=7)


@pytest.fixture()
def synthetic_data():
    n_features = 100
    X, y = make_blobs(n_samples=100, centers=5, n_features=n_features)
    X = pd.DataFrame(X)
    y_multitask = pd.DataFrame({"Column1": y, "Column2": y})
    y = pd.DataFrame(y)
    return X, y, y_multitask


@pytest.fixture(params=[True, False], ids=["single_target", "multi_target"])
def use_multitarget(request):
    return request.param


def test_optimize_and_train_tune(
    tuner, lasso_pipeline, synthetic_data, cv_cross_validation, use_multitarget, modeling_data
):
    if "ranking" in modeling_data[1]:
        pytest.skip("Skipping test for ranking tasks")

    X, y, y_multitask = synthetic_data

    # try with the functions defined in the model class
    model = tuner.optimize(
        lasso_pipeline,
        X=X,
        y=y if not use_multitarget else y_multitask,
        hyperparameter_space_function=lasso_pipeline.get_hyperparameter_space,
        default_parameters=lasso_pipeline.default_parameters(),
        cross_validation=cv_cross_validation,
    )

    # predict with the model
    prediction = model.predict(X)
    if not use_multitarget:
        assert prediction.ndim == 1

        assert not np.isnan(prediction).any()

        # Check if the default parameters have been evaluated in the study

    evaluated = tuner.study.trials[0].params
    enqueued = lasso_pipeline.default_parameters()
    all(evaluated[k] == enqueued[k] for k in evaluated.keys() & enqueued.keys())


@pytest.fixture()
def classification_tuner():
    """
    Fixture for the MotherTuner with accuracy as the scorer.
    """
    scorer = make_scorer(accuracy_score, greater_is_better=True)
    return MotherTuner(scorer=scorer, sampler=None, n_trials_optuna=5)


@pytest.mark.parametrize("n_classes, target_type", [(2, "single_target"), (2, "multi_target"), (3, "single_target")])
def test_tune_mother_model_classification_with_catboost(
    n_classes, target_type, classification_tuner, cv_cross_validation
):
    """
    Test the tuning of CatboostClassifierMother
    for binary (single and multi-target) and multiclass classification tasks, using a pipeline.
    """
    import pandas as pd

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=n_classes,
        n_informative=5,
        random_state=42,
    )

    # Convert X to a pandas DataFrame
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # Handle multi-target case for binary classification
    if n_classes == 2 and target_type == "multi_target":
        y = pd.DataFrame({"target_1": y, "target_2": y})
    else:
        y = pd.DataFrame(y, columns=["target"])

    # Initialize the appropriate CatBoost classifier
    if n_classes == 2:
        base_model = CatboostClassifierMother(
            target_type=target_type,
            model_type="classification_binary",
            iterations=10,
            learning_rate=0.1,
            max_depth=6,
            tune_boosting_type=True,
            random_seed=42,
        )
    else:
        base_model = CatboostClassifierMother(
            target_type=target_type,
            model_type="classification_multiclass",
            iterations=10,
            learning_rate=0.1,
            max_depth=6,
            random_seed=42,
        )

    # Wrap the model in a pipeline
    pipeline = PipelineWithHyperparameterRooting(
        [
            ("classifier", base_model),
        ]
    )

    # Optimize the pipeline using the tuner
    optimized_pipeline = classification_tuner.optimize(
        pipeline,
        X=X,
        y=y,
        cross_validation=cv_cross_validation,
    )

    # Make predictions
    probabilities = optimized_pipeline.predict_proba(X)

    # Assertions for binary classification
    if n_classes == 2:
        assert base_model.target_type == target_type
        assert probabilities.shape[1] == 2, "Binary classification probabilities should have 2 columns."

    # Assertions for multiclass classification
    else:
        assert base_model.target_type == target_type
        assert probabilities.shape[1] == n_classes, f"Multiclass probabilities should have {n_classes} columns."


@pytest.mark.slow
@pytest.mark.serial
class TestTuneMotherModels:
    @pytest.mark.parametrize("cv_cross_validation", [False, True], indirect=True, ids=["no_cv", "with_cv"])
    def test_tune_mother_model(self, modeling_data, ml_model, tuner, cv_cross_validation, preserve_metadata_routing):
        features, output_type, target, categorical_features, ranking_groups, cv_groups = modeling_data

        if isinstance(tuner.scorer, type(make_scorer(mean_squared_error))) and ("classification" in output_type):
            # skip testing of custom regression scorer with classification task
            return

        ranking_kwargs = {}
        if "ranking" in modeling_data[1]:
            ranking_kwargs["ranking_groups"] = ranking_groups

        _ = tuner.optimize(
            ml_model,
            X=features,
            y=target,
            cross_validation=cv_cross_validation,
            groups=cv_groups if isinstance(cv_cross_validation, GroupKFold) else None,
            **ranking_kwargs,
        )

        # Check if the default parameters have been evaluated in the study
        evaluated = tuner.study.trials[0].params
        enqueued = ml_model.default_parameters()
        all(evaluated[k] == enqueued[k] for k in evaluated.keys() & enqueued.keys())

    @pytest.mark.parametrize("cv_cross_validation", [False, True], indirect=True)
    def test_tune_mother_feature_selection_model_pipeline(
        self, modeling_data, model_pipeline, tuner, cv_cross_validation
    ):
        features, output_type, target, categorical_features, ranking_groups, cv_groups = modeling_data

        if isinstance(tuner.scorer, type(make_scorer(mean_squared_error))) and ("classification" in output_type):
            # skip testing of custom regression scorer with classification task
            pytest.skip("Skipping test for classification tasks with regression scorer")

        if cv_cross_validation and ("ranking" not in modeling_data[1]):
            # skip testing of cross validation with GroupKFold for non ranking tasks
            pytest.skip("Skipping test for non-ranking tasks with GroupKFold cross-validation")

        ranking_kwargs = {}
        if "ranking" in modeling_data[1]:
            ranking_kwargs["ranking_groups"] = ranking_groups

        _ = tuner.optimize(
            model_pipeline,
            X=features,
            y=target,
            groups=cv_groups if isinstance(cv_cross_validation, GroupKFold) else None,
            cross_validation=cv_cross_validation,
            **ranking_kwargs,
        )

        evaluated = tuner.study.trials[0].params
        enqueued = model_pipeline.default_parameters()
        all(evaluated[k] == enqueued[k] for k in evaluated.keys() & enqueued.keys())
