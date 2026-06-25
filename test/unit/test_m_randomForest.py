import logging
import pickle

import numpy as np
import pandas as pd
import pytest
from optuna.trial import FixedTrial
from sklearn.base import clone

from mother.ml.models.m_randomForest import (
    RandomForestClassifierMother,
    RandomForestRegressorMother,
)

SEED = 42


@pytest.fixture
def clf_sample_data():
    generator = np.random.default_rng(SEED)
    X = pd.DataFrame(generator.random((20, 4)))
    y = generator.integers(0, 2, size=20)
    return X, y


@pytest.fixture
def reg_sample_data():
    generator = np.random.default_rng(SEED)
    X = pd.DataFrame(generator.random((20, 4)))
    y = generator.random(20)
    return X, y


def test_classifier_mother_init_and_fit_predict(clf_sample_data):
    X, y = clf_sample_data
    clf = RandomForestClassifierMother(n_estimators=10)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})


def test_classifier_mother_predict_uncertainty_includes_probabilities(clf_sample_data):
    X, y = clf_sample_data
    clf = RandomForestClassifierMother(n_estimators=10)
    clf.fit(X, y)

    preds = clf.predict(X)
    preds_with_uncertainty = clf.predict_uncertainty(X)

    assert isinstance(preds_with_uncertainty, pd.DataFrame)
    assert preds_with_uncertainty.shape[0] == X.shape[0]
    assert "pred" in preds_with_uncertainty.columns
    assert "mean_predictions" in preds_with_uncertainty.columns
    assert "proba_0" in preds_with_uncertainty.columns
    assert "proba_1" in preds_with_uncertainty.columns
    assert preds_with_uncertainty["mean_predictions"].isna().all()
    assert np.allclose(preds_with_uncertainty["proba_0"] + preds_with_uncertainty["proba_1"], 1.0)
    np.testing.assert_array_equal(preds_with_uncertainty["pred"].to_numpy(), preds)


def test_regressor_mother_init_and_fit_predict(reg_sample_data):
    X, y = reg_sample_data
    reg = RandomForestRegressorMother(n_estimators=10)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (20,)
    assert np.issubdtype(preds.dtype, np.floating)


def test_regressor_mother_init_and_fit_predict_uncertainty(reg_sample_data):
    X, y = reg_sample_data
    reg = RandomForestRegressorMother(n_estimators=10)
    reg.fit(X, y)
    preds = reg.predict_uncertainty(X)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == (20, 5)
    required_columns = pd.Index(
        ["pred", "mean_predictions", "knowledge_uncertainty", "data_uncertainty", "total_uncertainty"]
    )
    assert required_columns.isin(preds.columns).all()


def test_regressor_mother_init_and_fit_predict_uncertainty_for_opt(reg_sample_data):
    X, y = reg_sample_data
    reg = RandomForestRegressorMother(n_estimators=10)
    reg.fit(X, y)
    preds_2 = reg.predict_uncertainty(X, quantiles=[0.10, 0.5, 0.9], uncertainty_for_opt=True)
    assert isinstance(preds_2, pd.DataFrame)
    assert preds_2.shape == (20, 1)
    assert preds_2.columns.equals(pd.Index(["total_uncertainty"]))


def test_regressor_mother_init_and_fit_predict_return_quantiles(reg_sample_data):
    X, y = reg_sample_data
    reg = RandomForestRegressorMother(n_estimators=10)
    reg.fit(X, y)
    _, quantiles = reg.predict_uncertainty(
        X, quantiles=[0.10, 0.5, 0.9], uncertainty_for_opt=True, return_quantiles=True
    )
    assert isinstance(quantiles, pd.DataFrame)
    # DEFAULT_QUANTILES [0.25, 0.5, 0.75] are automatically added
    assert quantiles.shape == (20, 5)
    expected_quantile_columns = ["quantile_0.1", "quantile_0.25", "quantile_0.5", "quantile_0.75", "quantile_0.9"]
    assert list(quantiles.columns) == expected_quantile_columns


def test_regressor_mother_predict_uncertainty_multitarget():
    generator = np.random.default_rng(SEED)
    X = pd.DataFrame(generator.random((20, 4)), index=[f"sample_{idx}" for idx in range(20)])
    y = pd.DataFrame(generator.random((20, 3)), columns=["a", "b", "c"], index=X.index)

    reg = RandomForestRegressorMother(n_estimators=10)
    reg.fit(X, y)

    preds = reg.predict_uncertainty(X)
    expected_columns = pd.Index(
        [
            "target_0_pred",
            "target_1_pred",
            "target_2_pred",
            "target_0_mean_predictions",
            "target_1_mean_predictions",
            "target_2_mean_predictions",
            "target_0_knowledge_uncertainty",
            "target_1_knowledge_uncertainty",
            "target_2_knowledge_uncertainty",
            "target_0_data_uncertainty",
            "target_1_data_uncertainty",
            "target_2_data_uncertainty",
            "target_0_total_uncertainty",
            "target_1_total_uncertainty",
            "target_2_total_uncertainty",
        ]
    )

    assert isinstance(preds, pd.DataFrame)
    assert preds.columns.equals(expected_columns)
    assert preds.index.equals(X.index)

    pred_values = reg.predict(X)
    np.testing.assert_allclose(preds[["target_0_pred", "target_1_pred", "target_2_pred"]].to_numpy(), pred_values)
    assert (
        preds[["target_0_knowledge_uncertainty", "target_1_knowledge_uncertainty", "target_2_knowledge_uncertainty"]]
        .isna()
        .all()
        .all()
    )
    assert (
        preds[["target_0_data_uncertainty", "target_1_data_uncertainty", "target_2_data_uncertainty"]]
        .isna()
        .all()
        .all()
    )

    preds_opt = reg.predict_uncertainty(X, uncertainty_for_opt=True)
    assert preds_opt.columns.equals(pd.Index(["total_uncertainty"]))
    assert len(preds_opt) == len(X)
    assert preds_opt.index.equals(X.index)


def test_classifier_mother_default_parameters():
    clf = RandomForestClassifierMother()
    defaults = clf.default_parameters()
    assert "criterion" in defaults
    assert "max_features" in defaults
    assert "min_samples_leaf" in defaults
    assert defaults["criterion"] in ["gini", "entropy", "log_loss"]


def test_regressor_mother_default_parameters():
    reg = RandomForestRegressorMother()
    defaults = reg.default_parameters()
    assert "criterion" in defaults
    assert "max_features" in defaults
    assert "min_samples_leaf" in defaults
    assert defaults["criterion"] == "squared_error"


def test_classifier_mother_hyperparameter_space(clf_sample_data):
    X, y = clf_sample_data
    trial = FixedTrial(
        {
            "criterion": "gini",
            "max_features": "sqrt",
            "min_samples_leaf": 2,
        }
    )
    clf = RandomForestClassifierMother()
    params = clf.get_hyperparameter_space(X, y, trial)  # type: ignore
    assert params["criterion"] == "gini"
    assert params["max_features"] == "sqrt"
    assert params["min_samples_leaf"] == 2


def test_regressor_mother_hyperparameter_space_with_negative_y(reg_sample_data):
    X, y = reg_sample_data
    trial = FixedTrial(
        {
            "max_features": "log2",
            "min_samples_leaf": 3,
            "criterion": "squared_error",
        }
    )
    reg = RandomForestRegressorMother()
    params = reg.get_hyperparameter_space(X, y, trial)
    assert params["max_features"] == "log2"
    assert params["min_samples_leaf"] == 3
    assert params["criterion"] == "squared_error"


def test_regressor_mother_hyperparameter_space_with_positive_y():
    generator = np.random.default_rng(SEED)
    X = generator.random((10, 2))
    y = generator.uniform(0, 6, size=10)

    trial = FixedTrial(
        {
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "criterion": "poisson",
        }
    )
    reg = RandomForestRegressorMother()
    params = reg.get_hyperparameter_space(X, y, trial)
    assert params["criterion"] == "poisson"
    assert params["max_features"] == "sqrt"
    assert params["min_samples_leaf"] == 1


@pytest.fixture
def multitask_sample_data():
    generator = np.random.default_rng(SEED)
    X = pd.DataFrame(generator.random((100, 5)), columns=[f"feature_{i}" for i in range(5)])
    y = pd.DataFrame({f"target_{i}": generator.standard_normal(100) for i in range(2)})
    return X, y


def test_model_fit_and_predict_multiclass(multitask_sample_data):
    X, y = multitask_sample_data
    model = RandomForestRegressorMother()
    model.fit(X, y)
    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == X.shape[0]
    assert predictions.shape[1] == y.shape[1]


def test_uncertainty_for_opt_multitask_regression(multitask_sample_data):
    X, y = multitask_sample_data
    model = RandomForestRegressorMother()
    model.fit(X, y)
    output = model.predict_uncertainty(X, uncertainty_for_opt=True)
    assert isinstance(output, pd.DataFrame)
    assert output.shape == (X.shape[0], 1)
    assert list(output.columns) == ["total_uncertainty"]
    assert all(output.index == X.index)
    assert (output >= 0).all().all()


def test_regressor_pickle_clone(reg_sample_data):
    X, y = reg_sample_data
    additional_params = {"max_samples": 0.8, "bootstrap": True}
    model = RandomForestRegressorMother(**additional_params)

    assert np.isclose(getattr(model, "max_samples", None), 0.8)
    assert getattr(model, "bootstrap", None) is True
    assert np.isclose(getattr(model, "max_features", None), 1.0)  # default for base class

    cloned_model = clone(model)
    assert np.isclose(getattr(cloned_model, "max_samples", None), 0.8)
    assert getattr(cloned_model, "bootstrap", None) is True
    assert np.isclose(getattr(cloned_model, "max_features", None), 1.0)

    cloned_model.fit(X, y)
    pickled_model = pickle.dumps(cloned_model)
    unpickled_model = pickle.loads(pickled_model)

    assert np.isclose(getattr(unpickled_model, "max_samples", None), 0.8)
    assert getattr(unpickled_model, "bootstrap", None) is True
    assert np.isclose(getattr(unpickled_model, "max_features", None), 1.0)


def test_classifier_pickle_clone(clf_sample_data):
    X, y = clf_sample_data
    additional_params = {"max_samples": 0.8, "bootstrap": True}
    model = RandomForestClassifierMother(**additional_params)

    assert getattr(model, "n_estimators", None) == 500
    assert getattr(model, "bootstrap", None) is True
    # default for scikit-learn's implementation
    assert getattr(model, "max_features", None) == "sqrt"

    cloned_model = clone(model)
    assert getattr(cloned_model, "n_estimators", None) == 500
    assert getattr(cloned_model, "bootstrap", None) is True
    assert getattr(cloned_model, "max_features", None) == "sqrt"

    cloned_model.fit(X, y)

    pickled_model = pickle.dumps(cloned_model)
    unpickled_model = pickle.loads(pickled_model)

    assert getattr(unpickled_model, "n_estimators", None) == 500
    assert getattr(unpickled_model, "bootstrap", None) is True
    assert getattr(unpickled_model, "max_features", None) == "sqrt"


def test_classifier_mother_init_invalid_class_weight():
    with pytest.raises(ValueError, match="Invalid class weight"):
        RandomForestClassifierMother(class_weight="invalid_weight")


def test_classifier_mother_default_parameters_with_kwargs():
    clf = RandomForestClassifierMother()
    custom_defaults = clf.default_parameters(custom_param=42)
    assert custom_defaults["custom_param"] == 42
    assert "custom_param" in custom_defaults


def test_regressor_mother_default_parameters_with_kwargs():
    reg = RandomForestRegressorMother()
    custom_defaults = reg.default_parameters(custom_param=99)
    assert custom_defaults["custom_param"] == 99
    assert "custom_param" in custom_defaults


def test_classifier_mother_logging_on_kwargs_override(caplog):
    clf = RandomForestClassifierMother(n_estimators=10)
    with caplog.at_level(logging.WARNING):
        clf.default_parameters(n_estimators=20)
    assert "Default parameters for RandomForestRegressorMother are being overridden by provided kwargs." in caplog.text


def test_regressor_mother_logging_on_kwargs_override(caplog):
    reg = RandomForestRegressorMother(min_samples_leaf=3)
    with caplog.at_level(logging.WARNING):
        reg.default_parameters(min_samples_leaf=10)
    assert "Default parameters for RandomForestRegressorMother are being overridden by provided kwargs." in caplog.text


def test_classifier_mother_fit_empty_data():
    X_empty = pd.DataFrame(columns=["feature_1", "feature_2", "feature_3", "feature_4"])
    y_empty = pd.Series(dtype=int)
    clf = RandomForestClassifierMother(n_estimators=10)
    with pytest.raises(ValueError, match="while a minimum of 1 is required by RandomForestClassifierMother."):
        clf.fit(X_empty, y_empty)


def test_regressor_mother_fit_empty_data():
    X_empty = pd.DataFrame(columns=["feature_1", "feature_2", "feature_3", "feature_4"])
    y_empty = pd.Series(dtype=float)
    reg = RandomForestRegressorMother(n_estimators=10)
    with pytest.raises(ValueError, match="while a minimum of 1 is required by RandomForestRegressorMother."):
        reg.fit(X_empty, y_empty)


def test_classifier_clone_after_set_params(clf_sample_data):
    """Test that cloning works after setting params via set_params (Optuna scenario)."""
    X, y = clf_sample_data

    # Create base instance
    model = RandomForestClassifierMother(n_estimators=100)

    # Set params via set_params (like Optuna does)
    model.set_params(criterion="gini", max_depth=5, min_samples_split=3)

    # Verify params are set
    params = model.get_params()
    assert params["criterion"] == "gini"
    assert params["max_depth"] == 5
    assert params["min_samples_split"] == 3

    # Clone should work without KeyError
    cloned_model = clone(model)

    # Verify all params are preserved
    cloned_params = cloned_model.get_params()
    assert cloned_params["criterion"] == "gini"
    assert cloned_params["max_depth"] == 5
    assert cloned_params["min_samples_split"] == 3
    assert cloned_params["n_estimators"] == 100

    # Verify cloned model can be fitted
    cloned_model.fit(X, y)
    predictions = cloned_model.predict(X)
    assert len(predictions) == len(y)


def test_regressor_clone_after_set_params(reg_sample_data):
    """Test that cloning works after setting params via set_params (Optuna scenario)."""
    X, y = reg_sample_data

    # Create base instance
    model = RandomForestRegressorMother(n_estimators=100)

    # Set params via set_params (like Optuna does)
    model.set_params(criterion="absolute_error", max_depth=10, min_samples_split=2)

    # Verify params are set
    params = model.get_params()
    assert params["criterion"] == "absolute_error"
    assert params["max_depth"] == 10
    assert params["min_samples_split"] == 2

    # Clone should work without KeyError
    cloned_model = clone(model)

    # Verify all params are preserved
    cloned_params = cloned_model.get_params()
    assert cloned_params["criterion"] == "absolute_error"
    assert cloned_params["max_depth"] == 10
    assert cloned_params["min_samples_split"] == 2
    assert cloned_params["n_estimators"] == 100

    # Verify cloned model can be fitted
    cloned_model.fit(X, y)
    predictions = cloned_model.predict(X)
    assert len(predictions) == len(y)


def test_classifier_optuna_optimization_with_cv(clf_sample_data):
    """Test actual Optuna optimization with cross-validation (real-world scenario)."""
    import optuna
    from sklearn.model_selection import cross_val_score

    X, y = clf_sample_data

    def objective(trial):
        # Create base model
        model = RandomForestClassifierMother(n_estimators=50)

        # Get hyperparameter suggestions
        params = model.get_hyperparameter_space(X, y, trial)

        # Set parameters (this is where the bug would occur)
        model.set_params(**params)

        # Cross-validation with cloning (this triggers the bug if not fixed)
        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")

        return scores.mean()

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3, show_progress_bar=False)

    # Verify we got valid results
    assert len(study.trials) == 3
    assert all(trial.value is not None for trial in study.trials)
    assert all(0 <= trial.value <= 1 for trial in study.trials)

    # Verify best params can be used to create a working model
    best_params = study.best_params
    final_model = RandomForestClassifierMother(n_estimators=50)
    final_model.set_params(**best_params)
    final_model.fit(X, y)
    predictions = final_model.predict(X)
    assert len(predictions) == len(y)


def test_regressor_optuna_optimization_with_cv(reg_sample_data):
    """Test actual Optuna optimization with cross-validation (real-world scenario)."""
    import optuna
    from sklearn.model_selection import cross_val_score

    X, y = reg_sample_data

    def objective(trial):
        # Create base model
        model = RandomForestRegressorMother(n_estimators=50)

        # Get hyperparameter suggestions
        params = model.get_hyperparameter_space(X, y, trial)

        # Set parameters (this is where the bug would occur)
        model.set_params(**params)

        # Cross-validation with cloning (this triggers the bug if not fixed)
        scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error")

        return scores.mean()

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3, show_progress_bar=False)

    # Verify we got valid results
    assert len(study.trials) == 3
    assert all(trial.value is not None for trial in study.trials)
    assert all(trial.value <= 0 for trial in study.trials)  # negative MSE

    # Verify best params can be used to create a working model
    best_params = study.best_params
    final_model = RandomForestRegressorMother(n_estimators=50)
    final_model.set_params(**best_params)
    final_model.fit(X, y)
    predictions = final_model.predict(X)
    assert len(predictions) == len(y)


# ---------------------------------------------------------------------------
# Issue #444 — point 2: uncertainty_for_opt logs a warning for multi-task
# ---------------------------------------------------------------------------


def test_rf_regressor_uncertainty_for_opt_multitarget_warns(caplog):
    """RandomForestRegressorMother logs a warning and returns total_uncertainty for multi-target uncertainty_for_opt."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((60, 5)), columns=[f"f_{i}" for i in range(5)])
    y = pd.DataFrame(rng.random((60, 2)), columns=["t0", "t1"])
    X_train, X_test = X.iloc[:50], X.iloc[50:]

    model = RandomForestRegressorMother(n_estimators=20)
    model.fit(X_train, y.iloc[:50])

    with caplog.at_level(logging.WARNING, logger="mother.ml.models.m_randomForest"):
        result = model.predict_uncertainty(X_test, uncertainty_for_opt=True)

    assert any("max of per-target" in m for m in caplog.messages), (
        f"Expected a 'max of per-target' warning; got: {caplog.messages}"
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["total_uncertainty"]
    assert len(result) == len(X_test)
