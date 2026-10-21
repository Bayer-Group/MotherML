import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from optuna.trial import FixedTrial
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pytest.importorskip("torch")
pytest.importorskip("skorch")

import torch
from skorch import NeuralNet
from torch import nn

from mother import ml
from mother.ml.core import PipelineWithHyperparameterRooting
from mother.ml.models.m_tabm import (
    TabMClassifierMother,
    TabMRegressorMother,
    _EnsembleLoss,
)
from mother.optimization.core import MotherTuner
from mother.pipeline_utils import get_model, mother_cv
from mother.settings import MotherSettings


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    return X, (2 * X["a"] - X["b"]).to_numpy(dtype=np.float32)


@pytest.fixture
def small_params():
    return {"n_blocks": 1, "d_block": 16, "k": 4, "max_epochs": 2, "batch_size": 32}


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_clone_and_parameters(estimator, small_params):
    model = estimator(**small_params, cat_features=["category"])
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    cloned.set_params(lr=0.01)
    assert cloned.lr == 0.01
    assert model.lr != cloned.lr
    assert is_classifier(model) == (estimator is TabMClassifierMother)
    assert is_regressor(model) == (estimator is TabMRegressorMother)
    with pytest.raises(NotFittedError):
        model.predict(np.ones((2, 3), dtype=np.float32))


@pytest.mark.parametrize("as_frame", [False, True])
def test_regression_fit_prediction_and_serialization(data, small_params, as_frame):
    X, y = data
    X = X if as_frame else X.to_numpy(dtype=np.float32)
    model = TabMRegressorMother(**small_params).fit(X, y)
    assert isinstance(model.net_, NeuralNet)
    predictions = model.predict(X[:5])
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(X[:5]), predictions)
    if as_frame:
        assert list(model.feature_names_in_) == list(X.columns)
    result = model.predict_uncertainty(X[:5])
    assert list(result.columns) == [
        "pred",
        "mean_predictions",
        "knowledge_uncertainty",
        "data_uncertainty",
        "total_uncertainty",
    ]
    np.testing.assert_allclose(result["pred"], predictions)
    np.testing.assert_allclose(result["mean_predictions"], predictions)
    assert np.isfinite(result["knowledge_uncertainty"]).all()
    assert (result["knowledge_uncertainty"] >= 0).all()
    assert result["data_uncertainty"].isna().all()
    assert result["total_uncertainty"].isna().all()
    pd.testing.assert_frame_equal(
        model.predict_uncertainty(X[:5], uncertainty_for_opt=True), result[["knowledge_uncertainty"]]
    )


@pytest.mark.parametrize("values, expected_std", [([1.0, 3.0], 1.0), ([2.0, 2.0], 0.0), ([2.0], 0.0)])
@pytest.mark.parametrize("as_frame", [False, True])
def test_regression_uncertainty_population_std_and_target_scale(
    data, small_params, monkeypatch, values, expected_std, as_frame
):
    X, y = data
    model = TabMRegressorMother(**small_params).fit(X, y)
    members = torch.tensor([[values], [values]], dtype=torch.float32).transpose(1, 2)
    calls = []

    def predict_members(X):
        calls.append(len(X))
        return members

    monkeypatch.setattr(model, "_member_predictions", predict_members)
    inputs = X.iloc[[4, 1]] if as_frame else X.iloc[[4, 1]].to_numpy()
    result = model.predict_uncertainty(inputs)
    assert calls == [2]
    expected_members = model.target_scaler_.inverse_transform(np.array(values)[:, None]).ravel()
    np.testing.assert_allclose(result["knowledge_uncertainty"], expected_std * model.target_scaler_.scale_[0])
    np.testing.assert_allclose(result["knowledge_uncertainty"], expected_members.std(ddof=0))
    np.testing.assert_allclose(result["mean_predictions"], expected_members.mean(), rtol=1e-6)
    np.testing.assert_array_equal(result["pred"], model.predict(inputs))
    assert result.index.tolist() == ([4, 1] if as_frame else [0, 1])
    assert result[["data_uncertainty", "total_uncertainty"]].isna().all().all()


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classification_labels_probabilities_and_serialization(data, small_params, n_classes):
    X, y = data
    labels = np.array(["low", "medium", "high"][:n_classes])
    targets = labels[np.arange(len(y)) % n_classes]
    model = TabMClassifierMother(**small_params).fit(X, targets)
    probabilities = model.predict_proba(X[:5])
    assert probabilities.shape == (5, n_classes)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_array_equal(model.predict(X[:5]), model.classes_[probabilities.argmax(axis=1)])
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict_proba(X[:5]), probabilities)
    result = model.predict_uncertainty(X[:5])
    assert all(f"proba_{i}" in result for i in range(n_classes))
    assert list(result.columns) == ["pred"] + [f"proba_{i}" for i in range(n_classes)] + [
        "mean_predictions",
        "knowledge_uncertainty",
        "data_uncertainty",
        "total_uncertainty",
    ]
    pd.testing.assert_index_equal(result.index, X[:5].index)
    np.testing.assert_array_equal(result["pred"], model.predict(X[:5]))
    np.testing.assert_allclose(result[[f"proba_{i}" for i in range(n_classes)]], probabilities)
    np.testing.assert_allclose(
        result["total_uncertainty"], result["data_uncertainty"] + result["knowledge_uncertainty"], atol=1e-12
    )
    terms = result[["knowledge_uncertainty", "data_uncertainty", "total_uncertainty"]]
    assert np.isfinite(terms).all().all()
    assert (terms >= -1e-12).all().all()
    assert (terms <= np.log(n_classes) + 1e-6).all().all()
    pd.testing.assert_frame_equal(
        model.predict_uncertainty(X[:5], uncertainty_for_opt=True), result[["knowledge_uncertainty"]]
    )


@pytest.mark.parametrize(
    "member_logits, expected",
    [
        ([[1000.0, -1000.0], [1000.0, -1000.0]], (0.0, 0.0, 0.0)),
        ([[1000.0, -1000.0], [-1000.0, 1000.0]], (np.log(2), 0.0, np.log(2))),
        ([[0.0, 0.0], [0.0, 0.0]], (0.0, np.log(2), np.log(2))),
        ([[0.0, 0.0]], (0.0, np.log(2), np.log(2))),
        (
            [[1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0], [-1000.0, -1000.0, 1000.0]],
            (np.log(3), 0.0, np.log(3)),
        ),
    ],
)
def test_entropy_decomposition_known_cases(data, small_params, monkeypatch, member_logits, expected):
    X, _ = data
    logits = torch.tensor([member_logits])
    n_classes = logits.shape[-1]
    model = TabMClassifierMother(**small_params).fit(X, np.arange(len(X)) % n_classes)
    calls = []

    def predict_members(X):
        calls.append(len(X))
        return logits

    monkeypatch.setattr(model, "_member_predictions", predict_members)
    result = model.predict_uncertainty(X.iloc[[3]])
    assert calls == [1]
    actual = result[["knowledge_uncertainty", "data_uncertainty", "total_uncertainty"]].to_numpy()[0]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert result["mean_predictions"].isna().all()
    assert result.index.tolist() == [3]
    array_result = model.predict_uncertainty(X.iloc[[3]].to_numpy())
    assert array_result.index.tolist() == [0]


@pytest.mark.parametrize("integer_categories", [False, True])
@pytest.mark.parametrize("categorical_only", [False, True])
def test_declared_categories_and_unseen_values(data, small_params, integer_categories, categorical_only):
    X, y = data
    categories = [0, 1, 2] if integer_categories else ["red", "green", "blue"]
    X["category"] = np.tile(categories, len(X) // 3)
    if categorical_only:
        X = X[["category"]]
    model = TabMRegressorMother(**small_params, cat_features=["category"]).fit(X, y)
    unseen = X.iloc[:5].copy()
    unseen["category"] = 100 if integer_categories else "unseen"
    assert model.cat_cardinalities_ == [3]
    assert model.predict(unseen).shape == (5,)
    assert np.isfinite(model.predict(unseen)).all()
    assert list(model.encoder_.categories_[0]) == sorted(categories)


def test_categorical_indices_for_arrays(data, small_params):
    X, y = data
    X["category"] = np.tile([1, 2, 3], len(X) // 3)
    model = TabMRegressorMother(**small_params, cat_features=[3]).fit(X.to_numpy(), y)
    assert model.numeric_indices_ == [0, 1, 2]
    assert model.predict(X.to_numpy()[:5]).shape == (5,)


def test_ensemble_loss_trains_each_member():
    predictions = torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]], requires_grad=True)
    targets = torch.tensor([[1.0], [2.0]])
    loss = _EnsembleLoss(classification=False)(predictions, targets)
    assert loss.item() == pytest.approx(1.0)
    assert nn.functional.mse_loss(predictions.mean(dim=1), targets).item() == 0
    loss.backward()
    assert (predictions.grad != 0).all()

    logits = torch.tensor([[[2.0, -1.0], [-1.0, 2.0]], [[0.0, 1.0], [1.0, 0.0]]])
    labels = torch.tensor([0, 1])
    expected = torch.stack([nn.functional.cross_entropy(logits[:, i], labels) for i in range(2)]).mean()
    torch.testing.assert_close(_EnsembleLoss(classification=True)(logits, labels), expected)


def test_prediction_aggregation(data, small_params, monkeypatch):
    X, y = data
    classifier = TabMClassifierMother(**small_params).fit(X, (y > 0).astype(int))
    logits = torch.tensor([[[4.0, 0.0], [0.0, 1.0]]])
    monkeypatch.setattr(classifier, "_member_predictions", lambda X: logits)
    expected = logits.softmax(dim=-1).mean(dim=1).numpy()
    np.testing.assert_allclose(classifier.predict_proba(X[:1]), expected)
    assert not np.allclose(expected, logits.mean(dim=1).softmax(dim=-1).numpy())

    regressor = TabMRegressorMother(**small_params).fit(X, y)
    members = torch.tensor([[[1.0], [3.0]]])
    monkeypatch.setattr(regressor, "_member_predictions", lambda X: members)
    expected_regression = regressor.target_scaler_.inverse_transform([[2.0]]).ravel()
    np.testing.assert_allclose(regressor.predict(X[:1]), expected_regression)


@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini"])
def test_training_learns_and_is_reproducible(data, arch_type):
    X, y = data
    params = dict(n_blocks=1, d_block=16, k=4, max_epochs=25, batch_size=60, lr=0.03, dropout=0, arch_type=arch_type)
    state = torch.random.get_rng_state().clone()
    first = TabMRegressorMother(**params).fit(X, y)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    second = TabMRegressorMother(**params).fit(X, y)
    np.testing.assert_allclose(first.predict(X), second.predict(X))
    assert first.score(X, y) > 0.8
    assert first.net_.history[-1]["train_loss"] < first.net_.history[0]["train_loss"]
    classifier = TabMClassifierMother(**params).fit(X, np.where(y > 0, "yes", "no"))
    assert accuracy_score(np.where(y > 0, "yes", "no"), classifier.predict(X)) > 0.85


def test_optimizer_weight_decay_groups(data, small_params):
    X, y = data
    model = TabMRegressorMother(**small_params, weight_decay=0.1).fit(X, y)
    groups = model.net_.optimizer_.param_groups
    assert [group["weight_decay"] for group in groups] == [0.1, 0.0]
    bias_ids = {id(p) for name, p in model.net_.module_.named_parameters() if name.endswith("bias")}
    assert bias_ids <= {id(p) for p in groups[1]["params"]}
    all_params = [id(p) for group in groups for p in group["params"]]
    assert len(all_params) == len(set(all_params))


@pytest.mark.parametrize("model_type", ["regression", "classification_binary", "classification_multiclass"])
def test_registry_and_settings_builder(small_params, model_type):
    settings = MotherSettings.create()
    settings.model.algorithm = "tabm"
    settings.model.model_type = model_type
    settings.model.parameters = small_params
    estimator = get_model(settings)
    expected = TabMRegressorMother if model_type == "regression" else TabMClassifierMother
    assert isinstance(estimator, expected)
    assert ml.get_model_class_by_algorithm_and_type("tabm", model_type) is expected
    assert estimator.get_params()["k"] == 4


def test_pipeline_cross_validation_and_tuning_contract(data, small_params):
    X, y = data
    model = TabMRegressorMother(**small_params)
    pipeline = PipelineWithHyperparameterRooting([("scale", StandardScaler()), ("model", model)])
    trial = FixedTrial({"model__" + name: value for name, value in model.default_parameters().items()})
    suggested = pipeline.get_hyperparameter_space(X, y, trial)
    assert suggested == pipeline.default_parameters()
    assert clone(pipeline).set_params(**suggested).get_params()["model__k"] == 32
    scores = cross_val_score(Pipeline([("model", model)]), X, y, cv=2, scoring="neg_mean_squared_error")
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_mother_cv(data, small_params):
    X, y = data
    result = mother_cv(
        TabMRegressorMother(**small_params),
        X=X,
        y=pd.Series(y, name="target"),
        cv=KFold(2, shuffle=True, random_state=42),
        return_estimators=False,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(X)
    assert result["pred_target"].notna().all()
    assert result["pred_knowledge_uncertainty"].notna().all()
    assert "pred_total_uncertainty" not in result


def test_classification_uncertainty_mother_cv(data, small_params):
    X, y = data
    result = mother_cv(
        TabMClassifierMother(**small_params),
        X=X,
        y=pd.Series(np.where(y > 0, "yes", "no"), name="label"),
        cv=KFold(2, shuffle=True, random_state=42),
    )
    assert len(result) == len(X)
    terms = result[["pred_knowledge_uncertainty", "pred_data_uncertainty", "pred_total_uncertainty"]]
    assert np.isfinite(terms).all().all()
    np.testing.assert_allclose(terms.iloc[:, 2], terms.iloc[:, 0] + terms.iloc[:, 1], atol=1e-12)


def test_mother_tuner(data, small_params):
    X, y = data
    pipeline = PipelineWithHyperparameterRooting([("model", TabMRegressorMother(**small_params))])
    tuner = MotherTuner(scorer="neg_mean_squared_error", n_trials_optuna=1, n_startup_trials=1)
    fitted = tuner.optimize(pipeline, X, pd.Series(y), cross_validation=KFold(2))
    assert fitted.predict(X[:5]).shape == (5,)
    assert tuner.study is not None
    assert len(tuner.study.trials) == 1


@pytest.mark.parametrize(
    "params, message",
    [
        ({"k": 0}, "k must"),
        ({"arch_type": "tabm-packed"}, "arch_type"),
        ({"dropout": 1}, "dropout"),
        ({"lr": float("nan")}, "lr must"),
        ({"weight_decay": -1}, "weight_decay"),
        ({"cat_features": ["missing"]}, "matching DataFrame"),
        ({"cat_features": [-1]}, "Invalid categorical"),
        ({"cat_features": ["a", "a"]}, "Duplicate categorical"),
    ],
)
def test_invalid_parameters(data, small_params, params, message):
    X, y = data
    with pytest.raises(ValueError, match=message):
        TabMRegressorMother(**(small_params | params)).fit(X, y)


def test_invalid_data_and_feature_schema(data, small_params):
    X, y = data
    model = TabMRegressorMother(**small_params)
    with pytest.raises(ValueError, match="1d array"):
        model.fit(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, np.full(len(y), np.nan))
    with pytest.raises(ValueError, match="same number"):
        model.fit(X, y[:-1])
    with pytest.raises(ValueError, match="at least two classes"):
        TabMClassifierMother(**small_params).fit(X, np.zeros(len(y)))
    invalid = X.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(invalid, y)
    categorical = X.assign(category="known")
    categorical.loc[0, "category"] = None
    with pytest.raises(ValueError, match="nonmissing"):
        TabMRegressorMother(**small_params, cat_features=["category"]).fit(categorical, y)
    model.fit(X, y)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(X[["c", "b", "a"]])
    with pytest.raises(ValueError):
        model.fit(invalid, y)
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_preprocessing_is_training_only_and_refit_resets(data, small_params):
    X, y = data
    model = TabMRegressorMother(**small_params).fit(X, y)
    np.testing.assert_allclose(model.scaler_.mean_, X.mean(), atol=1e-6)
    predictions = model.predict(X[:5])
    model.predict(X + 1000)
    np.testing.assert_allclose(model.predict(X[:5]), predictions)
    model.fit(X + 100, y)
    np.testing.assert_allclose(model.scaler_.mean_, X.mean() + 100, atol=1e-5)


def test_optional_skorch_dependency_does_not_break_other_models():
    code = """
import importlib.abc
import sys

class BlockSkorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "skorch":
            raise ModuleNotFoundError("skorch is unavailable", name="skorch")

sys.meta_path.insert(0, BlockSkorch())
from mother import ml
from mother.errors import ExtrasDependencyImportError
assert ml.algo_is_supported("catboost")
assert not ml.algo_is_supported("tabm")
try:
    from mother.ml.models.m_tabm import TabMRegressorMother
except ExtrasDependencyImportError as error:
    assert "mother-ml[tabm]" in str(error)
else:
    raise AssertionError("Expected a missing-extra error")
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_cuda_training_and_rng_restoration(data, small_params, estimator):
    X, y = data
    if estimator is TabMClassifierMother:
        y = np.where(y > 0, "yes", "no")
    device = torch.cuda.current_device()
    gpu_state = torch.cuda.get_rng_state(device).clone()
    cpu_state = torch.random.get_rng_state().clone()
    model = estimator(**small_params, device=f"cuda:{device}").fit(X, y)
    assert model.predict(X[:5]).shape == (5,)
    if estimator is TabMClassifierMother:
        assert np.isfinite(model.predict_proba(X[:5])).all()
        uncertainty = model.predict_uncertainty(X[:5])
        assert np.isfinite(uncertainty["knowledge_uncertainty"]).all()
    else:
        assert np.isfinite(model.predict(X[:5])).all()
        uncertainty = model.predict_uncertainty(X[:5])
        assert np.isfinite(uncertainty["knowledge_uncertainty"]).all()
    assert next(model.net_.module_.parameters()).device.type == "cuda"
    torch.testing.assert_close(torch.cuda.get_rng_state(device), gpu_state)
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_state)
    estimator(**small_params).fit(X, y)
    torch.testing.assert_close(torch.cuda.get_rng_state(device), gpu_state)
