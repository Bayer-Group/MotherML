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
from sklearn.model_selection import KFold, cross_val_score, cross_validate
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
    _TabMNet,
)
from mother.ml.models.tabm_utils import PiecewiseLinearEmbeddings
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


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
@pytest.mark.parametrize("num_emb_type", ["none", "pwl"])
@pytest.mark.parametrize("as_frame", [False, True])
def test_feature_scaling_preserves_large_offset_variation(small_params, estimator, num_emb_type, as_frame):
    values = np.array([[100000001.0], [100000002.0], [100000003.0]])
    X = pd.DataFrame(values, columns=["offset"]) if as_frame else values
    model = estimator(**small_params, num_emb_type=num_emb_type).fit(X, np.array([0, 1, 0]))
    expected = StandardScaler().fit(values)
    np.testing.assert_array_equal(model.scaler_.mean_, expected.mean_)
    np.testing.assert_array_equal(model.scaler_.scale_, expected.scale_)
    prepared = model._prepare_features(X, fitting=False)["x_num"]
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, expected.transform(values).astype(np.float32), rtol=0, atol=0)
    assert np.unique(prepared).size == 3
    shifted = pd.DataFrame(values + 1, columns=["offset"]) if as_frame else values + 1
    np.testing.assert_allclose(
        model._prepare_features(shifted, fitting=False)["x_num"],
        expected.transform(values + 1).astype(np.float32),
        rtol=0,
        atol=0,
    )


def test_target_scaling_and_outputs_preserve_large_offset_variation(small_params, monkeypatch):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([100000001.0, 100000002.0, 100000003.0])
    expected_scaler = StandardScaler().fit(y[:, None])
    original_fit = _TabMNet.fit
    captured = {}

    def capture_fit(net, inputs, targets):
        captured["targets"] = targets.copy()
        return original_fit(net, inputs, targets)

    monkeypatch.setattr(_TabMNet, "fit", capture_fit)
    model = TabMRegressorMother(**small_params).fit(X, y)
    np.testing.assert_array_equal(model.target_scaler_.mean_, expected_scaler.mean_)
    np.testing.assert_array_equal(model.target_scaler_.scale_, expected_scaler.scale_)
    standardized = expected_scaler.transform(y[:, None]).astype(np.float32)
    assert captured["targets"].dtype == np.float32
    np.testing.assert_array_equal(captured["targets"], standardized)
    assert np.unique(captured["targets"]).size == 3

    members = torch.from_numpy(np.stack([standardized - 0.25, standardized + 0.25], axis=1))
    monkeypatch.setattr(model, "_member_predictions", lambda inputs: members)
    expected = expected_scaler.inverse_transform(members.double().mean(dim=1).numpy()).ravel()
    predictions = model.predict(X)
    assert predictions.dtype == np.float64
    np.testing.assert_allclose(predictions, expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(predictions, y, rtol=0, atol=1e-7)
    assert np.unique(predictions).size == 3
    result = model.predict_uncertainty(X)
    for name in ("pred", "mean_predictions"):
        assert result[name].dtype == np.float64
        np.testing.assert_array_equal(result[name], predictions)
    np.testing.assert_allclose(
        result["knowledge_uncertainty"],
        members.double().std(dim=1, correction=0).numpy().ravel() * expected_scaler.scale_[0],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini"])
@pytest.mark.parametrize("scale_features", [False, True])
def test_piecewise_embeddings(data, small_params, estimator, arch_type, scale_features):
    X, y = data
    if estimator is TabMClassifierMother:
        y = np.where(y > 0, "yes", "no")
    X = X.assign(category=np.tile(["red", "green", "blue"], len(X) // 3))
    model = estimator(
        **small_params,
        arch_type=arch_type,
        num_emb_type="pwl",
        d_embedding=5,
        num_emb_n_bins=8,
        cat_features=["category"],
        scale_features=scale_features,
    )
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    state = torch.random.get_rng_state().clone()
    model.fit(X, y)
    assert isinstance(model.net_.module_.num_module, PiecewiseLinearEmbeddings)
    numeric = X[["a", "b", "c"]].to_numpy(dtype=np.float32)
    if model.scaler_ is not None:
        numeric = model.scaler_.transform(numeric)
    expected = torch.quantile(torch.from_numpy(numeric), torch.linspace(0, 1, 9), dim=0).T
    for actual, edges in zip(model.bins_, expected):
        torch.testing.assert_close(actual, edges.unique())
    embedded = model.net_.module_.num_module(torch.from_numpy(numeric))
    assert embedded.shape == (len(X), 3, 5)
    original_bins = [edges.clone() for edges in model.bins_]
    model.predict(X.assign(a=X.a + 1000))
    for actual, edges in zip(model.bins_, original_bins):
        torch.testing.assert_close(actual, edges)
    predictions = model.predict(X[:5])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X[:5]), predictions)
    np.testing.assert_array_equal(cloned.fit(X, y).predict(X[:5]), predictions)
    result = model.predict_uncertainty(X[:5])
    assert np.isfinite(result["knowledge_uncertainty"]).all()
    if estimator is TabMClassifierMother:
        np.testing.assert_allclose(model.predict_proba(X[:5]).sum(axis=1), 1, atol=1e-6)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    model.set_params(num_emb_type="none").fit(X, y)
    assert model.bins_ is None
    assert model.net_.module_.num_module is None


def test_piecewise_bins_are_fold_local_and_capped(data, small_params):
    X, y = data
    model = TabMRegressorMother(**small_params, num_emb_type="pwl", num_emb_n_bins=48)
    splitter = KFold(2)
    results = cross_validate(model, X, y, cv=splitter, return_estimator=True)
    for fitted, (train, _) in zip(results["estimator"], splitter.split(X)):
        numeric = fitted.scaler_.transform(X.iloc[train].to_numpy(dtype=np.float32))
        expected = torch.quantile(torch.from_numpy(numeric), torch.linspace(0, 1, len(train)), dim=0).T
        for actual, edges in zip(fitted.bins_, expected):
            torch.testing.assert_close(actual, edges.unique())
    model.fit(X.iloc[:3], y[:3])
    assert all(len(edges) <= 3 for edges in model.bins_)
    model.fit(X + 100, y)
    np.testing.assert_allclose(model.scaler_.mean_, X.mean() + 100, atol=1e-5)


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini"])
@pytest.mark.parametrize("num_emb_type", ["none", "pwl", "linear", "linear_relu", "periodic"])
def test_automatic_depth(data, small_params, estimator, arch_type, num_emb_type):
    X, y = data
    if estimator is TabMClassifierMother:
        y = (y > 0).astype(int)
    assert estimator().n_blocks is None
    model = estimator(**(small_params | {"n_blocks": None}), arch_type=arch_type, num_emb_type=num_emb_type)
    model.fit(X, y)
    expected_depth = 2 if num_emb_type == "pwl" else 3
    assert model.n_blocks is None
    assert model.n_blocks_ == expected_depth
    assert len(model.net_.module_.backbone.blocks) == expected_depth
    assert model.predict(X[:5]).shape == (5,)
    assert clone(model).n_blocks is None
    restored = pickle.loads(pickle.dumps(model))
    assert restored.n_blocks_ == expected_depth
    np.testing.assert_array_equal(restored.predict(X[:5]), model.predict(X[:5]))

    model.set_params(n_blocks=4).fit(X, y)
    assert model.n_blocks_ == 4
    assert len(model.net_.module_.backbone.blocks) == 4
    model.set_params(n_blocks=None, num_emb_type="none" if num_emb_type == "pwl" else "pwl").fit(X, y)
    assert model.n_blocks_ == (3 if num_emb_type == "pwl" else 2)


@pytest.mark.parametrize("n_blocks", [None, 2])
def test_tuning_overrides_depth(data, small_params, n_blocks):
    X, y = data
    model = TabMRegressorMother(**(small_params | {"n_blocks": n_blocks}), num_emb_type="pwl")
    trial = FixedTrial(model.default_parameters() | {"n_blocks": 4})
    suggested = model.get_hyperparameter_space(X, y, trial)
    assert suggested["n_blocks"] == 4
    assert trial.distributions["n_blocks"].low == 1
    assert trial.distributions["n_blocks"].high == 4
    fitted = clone(model).set_params(**suggested).fit(X, y)
    assert fitted.n_blocks_ == 4
    assert model.n_blocks == n_blocks


@pytest.mark.parametrize("num_emb_type", ["none", "pwl"])
def test_categorical_only_skips_numerical_embeddings(data, small_params, num_emb_type):
    X, y = data
    X = pd.DataFrame({"category": np.tile(["red", "green", "blue"], len(X) // 3)})
    model = TabMRegressorMother(
        **(small_params | {"n_blocks": None}),
        cat_features=["category"],
        num_emb_type=num_emb_type,
    ).fit(X, y)
    assert model.bins_ is None
    assert model.net_.module_.num_module is None
    assert model.n_blocks_ == 3
    assert np.isfinite(model.predict(X)).all()


def test_piecewise_embeddings_reject_unsupported_training_data(data, small_params):
    X, y = data
    model = TabMRegressorMother(**small_params, num_emb_type="pwl")
    with pytest.raises(ValueError, match="at least 3 training samples"):
        model.fit(X.iloc[:2], y[:2])
    with pytest.raises(ValueError, match="two distinct values"):
        model.fit(X.assign(a=1), y)
    with pytest.raises(NotFittedError):
        model.predict(X)
    # Constant numeric columns remain supported with embeddings disabled.
    model.set_params(num_emb_type="none").fit(X.assign(a=1), y)
    assert np.isfinite(model.predict(X.assign(a=1))).all()


def test_auto_embeddings_require_tuning(data, small_params):
    X, y = data
    model = TabMRegressorMother(**small_params, num_emb_type="auto")
    with pytest.raises(ValueError, match="only supported during MotherTuner"):
        model.fit(X, y)


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


@pytest.mark.parametrize("classification", [False, True])
def test_masked_multitarget_loss_and_gradients(classification):
    predictions = torch.tensor(
        [[[0.2, 1.0, 2.0], [0.8, 2.0, 3.0]], [[0.3, 3.0, 4.0], [0.7, 4.0, 5.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[1.0, 0.0, float("nan")], [0.0, float("nan"), float("nan")]])
    loss = _EnsembleLoss(classification=classification, multitarget=True)(predictions, targets)
    task_losses = []
    for task in range(2):
        observed = targets[:, task].isfinite()
        expected_targets = targets[observed, task, None].expand(-1, 2)
        if classification:
            task_losses.append(
                nn.functional.binary_cross_entropy_with_logits(predictions[observed, :, task], expected_targets)
            )
        else:
            task_losses.append(nn.functional.mse_loss(predictions[observed, :, task], expected_targets))
    torch.testing.assert_close(loss, torch.stack(task_losses).mean())
    loss.backward()
    assert predictions.grad.isfinite().all()
    assert (predictions.grad[:, :, 2] == 0).all()
    assert (predictions.grad[1, :, 1] == 0).all()
    assert (predictions.grad[:, :, 0] != 0).all()
    with pytest.raises(ValueError, match="at least one observed"):
        _EnsembleLoss(classification=classification, multitarget=True)(predictions, torch.full((2, 3), float("nan")))


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini"])
def test_multitarget_missing_labels(data, small_params, estimator, as_frame, arch_type, caplog):
    X, y = data
    targets = np.column_stack([y.astype(np.float64) + 1e8, y * 100])
    if estimator is TabMClassifierMother:
        targets = np.column_stack([(y > 0).astype(float), (y > np.median(y)).astype(float)])
    targets[::3, 0] = np.nan
    targets[1::3, 1] = np.nan
    targets[-1] = np.nan
    input_targets = pd.DataFrame(targets, columns=["task_a", "task_b"]) if as_frame else targets
    model = estimator(**small_params, target_type="multi_target", num_emb_type="pwl", arch_type=arch_type).fit(
        X, input_targets
    )
    assert "no observed target labels" in caplog.text
    observed_rows = np.isfinite(targets).any(axis=1)
    np.testing.assert_allclose(model.scaler_.mean_, X.loc[observed_rows].mean(), atol=1e-12)
    predictions = model.predict(X)
    assert predictions.shape == targets.shape
    assert np.isfinite(predictions).all()
    assert np.isfinite(model.score(X, input_targets))
    assert model.net_.module_.output.weight.shape[-1] == 2
    assert np.isfinite(model.net_.history[-1]["train_loss"])
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), predictions)
    result = model.predict_uncertainty(X)
    pd.testing.assert_index_equal(result.index, X.index)
    for task in range(2):
        np.testing.assert_array_equal(result[f"target_{task}_pred"], predictions[:, task])
        assert np.isfinite(result[f"target_{task}_knowledge_uncertainty"]).all()
    assert model.predict_uncertainty(X, uncertainty_for_opt=True).columns.tolist() == [
        "target_0_knowledge_uncertainty",
        "target_1_knowledge_uncertainty",
    ]
    if estimator is TabMRegressorMother:
        np.testing.assert_allclose(model.target_scaler_.mean_, np.nanmean(targets, axis=0), rtol=0, atol=1e-8)
        np.testing.assert_allclose(model.target_scaler_.scale_, np.nanstd(targets, axis=0), atol=1e-7)
        assert predictions.dtype == np.float64
        for task in range(2):
            assert result[f"target_{task}_data_uncertainty"].isna().all()
    else:
        probabilities = model.predict_proba(X)
        assert isinstance(probabilities, list)
        assert len(probabilities) == 2
        assert len(model.classes_) == 2
        for task in range(2):
            assert probabilities[task].shape == (len(X), 2)
            np.testing.assert_allclose(probabilities[task].sum(axis=1), 1, atol=1e-6)
            np.testing.assert_array_equal(
                predictions[:, task], model.classes_[task][probabilities[task].argmax(axis=1)]
            )
            np.testing.assert_allclose(
                result[f"target_{task}_total_uncertainty"],
                result[f"target_{task}_knowledge_uncertainty"] + result[f"target_{task}_data_uncertainty"],
                atol=1e-12,
            )
    # Refit with another target layout must not retain task-specific state.
    single_targets = (y > 0).astype(int) if estimator is TabMClassifierMother else y
    model.set_params(target_type="single_target").fit(X, single_targets)
    assert model.predict(X).shape == (len(X),)


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_single_target_missing_labels(data, small_params, estimator):
    X, y = data
    targets = y.astype(np.float64) if estimator is TabMRegressorMother else (y > 0).astype(float)
    targets[::3] = np.nan
    model = estimator(**small_params).fit(X, targets)
    assert model.predict(X).shape == (len(X),)
    assert np.isfinite(model.score(X, targets))
    assert np.isfinite(model.net_.history[-1]["train_loss"])


def test_missing_multiclass_labels_and_masked_loss(data, small_params):
    X, y = data
    targets = (np.arange(len(y)) % 3).astype(float)
    targets[::4] = np.nan
    model = TabMClassifierMother(**small_params).fit(X, targets)
    assert model.predict_proba(X).shape == (len(X), 3)
    assert np.isfinite(model.score(X, targets))
    logits = torch.tensor([[[2.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    loss = _EnsembleLoss(classification=True)(logits, torch.tensor([0, -1]))
    torch.testing.assert_close(loss, nn.functional.cross_entropy(logits[0], torch.tensor([0, 0])))
    loss.backward()
    assert (logits.grad[1] == 0).all()


@pytest.mark.parametrize("labels", [np.array(["a", "b", "c"]), np.array([0, 1, 2])])
def test_classification_score_with_unseen_labels(data, small_params, labels):
    X, y = data
    model = TabMClassifierMother(**small_params).fit(X, labels[(y > 0).astype(int)])
    predictions = model.predict(X)
    targets = pd.Series(predictions).astype(object)
    targets.iloc[0] = labels[2]
    targets.iloc[1] = None
    weights = np.arange(1, len(X) + 1)
    expected = weights[2:].sum() / (weights[0] + weights[2:].sum())
    assert model.score(X, targets, sample_weight=weights) == pytest.approx(expected)


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_multitarget_one_column_shapes(data, small_params, estimator):
    X, y = data
    targets = y[:, None] if estimator is TabMRegressorMother else (y[:, None] > 0).astype(float)
    model = estimator(**small_params, target_type="multi_target").fit(X, targets)
    assert model.predict(X).shape == (len(X), 1)
    assert model.predict_uncertainty(X, uncertainty_for_opt=True).shape == (len(X), 1)
    if estimator is TabMClassifierMother:
        probabilities = model.predict_proba(X)
        assert isinstance(probabilities, list)
        assert len(probabilities) == 1
        assert probabilities[0].shape == (len(X), 2)


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
@pytest.mark.parametrize("as_frame", [True, False])
def test_multitarget_one_task_cv_and_tuning(data, small_params, estimator, as_frame):
    X, y = data
    targets = y.astype(float) if estimator is TabMRegressorMother else (y > 0).astype(float)
    targets[::4] = np.nan
    labels = pd.DataFrame({"task": targets}) if as_frame else pd.Series(targets, name="task")
    model = estimator(**small_params, target_type="multi_target")
    result = mother_cv(model, X=X, y=labels, cv=KFold(2))
    assert len(result) == len(X)
    assert result["pred_target_0_pred"].notna().all()
    pipeline = PipelineWithHyperparameterRooting([("model", model)])
    tuner = MotherTuner(
        scorer=lambda fitted, features, values: fitted.score(features, values),
        n_trials_optuna=1,
        n_startup_trials=1,
    )
    fitted = tuner.optimize(pipeline, X, labels, cross_validation=KFold(2))
    assert fitted.predict(X).shape == (len(X), 1)
    assert np.isfinite(fitted.score(X, labels))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_multitarget_cuda_missing_labels(data, small_params, estimator):
    X, y = data
    targets = np.column_stack([y, -y]).astype(np.float64)
    if estimator is TabMClassifierMother:
        targets = (targets > 0).astype(float)
    targets[::3, 0] = np.nan
    targets[1::3, 1] = np.nan
    state = torch.cuda.get_rng_state().clone()
    model = estimator(**small_params, target_type="multi_target", device="cuda").fit(X, targets)
    assert model.predict(X).shape == targets.shape
    assert np.isfinite(model.score(X, targets))
    torch.testing.assert_close(torch.cuda.get_rng_state(), state)


def test_multitarget_string_labels_and_nullable_targets(data, small_params):
    X, y = data
    targets = pd.DataFrame(
        {
            "first": pd.Series(np.where(y > 0, "yes", "no"), dtype="string"),
            "second": pd.Series((y > np.median(y)).astype(int), dtype="Int64"),
        }
    )
    targets.iloc[::3, 0] = pd.NA
    targets.iloc[1::3, 1] = pd.NA
    model = TabMClassifierMother(**small_params, target_type="multi_target").fit(X, targets)
    predictions = model.predict(X)
    assert set(predictions[:, 0]) <= {"yes", "no"}
    assert set(predictions[:, 1]) <= {0, 1}
    assert np.isfinite(model.score(X, targets))
    regression_targets = pd.DataFrame({"a": pd.Series(y, dtype="Float64"), "b": y * 2})
    regression_targets.iloc[::3, 0] = pd.NA
    regressor = TabMRegressorMother(**small_params, target_type="multi_target").fit(X, regression_targets)
    assert np.isfinite(regressor.predict(X)).all()


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_multitarget_invalid_labels(data, small_params, estimator):
    X, y = data
    model = estimator(**small_params, target_type="multi_target")
    with pytest.raises(ValueError, match="2D"):
        model.fit(X, np.zeros((len(X), 1, 1)))
    with pytest.raises(ValueError, match="observed"):
        model.fit(X, np.column_stack([y, np.full(len(y), np.nan)]))
    targets = np.column_stack([(y > 0).astype(float), (y > 0).astype(float)])
    targets[0, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, targets)
    if estimator is TabMClassifierMother:
        with pytest.raises(ValueError, match="only binary"):
            model.fit(X, np.column_stack([np.arange(len(y)) % 3, np.arange(len(y)) % 2]))
        with pytest.raises(ValueError, match="at least two classes"):
            model.fit(X, np.column_stack([np.zeros(len(y)), np.arange(len(y)) % 2]))


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_multitarget_settings_cv_and_tuning(data, small_params, estimator):
    X, y = data
    targets = np.column_stack([y, y * 2]).astype(np.float64)
    if estimator is TabMClassifierMother:
        targets = (targets > 0).astype(float)
    targets[::4, 0] = np.nan
    targets[1::4, 1] = np.nan
    targets = pd.DataFrame(targets, columns=["a", "b"])
    settings = MotherSettings.create()
    settings.model.algorithm = "tabm"
    settings.model.model_type = "regression" if estimator is TabMRegressorMother else "classification_binary"
    settings.model.target_type = "multi_target"
    settings.model.parameters = small_params
    model = get_model(settings)
    assert isinstance(model, estimator)
    assert model.target_type == "multi_target"
    assert model.__sklearn_tags__().target_tags.multi_output
    scores = cross_val_score(model, X, targets, cv=KFold(2))
    assert np.isfinite(scores).all()
    result = mother_cv(model, X=X, y=targets, cv=KFold(2))
    assert len(result) == len(X)
    for task in range(2):
        assert result[f"pred_target_{task}_pred"].notna().all()
        assert result[f"pred_target_{task}_knowledge_uncertainty"].notna().all()
    pipeline = PipelineWithHyperparameterRooting([("model", model)])
    tuner = MotherTuner(
        scorer=lambda fitted, features, labels: fitted.score(features, labels),
        n_trials_optuna=1,
        n_startup_trials=1,
    )
    fitted = tuner.optimize(pipeline, X, targets, cross_validation=KFold(2))
    assert fitted.predict(X[:5]).shape == (5, 2)
    assert np.isfinite(fitted.score(X, targets))
    settings.model.model_type = "classification_multiclass"
    with pytest.raises(ValueError, match="only binary"):
        get_model(settings)


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_multitarget_learning_with_missing_labels(data, estimator):
    X, y = data
    targets = np.column_stack([y, -y]).astype(np.float64)
    if estimator is TabMClassifierMother:
        targets = (targets > 0).astype(float)
    targets[::3, 0] = np.nan
    targets[1::3, 1] = np.nan
    model = estimator(
        target_type="multi_target",
        n_blocks=1,
        d_block=32,
        k=4,
        max_epochs=50,
        batch_size=60,
        lr=0.03,
        dropout=0,
    ).fit(X, targets)
    assert model.score(X, targets) > 0.8
    assert model.net_.history[-1]["train_loss"] < model.net_.history[0]["train_loss"]


def test_multitarget_uncertainty_known_predictions(data, small_params, monkeypatch):
    X, y = data
    targets = np.column_stack([y, y * 100 + 1e8])
    model = TabMRegressorMother(**small_params, target_type="multi_target").fit(X, targets)
    members = torch.tensor([[[1.0, -2.0], [3.0, 2.0]]], dtype=torch.float32)
    monkeypatch.setattr(model, "_member_predictions", lambda inputs: members)
    result = model.predict_uncertainty(X.iloc[:1])
    expected = model.target_scaler_.inverse_transform(np.array([[2.0, 0.0]]))
    for task, spread in enumerate([1.0, 2.0]):
        assert result[f"target_{task}_pred"].iloc[0] == pytest.approx(expected[0, task])
        assert result[f"target_{task}_knowledge_uncertainty"].iloc[0] == pytest.approx(
            spread * model.target_scaler_.scale_[task]
        )
    classifier = TabMClassifierMother(**small_params, target_type="multi_target").fit(X, (targets > targets.mean(0)))
    logits = torch.tensor([[[10.0, 0.0], [-10.0, 0.0]]])
    monkeypatch.setattr(classifier, "_member_predictions", lambda inputs: logits)
    result = classifier.predict_uncertainty(X.iloc[:1])
    assert result["target_0_knowledge_uncertainty"].iloc[0] > 0.69
    assert result["target_0_data_uncertainty"].iloc[0] < 0.001
    assert result["target_1_knowledge_uncertainty"].iloc[0] == pytest.approx(0)
    assert result["target_1_data_uncertainty"].iloc[0] == pytest.approx(np.log(2))


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
    settings.model.parameters = small_params | {"num_emb_type": "pwl", "d_embedding": 8, "num_emb_n_bins": 12}
    estimator = get_model(settings)
    expected = TabMRegressorMother if model_type == "regression" else TabMClassifierMother
    assert isinstance(estimator, expected)
    assert ml.get_model_class_by_algorithm_and_type("tabm", model_type) is expected
    assert estimator.get_params()["k"] == 4
    assert estimator.model_type == model_type
    assert estimator.num_emb_type == "pwl"
    assert estimator.d_embedding == 8
    assert estimator.num_emb_n_bins == 12


def test_model_type_validation(data, small_params):
    X, y = data
    assert TabMRegressorMother(**small_params, model_type="regression").fit(X, y).model_type == "regression"
    with pytest.raises(ValueError, match="model_type for TabMRegressorMother"):
        TabMRegressorMother(**small_params, model_type="classification_binary").fit(X, y)
    with pytest.raises(ValueError, match="model_type for TabMClassifierMother"):
        TabMClassifierMother(**small_params, model_type="regression").fit(X, (y > 0).astype(int))
    multiclass = np.arange(len(X)) % 3
    with pytest.raises(ValueError, match="requires two classes"):
        TabMClassifierMother(**small_params, model_type="classification_binary").fit(X, multiclass)
    TabMClassifierMother(**small_params, model_type="classification_multiclass").fit(X, multiclass)
    with pytest.raises(ValueError, match="only binary"):
        TabMClassifierMother(**small_params, model_type="classification_multiclass", target_type="multi_target").fit(
            X, np.column_stack([y > 0, y > 1])
        )


@pytest.mark.parametrize("num_emb_type", ["none", "pwl", "auto"])
def test_pipeline_cross_validation_and_tuning_contract(data, small_params, num_emb_type):
    X, y = data
    model = TabMRegressorMother(**small_params, num_emb_type=num_emb_type)
    pipeline = PipelineWithHyperparameterRooting([("scale", StandardScaler()), ("model", model)])
    trial = FixedTrial({"model__" + name: value for name, value in model.default_parameters().items()})
    suggested = pipeline.get_hyperparameter_space(X, y, trial)
    assert suggested == pipeline.default_parameters()
    assert clone(pipeline).set_params(**suggested).get_params()["model__k"] == 32
    if num_emb_type == "pwl":
        assert {"model__d_embedding", "model__num_emb_n_bins"} <= suggested.keys()
        assert "model__num_emb_type" not in suggested
    elif num_emb_type == "auto":
        assert suggested["model__num_emb_type"] == "pwl"
        assert {"model__d_embedding", "model__num_emb_n_bins"} <= suggested.keys()
    else:
        assert {"model__d_embedding", "model__num_emb_n_bins", "model__num_emb_type"}.isdisjoint(suggested)
    cv_model = clone(model).set_params(num_emb_type="pwl") if num_emb_type == "auto" else model
    scores = cross_val_score(Pipeline([("model", cv_model)]), X, y, cv=2, scoring="neg_mean_squared_error")
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
def test_embedding_tuning_ranges_and_nondefault_values(data, small_params, estimator):
    X, y = data
    if estimator is TabMClassifierMother:
        y = (y > 0).astype(int)
    model = estimator(**small_params, num_emb_type="pwl")
    values = model.default_parameters() | {"d_embedding": 7, "num_emb_n_bins": 13}
    trial = FixedTrial(values)
    suggested = model.get_hyperparameter_space(X, y, trial)
    assert suggested == values
    for name, low, high in [("d_embedding", 4, 64), ("num_emb_n_bins", 2, 128)]:
        distribution = trial.distributions[name]
        assert distribution.low == low
        assert distribution.high == high
        assert distribution.log
    fitted = clone(model).set_params(**suggested).fit(X, y)
    assert fitted.net_.module_.num_module(torch.zeros(2, 3)).shape == (2, 3, 7)
    assert all(len(edges) <= 14 for edges in fitted.bins_)
    assert fitted.predict(X[:5]).shape == (5,)


@pytest.mark.parametrize(
    "embedding_type, extra_params, expected_module",
    [
        ("linear", {}, "LinearEmbeddings"),
        ("linear_relu", {}, "LinearReLUEmbeddings"),
        (
            "periodic",
            {
                "num_emb_n_frequencies": 12,
                "num_emb_frequency_init_scale": 0.02,
                "num_emb_activation": False,
            },
            "PeriodicEmbeddings",
        ),
        ("pwl", {"num_emb_n_bins": 12}, "PiecewiseLinearEmbeddings"),
    ],
)
def test_auto_embedding_type_tuning(data, small_params, embedding_type, extra_params, expected_module):
    X, y = data
    model = TabMRegressorMother(**small_params, num_emb_type="auto")
    defaults = model.default_parameters()
    values = {name: defaults[name] for name in ("lr", "weight_decay", "n_blocks", "d_block", "dropout", "k")}
    values.update({"num_emb_type": embedding_type, "d_embedding": 7} | extra_params)

    trial = FixedTrial(values)
    suggested = model.get_hyperparameter_space(X, y, trial)
    assert suggested["num_emb_type"] == embedding_type
    assert suggested["d_embedding"] == 7
    assert trial.distributions["num_emb_type"].choices == ("none", "linear", "linear_relu", "periodic", "pwl")
    fitted = clone(model).set_params(**suggested).fit(X, y)
    assert fitted.net_.module_.num_module.__class__.__name__ == expected_module
    assert fitted.predict(X[:5]).shape == (5,)

    if embedding_type == "periodic":
        assert suggested["num_emb_n_frequencies"] == 12
        assert suggested["num_emb_frequency_init_scale"] == 0.02
        assert suggested["num_emb_activation"] is False
        assert trial.distributions["num_emb_n_frequencies"].log
        assert trial.distributions["num_emb_frequency_init_scale"].log
    elif embedding_type == "pwl":
        assert suggested["num_emb_n_bins"] == 12
    else:
        assert "num_emb_n_bins" not in suggested


def test_auto_embedding_tuning_can_select_none(data, small_params):
    X, y = data
    model = TabMRegressorMother(**small_params, num_emb_type="auto")
    defaults = model.default_parameters()
    values = {name: defaults[name] for name in ("lr", "weight_decay", "n_blocks", "d_block", "dropout", "k")}
    trial = FixedTrial(values | {"num_emb_type": "none"})
    suggested = model.get_hyperparameter_space(X, y, trial)
    assert suggested["num_emb_type"] == "none"
    assert {"d_embedding", "num_emb_n_bins", "num_emb_n_frequencies"}.isdisjoint(suggested)
    fitted = clone(model).set_params(**suggested).fit(X, y)
    assert fitted.net_.module_.num_module is None
    assert fitted.predict(X[:5]).shape == (5,)


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


@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini", "auto"])
@pytest.mark.parametrize("num_emb_type", ["none", "pwl", "auto"])
def test_mother_tuner(data, small_params, num_emb_type, arch_type):
    X, y = data
    pipeline = PipelineWithHyperparameterRooting(
        [("model", TabMRegressorMother(**small_params, num_emb_type=num_emb_type, arch_type=arch_type))]
    )
    tuner = MotherTuner(scorer="neg_mean_squared_error", n_trials_optuna=1, n_startup_trials=1)
    fitted = tuner.optimize(pipeline, X, pd.Series(y), cross_validation=KFold(2))
    assert fitted.predict(X[:5]).shape == (5,)
    assert tuner.study is not None
    assert len(tuner.study.trials) == 1
    if arch_type == "auto":
        assert tuner.study.best_params["model__arch_type"] == "tabm"
    else:
        assert "model__arch_type" not in tuner.study.best_params
    assert fitted.named_steps["model"].arch_type == ("tabm" if arch_type == "auto" else arch_type)
    if num_emb_type in ("pwl", "auto"):
        if num_emb_type == "auto":
            assert tuner.study.best_params["model__num_emb_type"] == "pwl"
        assert tuner.study.best_params["model__d_embedding"] == 16
        assert tuner.study.best_params["model__num_emb_n_bins"] == 48
        assert fitted.named_steps["model"].net_.module_.num_module is not None


@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini"])
def test_auto_architecture_tuning(data, small_params, estimator, arch_type):
    X, y = data
    if estimator is TabMClassifierMother:
        y = (y > 0).astype(int)
    model = estimator(**small_params, arch_type="auto")
    assert clone(model).get_params() == model.get_params()
    with pytest.raises(ValueError, match="arch_type='auto'.*MotherTuner"):
        model.fit(X, y)

    pipeline = PipelineWithHyperparameterRooting([("model", model)])
    values = pipeline.default_parameters() | {"model__arch_type": arch_type}
    assert pipeline.default_parameters()["model__arch_type"] == "tabm"
    trial = FixedTrial(values)
    suggested = pipeline.get_hyperparameter_space(X, y, trial)
    assert suggested == values
    assert trial.distributions["model__arch_type"].choices == ("tabm", "tabm-mini")
    fitted = clone(pipeline).set_params(**suggested).fit(X, y)
    module = fitted.named_steps["model"].net_.module_
    assert module.arch_type == arch_type
    assert (module.minimal_ensemble_adapter is not None) == (arch_type == "tabm-mini")
    assert fitted.predict(X[:5]).shape == (5,)
    assert model.arch_type == "auto"


@pytest.mark.parametrize(
    "params, message",
    [
        ({"k": 0}, "k must"),
        ({"n_blocks": 0}, "n_blocks"),
        ({"n_blocks": True}, "n_blocks"),
        ({"n_blocks": 2.5}, "n_blocks"),
        ({"n_blocks": "auto"}, "n_blocks"),
        ({"arch_type": "tabm-packed"}, "arch_type"),
        ({"num_emb_type": "fourier"}, "num_emb_type"),
        ({"d_embedding": 0}, "d_embedding"),
        ({"d_embedding": True}, "d_embedding"),
        ({"num_emb_n_bins": 1}, "num_emb_n_bins"),
        ({"num_emb_n_bins": 2.5}, "num_emb_n_bins"),
        ({"num_emb_n_bins": True}, "num_emb_n_bins"),
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


def test_tabm_documentation_without_optional_dependencies():
    code = """
import importlib.abc
import pathlib
import runpy
import sys

class BlockSkorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "skorch":
            raise ModuleNotFoundError("skorch is unavailable", name="skorch")

sys.meta_path.insert(0, BlockSkorch())
checker = runpy.run_path("scripts/check_docs_python_fences.py")
blocks = checker["_extract_python_blocks"](pathlib.Path("docs/mother/models.md").read_text())
namespace = {}
skipped = 0
for block in blocks:
    try:
        exec(block, namespace, namespace)
    except Exception as error:
        if error.__class__.__name__ != "ExtrasDependencyImportError":
            raise
        skipped += 1
assert skipped >= 3
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("estimator", [TabMRegressorMother, TabMClassifierMother])
@pytest.mark.parametrize("num_emb_type", ["none", "pwl"])
def test_cuda_training_and_rng_restoration(data, small_params, estimator, num_emb_type):
    X, y = data
    if estimator is TabMClassifierMother:
        y = np.where(y > 0, "yes", "no")
    device = torch.cuda.current_device()
    gpu_state = torch.cuda.get_rng_state(device).clone()
    cpu_state = torch.random.get_rng_state().clone()
    model = estimator(**small_params, device=f"cuda:{device}", num_emb_type=num_emb_type).fit(X, y)
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
