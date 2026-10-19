"""Flow target scaling and original-unit inference regression tests."""

import copy
import pickle

import numpy as np
import pytest
from sklearn.base import clone

pytest.importorskip("skorch")
torch = pytest.importorskip("torch")

from skorch.dataset import ValidSplit  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

from mother.ml.models.m_node import NODERegressor  # noqa: E402
from mother.ml.models.node_head_utils import FlowHead  # noqa: E402

pytestmark = [pytest.mark.serial, pytest.mark.slow]


def _model(**kwargs):
    params = dict(
        head_type="flow",
        flow_type="NICE",
        num_trees=4,
        depth=2,
        max_epochs=1,
        batch_size=32,
        device="cpu",
        verbose=0,
    )
    params.update(kwargs)
    return NODERegressor(**params)


def _data(n_targets):
    features = np.random.default_rng(42).normal(size=(32, 3)).astype(np.float32)
    targets = 1000 + 50 * features[:, 0]
    if n_targets == 2:
        targets = np.column_stack([targets, -200 + 10 * features[:, 1]])
    return features, targets.astype(np.float32)


def _seed():
    torch.manual_seed(123)
    np.random.seed(123)


@pytest.fixture(scope="module", params=[(n, d, p) for n in (1, 2) for d in ("cpu", "cuda") for p in (0.0, 0.15)])
def fitted_pair(request):
    n_targets, device, dropout = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    features, targets = _data(n_targets)
    scaled = _model(device=device, input_dropout=dropout, tree_dropout=dropout).fit(features, targets)
    standardized = copy.deepcopy(scaled)
    del standardized.target_scaler_
    standardized.scale_targets = False
    return scaled, standardized, features[:3]


def test_flow_point_samples_and_quantiles_in_original_units(fitted_pair):
    scaled, standardized, features = fitted_pair
    mean, scale = scaled.target_scaler_.mean_, scaled.target_scaler_.scale_
    for method, kwargs in (
        ("predict", {}),
        ("predict_flow_head", {}),
        ("predict", {"return_sample_distribution": True}),
        ("predict_flow_head", {"return_sample_distribution": True}),
        ("predict_quantiles", {"quantiles": [0.1, 0.5, 0.9]}),
    ):
        _seed()
        expected = getattr(standardized, method)(features, num_samples=40, **kwargs)
        _seed()
        actual = getattr(scaled, method)(features, num_samples=40, **kwargs)
        if mean.size == 1:
            expected = expected * scale.item() + mean.item()
        else:
            expected = expected * scale + mean
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)


def test_flow_entropy_and_diagnostics_in_original_units(fitted_pair):
    scaled, standardized, features = fitted_pair
    mean, scale = scaled.target_scaler_.mean_, scaled.target_scaler_.scale_
    shift = np.log(scale).sum()
    kwargs = dict(num_mc_samples=3, num_flow_samples=30, return_all=True)
    _seed()
    expected = standardized.predict_with_combined_uncertainty(features, **kwargs)
    _seed()
    actual = scaled.predict_with_combined_uncertainty(features, **kwargs)
    for key in ("predictions", "mc_means"):
        if expected[key] is not None:
            factor, offset = (scale.item(), mean.item()) if key == "predictions" and mean.size == 1 else (scale, mean)
            np.testing.assert_allclose(actual[key], expected[key] * factor + offset, rtol=2e-5, atol=2e-4)
    for key in ("data_uncertainty", "total_uncertainty", "mc_uncertainties"):
        if key in expected and expected[key] is not None:
            np.testing.assert_allclose(actual[key], expected[key] + shift, rtol=2e-5, atol=2e-5)
    if expected["knowledge_uncertainty"] is not None:
        np.testing.assert_allclose(actual["knowledge_uncertainty"], expected["knowledge_uncertainty"], atol=2e-5)
        np.testing.assert_allclose(actual["mc_stds"], expected["mc_stds"] * scale, rtol=2e-5, atol=2e-4)
    _seed()
    prediction, knowledge, data = scaled.predict_with_combined_uncertainty(
        features, num_mc_samples=3, num_flow_samples=30
    )
    np.testing.assert_allclose(prediction, actual["predictions"])
    np.testing.assert_allclose(data, actual["data_uncertainty"])
    if knowledge is not None:
        np.testing.assert_allclose(knowledge, actual["knowledge_uncertainty"])


def test_flow_uncertainty_dataframe_quantiles_and_acquisition_units(fitted_pair, monkeypatch):
    import mother.ml.models.m_node as node_module

    scaled, standardized, features = fitted_pair
    mean, scale = scaled.target_scaler_.mean_, scaled.target_scaler_.scale_
    shift = np.log(scale).sum()
    captured = []
    original_emd = node_module.balsa_emd_from_mc_samples

    def capture(samples, **kwargs):
        captured.append(samples)
        return original_emd(samples, **kwargs)

    monkeypatch.setattr(node_module, "balsa_emd_from_mc_samples", capture)
    for method in ("bald", "balsa_emd"):
        kwargs = dict(num_samples=30, num_mc_samples=3, num_flow_samples=20, knowledge_method=method)
        _seed()
        expected, expected_q = standardized.predict_uncertainty(features, return_quantiles=True, **kwargs)
        _seed()
        actual, actual_q = scaled.predict_uncertainty(features, return_quantiles=True, **kwargs)
        factor, offset = (scale.item(), mean.item()) if mean.size == 1 else (scale, mean)
        np.testing.assert_allclose(actual_q, expected_q * factor + offset, rtol=2e-5, atol=2e-4)
        for key in ("pred", "mean_predictions"):
            np.testing.assert_allclose(
                np.stack(actual[key]), np.stack(expected[key]) * factor + offset, rtol=2e-5, atol=2e-4
            )
        np.testing.assert_allclose(actual["data_uncertainty"], expected["data_uncertainty"] + shift, atol=2e-5)
        if method == "bald" or not scaled._has_active_dropout(include_mlp=False):
            np.testing.assert_allclose(actual["total_uncertainty"], expected["total_uncertainty"] + shift, atol=2e-5)
        else:
            assert actual["total_uncertainty"].isna().all()
            raw_samples, target_samples = captured[-2:]
            transformed = [
                [samples * samples.new_tensor(scale) + samples.new_tensor(mean) for samples in batch]
                for batch in raw_samples
            ]
            for expected_batch, actual_batch in zip(transformed, target_samples):
                for expected_samples, actual_samples in zip(expected_batch, actual_batch):
                    torch.testing.assert_close(actual_samples, expected_samples)
            if mean.size == 1:
                np.testing.assert_allclose(
                    actual["knowledge_uncertainty"],
                    expected["knowledge_uncertainty"] * scale.item(),
                    rtol=2e-5,
                    atol=2e-4,
                )
        _seed()
        acquisition = scaled.predict_uncertainty(features, uncertainty_for_opt=True, **kwargs)
        _seed()
        frame = scaled.predict_uncertainty(features, **{**kwargs, "knowledge_method": "balsa_emd"})
        column = "knowledge_uncertainty" if frame["knowledge_uncertainty"].notna().any() else "total_uncertainty"
        np.testing.assert_allclose(acquisition, frame[column])


def test_flow_scaler_uses_only_internal_training_partition_and_refits():
    features, targets = _data(2)

    def split(dataset, y):
        return Subset(dataset, list(range(24))), Subset(dataset, list(range(24, 32)))

    targets[24:] += 100000
    model = _model(train_split=split).fit(features, targets)
    np.testing.assert_allclose(model.target_scaler_.mean_, targets[:24].mean(axis=0), rtol=1e-6)
    previous = model.target_scaler_.mean_.copy()
    model.partial_fit(features, targets + 100)
    np.testing.assert_array_equal(model.target_scaler_.mean_, previous)
    model.set_params(warm_start=True).fit(features, targets + 100)
    np.testing.assert_array_equal(model.target_scaler_.mean_, previous)
    model.set_params(warm_start=False).fit(features, targets + 100)
    np.testing.assert_allclose(model.target_scaler_.mean_, previous + 100)


def test_flow_scaling_constant_targets_clone_pickle_and_opt_out():
    features, _ = _data(1)
    targets = np.full(len(features), 5000, dtype=np.float32)
    model = _model(train_split=ValidSplit(cv=0.25, random_state=42)).fit(features, targets)
    np.testing.assert_array_equal(model.target_scaler_.scale_, [1.0])
    assert np.isfinite(model.history[-1, "train_loss"])
    assert np.isfinite(model.history[-1, "valid_loss"])
    restored = pickle.loads(pickle.dumps(model))
    _seed()
    expected = model.predict(features[:2], num_samples=20)
    _seed()
    np.testing.assert_array_equal(restored.predict(features[:2], num_samples=20), expected)
    cloned = clone(model)
    assert cloned.scale_targets
    assert not hasattr(cloned, "target_scaler_")
    unscaled = _model(scale_targets=False).fit(features, np.zeros(len(features), dtype=np.float32))
    assert not hasattr(unscaled, "target_scaler_")
    assert not clone(unscaled).scale_targets
    nonflow = _model(head_type="linear").fit(features, targets)
    assert not hasattr(nonflow, "target_scaler_")


def test_flow_loss_uses_standardized_training_and_validation_targets():
    features, targets = _data(2)
    model = _model().fit(features, targets)
    truth = torch.tensor(targets[:3])
    distribution = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros_like(truth), torch.ones_like(truth)), 1
    )
    expected = -distribution.log_prob(torch.tensor(model.target_scaler_.transform(targets[:3]))).mean()
    torch.testing.assert_close(model.get_loss(distribution, truth, torch.tensor(features[:3])), expected)
    raw = model.module_(torch.tensor(features[:3]))
    transformed = model._flow_in_target_units(raw)
    standardized_truth = torch.tensor(model.target_scaler_.transform(targets[:3]))
    torch.testing.assert_close(
        transformed.log_prob(truth),
        raw.log_prob(standardized_truth) - np.log(model.target_scaler_.scale_).sum(),
    )


@pytest.mark.parametrize("flow_type", FlowHead.SUPPORTED_FLOW_TYPES)
@pytest.mark.parametrize("n_targets", [1, 2])
def test_all_flow_families_support_scaled_training_and_inference(flow_type, n_targets):
    features, targets = _data(n_targets)
    if n_targets == 1:
        targets = targets.reshape(-1, 1)
    model = _model(flow_type=flow_type, flow_transforms=1).fit(features, targets)
    assert np.isfinite(model.history[-1, "train_loss"])
    samples = model.predict(features[:2], num_samples=8, return_sample_distribution=True)
    assert samples.shape == (2, 8, n_targets)
    assert np.isfinite(samples).all()
    distribution = next(model.forward_iter(features[:2]))
    assert distribution.event_shape == torch.Size([n_targets])
    assert torch.isfinite(distribution.log_prob(torch.tensor(targets[:2]))).all()


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_flow_scaling_rejects_nonfinite_targets(invalid):
    features, targets = _data(1)
    targets[0] = invalid
    with pytest.raises(ValueError, match="complete, finite targets"):
        _model().fit(features, targets)
