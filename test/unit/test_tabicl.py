import numpy as np
import pandas as pd
import pytest
from optuna.trial import FixedTrial
from sklearn.datasets import load_diabetes, load_wine
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from mother.ml.models.m_tabicl import (
    TabICLClassifierMother,
    TabICLEmbeddingTransformer,
    TabICLRegressorMother,
)

# Row-wise transformers embedding size (last dimension of the output before prediction head)
TABICL_EMBEDDING_SIZE = 512


def get_data_containers(X, y):
    """Convert X and y into different container formats for testing."""
    return [
        (X, y, "original"),
        (np.array(X), np.array(y), "numpy arrays"),
        (
            pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X,
            pd.Series(y) if not isinstance(y, pd.Series) else y,
            "pandas DataFrame/Series",
        ),
    ]


rng = np.random.default_rng(42)
X_small = rng.random((12, 4))
y_small = rng.integers(0, 2, 12)


# This fixture will run tests that use it with each of the different data container formats defined in
# get_data_containers.
@pytest.fixture(params=[container for container in get_data_containers(X_small, y_small)], ids=lambda x: x[2])
def data_containers(request):
    return request.param


@pytest.mark.slow
class TestTabICLRegression:
    model = TabICLRegressorMother(n_estimators=1)

    @pytest.fixture(autouse=True)
    def _fresh_model(self) -> None:
        self.model = TabICLRegressorMother(n_estimators=1)

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    def test_default_parameters(self):
        defaults = self.model.default_parameters()
        assert defaults["n_estimators"] == 8
        assert defaults["outlier_threshold"] == 4.0
        assert defaults["allow_auto_download"] is True
        assert defaults["kv_cache"] is False

    def test_set_and_get_params(self):
        params = {"n_estimators": 3, "outlier_threshold": 5.0, "kv_cache": False}
        self.model.set_params(**params)
        got = self.model.get_params()

        for key, value in params.items():
            assert getattr(self.model, key) == value
            assert got[key] == value

    def test_set_params_raises_error_on_unknown_keys(self):
        model = TabICLRegressorMother()
        with pytest.raises(ValueError):
            model.set_params(unknown_parameter=123)

    @pytest.mark.parametrize(
        "invalid_input", [(None, None), (1, 2), (0.1, 0.2)], ids=["None", "int not allowed", "float not allowed"]
    )
    def test_invalid_input_raises(self, invalid_input):
        with pytest.raises(TypeError):
            self.model.fit(*invalid_input)

    def test_empty_input_raises(self):
        X = np.empty((0, 4))
        y = np.empty((0,))
        with pytest.raises(ValueError):
            self.model.fit(X, y)

    def test_predict_shape_and_values(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == self.y_test.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_with_uncertainty_outputs(self):
        self.model.fit(self.X_train, self.y_train)
        output, quantiles = self.model.predict_uncertainty(self.X_test, return_quantiles=True)

        assert isinstance(output, pd.DataFrame)
        assert isinstance(quantiles, np.ndarray)
        assert list(output.columns) == [
            "mean_predictions",
            "knowledge_uncertainty",
            "data_uncertainty",
            "total_uncertainty",
        ]
        assert quantiles.shape[0] == len(self.X_test)
        assert quantiles.shape[1] >= 3
        assert (output["total_uncertainty"] >= 0).all()

    def test_predict_with_uncertainty_opt(self):
        self.model.fit(self.X_train, self.y_train)
        output = self.model.predict_uncertainty(self.X_test)
        output_opt = self.model.predict_uncertainty(self.X_test, uncertainty_for_opt=True)
        assert isinstance(output, pd.DataFrame)
        assert isinstance(output_opt, pd.Series)
        assert output["total_uncertainty"].equals(output_opt)

    def test_hyperparameter_space_regressor(self):
        trial = FixedTrial({"reg__n_estimators": 5, "reg__outlier_threshold": 3.2})
        space = self.model.get_hyperparameter_space(self.X_train, self.y_train, trial, prefix="reg__")

        assert space["reg__n_estimators"] == 5
        assert space["reg__outlier_threshold"] == 3.2
        assert "reg__softmax_temperature" not in space
        assert "reg__average_logits" not in space

    def test_set_params_after_fit_does_not_change(self):
        model = TabICLRegressorMother(n_estimators=1)
        model.fit(self.X_train, self.y_train)
        before = model.n_estimators
        model.set_params(n_estimators=6)
        assert model.n_estimators == before

    def test_predict_uncertainty_adds_default_quantiles(self):
        model = TabICLRegressorMother(n_estimators=1)
        model.fit(self.X_train, self.y_train)
        output, quantiles = model.predict_uncertainty(self.X_test, return_quantiles=True, quantiles=[0.5])
        assert isinstance(quantiles, np.ndarray)
        assert isinstance(output, pd.DataFrame)
        assert quantiles.shape[1] >= 3
        assert output.shape[0] == self.X_test.shape[0]

    def test_predict_uncertainty_with_numpy_input_index(self):
        model = TabICLRegressorMother(n_estimators=1)
        model.fit(self.X_train, self.y_train)
        output = model.predict_uncertainty(np.array(self.X_test))
        assert isinstance(output.index, pd.RangeIndex)

    def test_predict_uncertainty_raises_if_predict_returns_wrong_type(self, monkeypatch):
        model = TabICLRegressorMother(n_estimators=1)
        model.fit(self.X_train, self.y_train)

        def fake_predict(*args, **kwargs):  # noqa: ANN002, ANN003
            return {"not": "an array"}

        monkeypatch.setattr(model, "predict", fake_predict)
        with pytest.raises(TypeError):
            model.predict_uncertainty(self.X_test)


@pytest.mark.slow
class TestTabICLClassification:
    model = TabICLClassifierMother(n_estimators=1)

    @pytest.fixture(autouse=True)
    def _fresh_model(self) -> None:
        self.model = TabICLClassifierMother(n_estimators=1)

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    def test_default_parameters(self):
        defaults = self.model.default_parameters()
        assert defaults["n_estimators"] == 8
        assert defaults["softmax_temperature"] == 0.9
        assert defaults["average_logits"] is True
        assert defaults["allow_auto_download"] is True
        assert defaults["kv_cache"] is False

    def test_set_and_get_params(self):
        params = {"n_estimators": 3, "softmax_temperature": 1.2, "average_logits": False}
        self.model.set_params(**params)
        got = self.model.get_params()

        for key, value in params.items():
            assert getattr(self.model, key) == value
            assert got[key] == value

    def test_set_params_raises_error_on_unknown_keys(self):
        model = TabICLClassifierMother(n_estimators=1)
        with pytest.raises(ValueError):
            model.set_params(unknown_parameter=123)

    @pytest.mark.parametrize(
        "invalid_input", [(None, None), (1, 2), (0.1, 0.2)], ids=["None", "int not allowed", "float not allowed"]
    )
    def test_invalid_input_raises(self, invalid_input):
        with pytest.raises(TypeError):
            self.model.fit(*invalid_input)

    def test_predict_with_data_containers(self, data_containers: tuple):
        # Test if no error is raised when fitting with different data container formats
        self.model.fit(data_containers[0], data_containers[1])

    def test_predict_and_predict_proba_shapes(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        probabilities = self.model.predict_proba(self.X_test)

        assert predictions.shape == self.y_test.shape
        assert probabilities.shape[0] == len(self.X_test)
        assert probabilities.shape[1] == len(np.unique(self.y_train))
        assert np.all(np.isfinite(predictions))

    def test_hyperparameter_space_classifier(self):
        trial = FixedTrial(
            {
                "clf__n_estimators": 4,
                "clf__softmax_temperature": 1.1,
                "clf__average_logits": True,
                "clf__outlier_threshold": 3.0,
            }
        )
        space = self.model.get_hyperparameter_space(self.X_train, self.y_train, trial, prefix="clf__")

        assert space["clf__n_estimators"] == 4
        assert space["clf__softmax_temperature"] == 1.1
        assert space["clf__average_logits"] is True

    def test_hyperparameter_space_forces_average_logits_with_one_estimator(self):
        trial = FixedTrial({"clf__n_estimators": 1, "clf__softmax_temperature": 0.8, "clf__outlier_threshold": 3.0})
        space = self.model.get_hyperparameter_space(self.X_train, self.y_train, trial, prefix="clf__")

        assert space["clf__n_estimators"] == 1
        assert space["clf__average_logits"] is False

    def test_set_params_after_fit_does_not_change(self):
        model = TabICLClassifierMother(n_estimators=1)
        model.fit(self.X_train, self.y_train)
        before = model.n_estimators
        model.set_params(n_estimators=6)
        assert model.n_estimators == before


@pytest.mark.slow
class TestTabICLEmbeddingTransformer:
    X = pd.DataFrame(np.random.rand(24, 8), columns=[f"f{i}" for i in range(8)])
    regression_y = pd.Series(np.random.rand(24))
    classification_y = pd.Series(np.random.randint(0, 2, size=24))

    def test_init_invalid_kfold_with_prefitted_model(self):
        with pytest.raises(ValueError):
            TabICLEmbeddingTransformer(
                model_type="regression",
                use_kfold=True,
                model=TabICLRegressorMother(n_estimators=1),
            )

    def test_fit_transform_regression_basic(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            random_state=42,
            return_separate_columns=True,
        )
        result = transformer.fit_transform(self.X, self.regression_y)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == self.X.shape[0]
        assert result.shape[1] == TABICL_EMBEDDING_SIZE

    def test_fit_transform_with_groups_classification(self):
        groups = np.random.randint(0, 3, size=self.X.shape[0])
        transformer = TabICLEmbeddingTransformer(
            model_type="classification",
            use_kfold=True,
            n_folds=3,
            n_estimators=1,
            random_state=0,
        )
        result = transformer.fit_transform(self.X, self.classification_y, groups=groups)
        assert result.shape[0] == self.X.shape[0]
        assert result.shape[1] == TABICL_EMBEDDING_SIZE

    def test_fit_transform_without_y_raises(self):
        transformer = TabICLEmbeddingTransformer(model_type="regression", use_kfold=False)
        with pytest.raises(ValueError):
            transformer.fit_transform(self.X, None)

    def test_transform_before_fit_raises(self):
        transformer = TabICLEmbeddingTransformer(model_type="regression", use_kfold=False)
        with pytest.raises(NotFittedError):
            transformer.transform(self.X)

    def test_missing_feature_raises(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            random_state=0,
        )
        transformer.fit(self.X, self.regression_y)
        bad_X = self.X.drop(columns=["f0"])

        with pytest.raises(ValueError):
            transformer.transform(bad_X)

    def test_return_single_vector_column(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            return_separate_columns=False,
            random_state=0,
        )
        result = transformer.fit_transform(self.X, self.regression_y)

        assert list(result.columns) == ["tabiclembedding"]
        assert result.shape[0] == self.X.shape[0]
        assert isinstance(result.iloc[0, 0], np.ndarray)

    def test_prefitted_model_transform(self):
        model = TabICLRegressorMother(n_estimators=1).fit(self.X, self.regression_y)
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            model=model,
            n_estimators=1,
        )
        result = transformer.transform(self.X)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == self.X.shape[0]
        assert result.shape[1] == TABICL_EMBEDDING_SIZE

    def test_array_input(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            random_state=0,
        )
        result = transformer.fit_transform(self.X.to_numpy(), self.regression_y)
        assert result.shape[0] == self.X.shape[0]
        assert result.shape[1] == TABICL_EMBEDDING_SIZE

    def test_get_feature_names_out(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            random_state=0,
        )
        result = transformer.fit_transform(self.X, self.regression_y)
        names = transformer.get_feature_names_out()

        assert result.shape[1] == names.shape[0]

    def test_get_feature_names_out_before_fit_raises(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            random_state=0,
        )
        with pytest.raises(NotFittedError):
            transformer.get_feature_names_out()

    def test_fit_with_prefitted_but_missing_model_raises_runtime_error(self):
        transformer = TabICLEmbeddingTransformer(model_type="regression", use_kfold=False)
        transformer.pre_fitted = True
        transformer.model = None
        with pytest.raises(RuntimeError):
            transformer.fit(self.X, self.regression_y)

    def test_fit_with_prefitted_model_path(self):
        model = TabICLRegressorMother(n_estimators=1).fit(self.X, self.regression_y)
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            model=model,
            n_estimators=1,
        )
        transformer.fit(self.X, self.regression_y)
        assert transformer.train_embeddings_ is not None

    def test_fit_with_unfitted_prefitted_model_raises_not_fitted_error(self):
        model = TabICLRegressorMother(n_estimators=1)
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            model=model,
            n_estimators=1,
        )
        with pytest.raises(NotFittedError):
            transformer.fit(self.X, self.regression_y)

    def test_fit_falls_back_when_samples_less_than_folds(self):
        X_tiny = self.X.iloc[:3].copy()
        y_tiny = self.regression_y.iloc[:3].copy()
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=True,
            n_folds=5,
            n_estimators=1,
            random_state=0,
        )
        result = transformer.fit_transform(X_tiny, y_tiny)
        assert result.shape[0] == 3

    def test_fit_transform_regression_with_groups_uses_group_kfold(self):
        groups = np.random.randint(0, 3, size=self.X.shape[0])
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=True,
            n_folds=3,
            n_estimators=1,
            random_state=0,
        )
        result = transformer.fit_transform(self.X, self.regression_y, groups=groups)
        assert result.shape[0] == self.X.shape[0]

    def test_fit_transform_regression_without_groups_uses_kfold(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=True,
            n_folds=3,
            n_estimators=1,
            random_state=0,
        )
        result = transformer.fit_transform(self.X, self.regression_y)
        assert result.shape[0] == self.X.shape[0]

    def test_fit_transform_classification_without_groups(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="classification",
            use_kfold=True,
            n_folds=3,
            n_estimators=1,
            random_state=0,
        )
        result = transformer.fit_transform(self.X, self.classification_y)
        assert result.shape[0] == self.X.shape[0]

    def test_invalid_model_type_raises(self):
        transformer = TabICLEmbeddingTransformer(model_type="unknown", use_kfold=False)  # type: ignore
        with pytest.raises(ValueError):
            transformer.fit_transform(self.X, self.regression_y)

    def test_transform_reorders_dataframe_columns(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            random_state=0,
        )
        transformer.fit(self.X, self.regression_y)

        X_reordered = self.X[["f3", "f1", "f2", "f0", "f4", "f5", "f6", "f7"]].copy()
        X_reordered["extra"] = 1.0
        out = transformer.transform(X_reordered)
        assert out.shape[0] == self.X.shape[0]

    def test_get_feature_names_out_single_column_mode(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            return_separate_columns=False,
            random_state=0,
        )
        transformer.fit_transform(self.X, self.regression_y)
        names = transformer.get_feature_names_out()
        assert names.tolist() == ["tabiclembedding"]

    def test_single_column_mode_with_numpy_input_uses_default_index(self):
        transformer = TabICLEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=1,
            return_separate_columns=False,
            random_state=0,
        )
        out = transformer.fit_transform(self.X.to_numpy(), self.regression_y)
        assert isinstance(out.index, pd.RangeIndex)

    def test_extract_representations_wrong_estimator_type_raises(self):
        reg = TabICLRegressorMother(n_estimators=1).fit(self.X, self.regression_y)
        clf = TabICLClassifierMother(n_estimators=1).fit(self.X, self.classification_y)

        transformer_cls = TabICLEmbeddingTransformer(model_type="classification", use_kfold=False, n_estimators=1)
        with pytest.raises(TypeError):
            transformer_cls._extract_representations(reg, self.X)

        transformer_reg = TabICLEmbeddingTransformer(model_type="regression", use_kfold=False, n_estimators=1)
        with pytest.raises(TypeError):
            transformer_reg._extract_representations(clf, self.X)
