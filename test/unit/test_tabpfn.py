import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split

from mother.ml.models.m_tabpfn import (
    TabPFNClassifierMother,
    TabPFNEmbeddingTransformer,
    TabPFNRegressorMother,
)


def get_data_containers(X, y):
    """
    Convert X and y into different container formats for testing.

    Returns a list of tuples (X_converted, y_converted, description) where each tuple
    contains the data in a different container format.
    """
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
X = rng.random((10, 3))
y = rng.integers(1, 3, 10)


@pytest.fixture(params=[container for container in get_data_containers(X, y)], ids=lambda x: x[2])
def data_containers(request):
    """
    Pytest fixture that returns the get_data_containers function.
    """
    return request.param


@pytest.mark.slow
class TestRegression:
    # set up the test
    model = TabPFNRegressorMother()
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    def test_set_params(self):
        # Test if set_params works correctly
        # Replace with actual parameters
        params = {"n_estimators": 10, "softmax_temperature": 0.5}
        self.model.set_params(**params)

        for key, value in params.items():
            assert getattr(self.model, key) == value, f"Parameter {key} was not set correctly"

    def test_predict_shape(self):
        # Test if predict returns the correct shape

        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)

        assert predictions.shape == self.y_test.shape, "Predictions shape should match target shape"

    def test_predict(self, data_containers: tuple):
        # Test if predict returns the correct shape
        self.model.fit(data_containers[0], data_containers[1])

    @pytest.mark.parametrize(
        "invalid_input", [(None, None), (1, 2), (0.1, 0.2)], ids=["None", "int not allowed", "float not allowed"]
    )
    def test_invalid_input_raises(self, invalid_input):
        # Test if the model raises an error for invalid input
        with pytest.raises(TypeError):
            self.model.fit(*invalid_input)

    def test_regression(self):
        # TabPFNRegressorMother test
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        assert len(predictions) == len(self.y_test), "Predictions length should match target length"
        assert np.all(np.isfinite(predictions)), "Predictions should not contain NaN or infinite values"

    def test_predict_with_uncertainty_consistency(self):
        # Test if predict_with_uncertainty is consistent with predict
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        predictions_with_uncertainty = self.model.predict_uncertainty(self.X_test)

        assert np.allclose(predictions, predictions_with_uncertainty["pred"]), (
            "Predictions should match between predict and predict_with_uncertainty"
        )

        assert (predictions_with_uncertainty["total_uncertainty"] > 0).all()

    def test_predict_with_uncertainty_opt(self):
        # Test if predict_with_uncertainty is consistent with predict
        self.model.fit(self.X_train, self.y_train)
        pred_with_uncertainty = self.model.predict_uncertainty(self.X_test)
        pred_with_uncertainty_opt = self.model.predict_uncertainty(self.X_test, uncertainty_for_opt=True)

        assert pred_with_uncertainty["total_uncertainty"].equals(pred_with_uncertainty_opt)


@pytest.mark.slow
class TestClassification:
    # set up the test
    model = TabPFNClassifierMother()
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    def test_set_params(self):
        # Test if set_params works correctly
        # Replace with actual parameters
        params = {"n_estimators": 10, "softmax_temperature": 0.5}
        self.model.set_params(**params)

        for key, value in params.items():
            assert getattr(self.model, key) == value, f"Parameter {key} was not set correctly"

    def test_predict_shape(self):
        # Test if predict returns the correct shape

        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)

        assert predictions.shape == self.y_test.shape, "Predictions shape should match target shape"

    @pytest.mark.parametrize(
        "invalid_input", [(None, None), (1, 2), (0.1, 0.2)], ids=["None", "int not allowed", "float not allowed"]
    )
    def test_invalid_input_raises(self, invalid_input):
        # Test if the model raises an error for invalid input
        with pytest.raises(TypeError):
            self.model.fit(*invalid_input)

    def test_predict(self, data_containers: tuple):
        # accepted types: np.array, pd.DataFrame, list
        self.model.fit(data_containers[0], data_containers[1])

    def test_classification(self):
        # TabPFNRegressorMother test
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        assert len(predictions) == len(self.y_test), "Predictions length should match target length"
        assert np.all(np.isfinite(predictions)), "Predictions should not contain NaN or infinite values"


@pytest.mark.slow
class TestTabPFNEmbeddingTransformer:
    X = pd.DataFrame(np.random.rand(20, 10), columns=[f"f{i}" for i in range(10)])
    regression_y = pd.Series(np.random.rand(20))
    classification_y = pd.Series(np.random.randint(0, 2, size=20))

    def test_init_and_basic_fit_transform(self):
        # Create dummy data
        transformer = TabPFNEmbeddingTransformer(
            model_type="regression",
            use_kfold=False,
            n_estimators=2,
            random_state=42,
            return_separate_columns=True,
        )
        # Should fit and transform without error
        result = transformer.fit_transform(self.X, self.regression_y)
        assert isinstance(result, (np.ndarray, pd.DataFrame))
        assert result.shape[0] == self.X.shape[0]

    def test_invalid_kfold_and_model(self):
        # Should raise ValueError if both pre_fitted model and use_kfold=True
        class DummyModel:
            pass

        with pytest.raises(ValueError):
            TabPFNEmbeddingTransformer(
                model_type="regression",
                use_kfold=True,
                model=DummyModel(),
            )

    def test_fit_transform_with_groups(self):
        groups = np.random.randint(0, 3, size=20)
        transformer = TabPFNEmbeddingTransformer(
            model_type="classification",
            use_kfold=True,
            n_folds=3,
            n_estimators=4,
            random_state=0,
        )
        result = transformer.fit_transform(self.X, self.classification_y, groups=groups)
        assert result.shape[0] == self.X.shape[0]

    def test_invalid_input_raises(self):
        transformer = TabPFNEmbeddingTransformer()
        # Should raise error if X or y is None
        with pytest.raises(ValueError):
            transformer.fit_transform(None, None)

    def test_prefitted_model(self):
        model = TabPFNRegressorMother().fit(pd.DataFrame(np.random.random(self.X.shape)), self.regression_y)

        transformer_prefitted = TabPFNEmbeddingTransformer(
            model_type="regression", use_kfold=False, n_estimators=1, model=model, random_state=0
        )

        result_prefitted = transformer_prefitted.transform(self.X)

        transformer_newfit = TabPFNEmbeddingTransformer(
            model_type="regression", use_kfold=False, n_estimators=1, random_state=0
        )

        result_newfit = transformer_newfit.fit_transform(self.X, self.regression_y)
        assert not result_prefitted.equals(result_newfit)

    def test_array_input(self):
        transformer = TabPFNEmbeddingTransformer(
            model_type="regression", use_kfold=False, n_estimators=1, random_state=0
        )

        result = transformer.fit_transform(self.X.to_numpy(), self.regression_y)
        assert result.shape[0] == self.X.shape[0]

    def test_only_best_embeddings(self):
        transformer = TabPFNEmbeddingTransformer(
            model_type="regression", use_kfold=False, n_estimators=4, random_state=0
        )

        assert (
            transformer.fit_transform(self.X, self.regression_y).shape[1]
            == transformer.fit_transform(self.X, self.regression_y, only_best_embeddings=True).shape[1] * 4
        )

        transformer = TabPFNEmbeddingTransformer(
            model_type="classification", use_kfold=False, n_estimators=4, random_state=0
        )

        assert (
            transformer.fit_transform(self.X, self.classification_y).shape[1]
            == transformer.fit_transform(self.X, self.classification_y, only_best_embeddings=True).shape[1] * 4
        )

    def test_get_feature_names_out(self):
        transformer = TabPFNEmbeddingTransformer(
            model_type="regression", use_kfold=False, n_estimators=1, random_state=0
        )

        result = transformer.fit_transform(self.X, self.regression_y)
        feature_names = transformer.get_feature_names_out()
        assert result.shape[1] == feature_names.shape[0]

    def test_n_estimators(self):
        transformer1 = TabPFNEmbeddingTransformer(
            model_type="regression", use_kfold=False, n_estimators=1, random_state=0
        )

        transformer2 = TabPFNEmbeddingTransformer(
            model_type="regression", use_kfold=False, n_estimators=4, random_state=0
        )

        assert (
            transformer1.fit_transform(self.X, self.regression_y).shape[1] * 4
            == transformer2.fit_transform(self.X, self.regression_y).shape[1]
        )

    def test_kfold_transform_on_unseen_data(self):
        """Regression test for the bug where transform() raised ValueError after
        use_kfold=True fit because self.model was never assigned in the k-fold path."""
        X_train, X_test = self.X.iloc[:15], self.X.iloc[15:]
        y_train = self.regression_y.iloc[:15]

        transformer = TabPFNEmbeddingTransformer(
            model_type="regression",
            use_kfold=True,
            n_folds=3,
            n_estimators=2,
            random_state=42,
        )
        transformer.fit(X_train, y_train)

        # Must not raise ValueError("Transformer hasn't been fitted...")
        result = transformer.transform(X_test)
        assert result.shape[0] == X_test.shape[0]
        assert result.shape[1] == transformer._embedding_dim

    def test_kfold_fit_transform_then_transform_unseen(self):
        """fit_transform on train data followed by transform on held-out data must work
        and produce embeddings with the same feature dimension."""
        X_train, X_test = self.X.iloc[:15], self.X.iloc[15:]
        y_train = self.regression_y.iloc[:15]

        transformer = TabPFNEmbeddingTransformer(
            model_type="regression",
            use_kfold=True,
            n_folds=3,
            n_estimators=2,
            random_state=42,
        )
        train_embeddings = transformer.fit_transform(X_train, y_train)
        test_embeddings = transformer.transform(X_test)

        assert train_embeddings.shape[0] == X_train.shape[0]
        assert test_embeddings.shape[0] == X_test.shape[0]
        assert train_embeddings.shape[1] == test_embeddings.shape[1]

    def test_kfold_only_best_embeddings_transform_unseen(self):
        """only_best_embeddings=True must work end-to-end when transforming unseen data
        after a k-fold fit (best_estimator_idx reused on the full-data model)."""
        X_train, X_test = self.X.iloc[:15], self.X.iloc[15:]
        y_train = self.regression_y.iloc[:15]

        transformer = TabPFNEmbeddingTransformer(
            model_type="regression",
            use_kfold=True,
            n_folds=3,
            n_estimators=4,
            random_state=42,
        )
        train_embeddings = transformer.fit_transform(X_train, y_train, only_best_embeddings=True)
        test_embeddings = transformer.transform(X_test, only_best_embeddings=True)

        # With only_best_embeddings the dim should be 1/4 of the full embedding
        assert train_embeddings.shape[1] == test_embeddings.shape[1]
        assert transformer.best_estimator_idx is not None
