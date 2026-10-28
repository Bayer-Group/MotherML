import logging
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from optuna.trial import Trial
from scipy.special import rel_entr
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from mother.ml.core import AbstractMotherPipeline

module_logger: logging.Logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    __order__ = "KNOWLEDGE DATA TOTAL"
    KNOWLEDGE = "knowledge"
    DATA = "data"
    TOTAL = "total"


class EnsembleDerivedUncertaintyMethod(Enum):
    ENTROPY = "entropy"
    VARIANCE = "variance"
    DISAGREEMENT = "disagreement"


class CVEnsembleClassifierMother(AbstractMotherPipeline, MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """
    Homogeneous ensemble classifier that combines predictions from multiple fixed pre-fitted estimators. As it inherits
    from `BaseEstimator`, it follows the scikit-learn estimator interface. Ensembles are meta-estimators and can
    be recognized as such by scikit-learn utilities.

    Note:
        - Only single-target predictions are supported.
        - Assumes that member estimators are pre-fitted, so usage with MotherTuning or similar is not intended.
        - Can provide predicted probabilities and uncertainties with the `predict_uncertainty` method.

    Attributes:
        estimators (list): List of pre-fitted estimators to be used in the ensemble. Each estimator must have a
        `classes_` attribute and a `predict_uncertainty` method. All estimator instances have to be of the same type.

    Methods:
        get_hyperparameter_space: Returns the hyperparameter search space for the ensemble (no tunable hyperparameters
        so far, so an empty dictionary is returned).
        set_params: Sets the parameters of the ensemble (not allowed so far, so self is returned unchanged).
        get_params: Gets the parameters of the ensemble.
        fit: Fits the ensemble (no-op as estimators are pre-fitted).
        predict: Predicts labels using the aggregated outputs of the ensemble members.
        predict_proba: Predicts class probabilities using the ensemble.
        predict_uncertainty: Predicts uncertainties associated with the ensemble's predictions.
    """

    def __init__(self, estimators: list):
        self.estimators = estimators
        self._validate_estimators(estimators)

        # Set fitted attribute "classes_" since member estimators are already fitted and fit() is a no-op
        self._set_fitted_attributes()

    @staticmethod
    def _validate_estimators(estimators: list) -> None:
        """
        Validate the provided estimators for the ensemble and their class label consistency.

        Args:
            estimators (list): List of pre-fitted estimators to be used in the ensemble.

        Raises:
            ValueError: If the estimators list has fewer than two members, if estimator instances are not of
            the same type, if any estimator is not fitted, if any estimator lacks the required attributes,
            or if class labels are inconsistent across estimators.
        """
        if len(estimators) < 2:
            raise ValueError("At least two estimators must be provided as ensemble members.")

        classes = None
        expected_type = type(estimators[0])
        for i, estimator in enumerate(estimators):
            # Validate that each estimator has same model type
            if not isinstance(estimator, expected_type):
                raise ValueError(f"Estimator {i} must be of type {expected_type.__name__}.")

            # Validate that each member estimator is fitted
            try:
                check_is_fitted(estimator)
            except NotFittedError as e:
                raise ValueError(f"Estimator {i} is not fitted.") from e

            # Validate that each member estimator has required attributes
            if not hasattr(estimator, "classes_"):
                raise ValueError(f"Estimator {i} must have a 'classes_' attribute.")
            if not callable(getattr(estimator, "predict_uncertainty", None)):
                raise ValueError(f"Estimator {i} must have a 'predict_uncertainty' method.")
            if not hasattr(estimator, "n_features_in_"):
                raise ValueError(f"Estimator {i} must expose 'n_features_in_'.")
            if not hasattr(estimator, "feature_names_in_"):
                raise ValueError(f"Estimator {i} must expose 'feature_names_in_'.")

            # Validate that member classes are one-dimensional, non-empty, and unique
            member_classes = np.asarray(estimator.classes_)
            if (
                member_classes.ndim != 1
                or len(member_classes) == 0
                or len(np.unique(member_classes)) != len(member_classes)
            ):
                raise ValueError(f"Estimator {i} must have distinct one-dimensional class labels (classes_).")
            if i == 0:
                classes = member_classes
            if not (member_classes == classes).all():
                raise ValueError("All estimators must have the same class labels.")

    def _set_fitted_attributes(self) -> None:
        """
        Set the fitted attributes for the ensemble based on its member estimators.

        Raises:
            ValueError: If any estimator has invalid class labels or if class labels are inconsistent across estimators.
        """
        if hasattr(self, "classes_") and self.classes_ is not None:
            return

        # After validating member estimators and their class labels, set the ensemble's classes_ and feature attributes
        self._validate_estimators(self.estimators)
        self.classes_ = np.asarray(self.estimators[0].classes_)
        self.feature_names_in_ = getattr(self.estimators[0], "feature_names_in_")
        self.n_features_in_ = getattr(self.estimators[0], "n_features_in_")

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        """
        Returns the hyperparameter search space for the ensemble. As no tunable hyperparameters are available for the
        ensemble itself, returns an empty dictionary.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            trial (Trial): Optuna trial object for hyperparameter optimization.
            prefix (str): Prefix to add to hyperparameter names.

        Returns:
            dict: Dictionary representing the hyperparameter search space.
        """
        # no tunable hyperparameters for the ensemble itself
        return {}

    def __sklearn_clone__(self):
        """
        Return an independent frozen ensemble, preserving fitted ensemble estimators.
        Overrides the default sklearn clone behavior.
        """
        return deepcopy(self)

    def __sklearn_tags__(self):
        """
        Returns the scikit-learn estimator tags for the ensemble.
        Advertised only real ensemble capabilities.
        """
        tags = super().__sklearn_tags__()

        # Does not support multi-output targets
        tags.target_tags.multi_output = False

        return tags

    def set_params(self, **params):
        """
        Sets the parameters for the ensemble. Since the ensemble does not allow parameter updates, this method only
        logs a warning if any parameters are provided.

        Args:
            **params: Parameters to set.

        Returns:
            self: Returns the ensemble instance.
        """
        if params:
            module_logger.warning(f"Setting parameters is not allowed for {self.__class__.__name__}")
        return self

    def get_params(self, deep: bool = True) -> dict:
        """
        Retrieves the parameters of the ensemble.

        Args:
            deep (bool): If True, will return the parameters of the estimators as well.

        Returns:
            dict: Dictionary containing the parameters of the ensemble and its estimators.
        """
        params = {"estimators": self.estimators}

        if deep:
            params.update(
                {
                    **{
                        f"estimator_{i}__{key}": value
                        for i, est in enumerate(self.estimators)
                        for key, value in est.get_params(deep=deep).items()
                    }
                }
            )

        return params

    def fit(self, X, y, **fit_params):
        """
        No operation is performed. Since the ensemble members are pre-fitted, this method does not perform any
        refitting.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            **fit_params: Additional fit parameters.

        Returns:
            self: Returns the ensemble instance.
        """
        module_logger.warning(
            f"Ensemble members of {self.__class__.__name__} are already fitted. No refit will be performed."
        )
        return self

    def _aggregate_probas(self, estimator_probas: np.ndarray, strategy: str = "average") -> np.ndarray:
        """
        Aggregates predicted probabilities from multiple estimators using the specified strategy.

        Args:
            estimator_probas (np.ndarray): Predicted probabilities from multiple estimators,
            shape (n_estimators, n_samples, n_classes).
            strategy (str): Strategy for aggregating predictions. Current options are:
                - "average" (default): Aggregates predicted probabilities across estimators by computing the mean for
                    each sample.

        Returns:
            np.ndarray: Aggregated predicted probabilities for each sample, shape (n_samples, n_classes).
        """

        # input validation
        if estimator_probas.ndim != 3:  # shape (n_estimators, n_samples, n_classes)
            raise ValueError(
                f"Unexpected dimension for probas: {estimator_probas.ndim}."
                + "Should be 3 (n_estimators, n_samples, n_classes)."
            )

        if strategy == "average":
            mean_probas = np.mean(
                estimator_probas, axis=0
            )  # Mean across estimators for each sample -> shape (n_samples, n_classes)
            return mean_probas
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _collect_member_probas(self, X: pd.DataFrame) -> np.ndarray:
        """
        Coordinates probability prediction of all ensemble members.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Stacked probabilities from all members, shape (n_estimators, n_samples, n_classes)
        """
        probas = []
        for estimator in self.estimators:
            prediction = estimator.predict_proba(X)
            probas.append(self._extract_member_probas(prediction))
        return np.stack(probas)

    def _extract_member_probas(self, prediction: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Extracts and validates member probabilities for one estimator.

        Args:
            prediction (pd.DataFrame | np.ndarray): Raw output from the estimator.
                Can contain proba and uncertainty columns.

        Returns:
            np.ndarray: Validated class probabilities, shape (n_samples, n_classes)
        """
        proba_columns = [f"proba_{i}" for i in range(len(self.classes_))]

        member_probas = (
            prediction.loc[:, proba_columns].to_numpy(dtype=float)
            if isinstance(prediction, pd.DataFrame)
            else np.asarray(prediction, dtype=float)
        )

        if member_probas.shape[1] != len(self.classes_):
            raise ValueError(
                "Unexpected shape for member probas. Got "
                + str(member_probas.shape)
                + "probability columns; expected probabilities for"
                + str(len(self.classes_))
                + "classes."
            )

        if (
            not np.isfinite(member_probas).all()
            or np.any((member_probas < 0) | (member_probas > 1))
            or not np.allclose(np.sum(member_probas, axis=1), 1)
        ):
            raise ValueError("Estimator returned invalid class probabilities.")

        return member_probas

    @staticmethod
    def _build_result(
        X: pd.DataFrame,
        ensemble_pred: np.ndarray,
        ensemble_probas: np.ndarray,
        knowledge_uncertainty: np.ndarray | None = None,
        data_uncertainty: np.ndarray | None = None,
        total_uncertainty: np.ndarray | None = None,
    ) -> pd.DataFrame:
        pred_res = pd.DataFrame(
            {
                "pred": ensemble_pred,
                **{f"proba_{i}": ensemble_probas[:, i] for i in range(ensemble_probas.shape[1])},
                "mean_predictions": None,  # intentionally unset for classifier
                "knowledge_uncertainty": knowledge_uncertainty,
                "data_uncertainty": data_uncertainty,
                "total_uncertainty": total_uncertainty,
            },
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )

        return pred_res

    def _probas_to_preds(self, probas: np.ndarray) -> np.ndarray:
        """
        Convert class probabilities to predicted class labels.

        Args:
            probas (np.ndarray): Array of class probabilities.
        Returns:
            np.ndarray: Predicted class labels.
        """
        class_indices = np.argmax(probas, axis=1)
        return self.classes_[class_indices]

    def predict(self, X: pd.DataFrame, aggregation_strategy: str = "average") -> np.ndarray:
        """
        Predict class labels for the input data X using the ensemble.

        Args:
            X (pd.DataFrame): DataFrame containing features.
            aggregation_strategy (str): Strategy to aggregate predictions from ensemble members. Default is "average".

        Returns:
            np.ndarray: Array of predicted class labels, shape (n_samples,).
        """
        aggregated_probas = self.predict_proba(X, aggregation_strategy=aggregation_strategy)
        predictions = self._probas_to_preds(aggregated_probas)
        return predictions

    def predict_proba(self, X: pd.DataFrame, aggregation_strategy: str = "average") -> np.ndarray:
        """
        Predict class probabilities for the input data X using the ensemble.

        Args:
            X (pd.DataFrame): DataFrame containing features.
            aggregation_strategy (str): Strategy to aggregate predictions from ensemble members. Default is "average".

        Returns:
            np.ndarray: Array of predicted class probabilities, shape (n_samples, n_classes).
        """
        member_probas = self._collect_member_probas(X)
        return self._aggregate_probas(member_probas, strategy=aggregation_strategy)

    @abstractmethod
    def predict_uncertainty(
        self, X: pd.DataFrame, uncertainty_for_opt: bool = False, aggregation_strategy: str = "average"
    ) -> pd.DataFrame:
        """
        Predict class probabilities and estimate uncertainty for the input data X using the ensemble.

        Args:
            X (pd.DataFrame): DataFrame containing features.
            uncertainty_for_opt (bool): Flag indicating if uncertainty is for optimization purposes. Default is False.
            aggregation_strategy (str): Strategy to aggregate predictions from ensemble members. Default is "average".

        Returns:
            pd.DataFrame: DataFrame containing predicted class probabilities and uncertainty estimates (if
            uncertainty_for_opt=False) or only estimates of knowledge uncertainty (if uncertainty_for_opt=True).
        """
        raise NotImplementedError


class CVEnsembleDerivedClassifier(CVEnsembleClassifierMother):
    def __init__(self, estimators: list):
        super().__init__(estimators=estimators)

    def _calculate_uncertainty(
        self, estimator_probas: np.ndarray, uncertainty_method: EnsembleDerivedUncertaintyMethod | str
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if uncertainty_method == EnsembleDerivedUncertaintyMethod.ENTROPY:
            # This approach approximately captures the mutual information among ensemble members and
            # resulting knowledge uncertainty equals np.mean(kl_matrix, axis=0) with close-to-zero
            # numerical difference

            ensemble_probs = np.mean(estimator_probas, axis=0)  # shape: (n_samples, n_classes)

            # rel_entr computes p * log(p / q), here q=1.0 so it simplifies to p * log(p) (zero-safe)
            total_uncertainty = -np.sum(rel_entr(ensemble_probs, 1.0), axis=-1)  # shape: (n_samples,)

            data_uncertainty = np.mean(
                -np.sum(rel_entr(estimator_probas, 1.0), axis=-1),
                axis=0,
            )  # shape (n_samples,)

            knowledge_uncertainty = total_uncertainty - data_uncertainty
        elif uncertainty_method == EnsembleDerivedUncertaintyMethod.VARIANCE:
            # Variance across estimator probas for each sample
            knowledge_uncertainty = np.var(estimator_probas, axis=0).mean(axis=1)  # shape (n_samples,)
            data_uncertainty = None
            total_uncertainty = None
        elif uncertainty_method == EnsembleDerivedUncertaintyMethod.DISAGREEMENT:
            # Compute prediction of ensemble per sample (by averaging member predictions)
            ensemble_probs = np.mean(estimator_probas, axis=0)  # shape: (n_samples, n_classes)

            # Compute KL divergence for each member's prediction against ensemble prediction
            kl_matrix = np.sum(
                rel_entr(estimator_probas, ensemble_probs[None, :, :]), axis=-1
            )  # shape: (n_estimators, n_samples)

            # Sum up KL divergences across members to get disagreement per sample
            disagreement_score = np.sum(kl_matrix, axis=0)  # shape: (n_samples,)

            knowledge_uncertainty = disagreement_score
            data_uncertainty = None
            total_uncertainty = None
        else:
            raise ValueError(f"Unknown uncertainty estimation method for ensemble model: {uncertainty_method}")

        return knowledge_uncertainty, data_uncertainty, total_uncertainty

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        uncertainty_for_opt: bool = False,
        aggregation_strategy: str = "average",
        uncertainty_method: str | EnsembleDerivedUncertaintyMethod = EnsembleDerivedUncertaintyMethod.ENTROPY,
    ) -> pd.DataFrame:
        """
        Predicts ensemble labels, probabilities, and uncertainties for input data X.

        Args:
            X (pd.DataFrame): Input features.
            uncertainty_for_opt (bool): If True, only return knowledge uncertainty.
            aggregation_strategy (str): How to combine member predictions. Default is "average".
            uncertainty_method (str | EnsembleDerivedUncertaintyMethod): Method to estimate uncertainty.
                Current options are:
                - "entropy" (default): Computes the entropy of the average predicted probabilities across estimators.
                    Uncertainty decomposition (total = knowledge + data).
                - "variance": Computes the variance of predicted probabilities across estimators = knowledge uncertainty
                - "disagreement": Computes disagreement among estimators using sum of Kullback-Leibler divergences (see
                    Lakshminarayanan et al., 2017). Knowledge uncertainty only.

        Returns:
            pd.DataFrame: DataFrame containing predictions and uncertainties.
        """
        estimator_probas = self._collect_member_probas(X)
        knowledge_uncertainty, data_uncertainty, total_uncertainty = self._calculate_uncertainty(
            estimator_probas, uncertainty_method=uncertainty_method
        )

        ensemble_probas = self._aggregate_probas(estimator_probas, strategy=aggregation_strategy)
        ensemble_pred = self._probas_to_preds(ensemble_probas)

        if uncertainty_for_opt:
            return pd.DataFrame(
                {"knowledge_uncertainty": knowledge_uncertainty},
                index=X.index if isinstance(X, pd.DataFrame) else None,
            )

        pred_res = self._build_result(
            X, ensemble_pred, ensemble_probas, knowledge_uncertainty, data_uncertainty, total_uncertainty
        )

        return pred_res


class CVEnsembleAveragingClassifier(CVEnsembleClassifierMother):
    def __init__(self, estimators: list):
        super().__init__(estimators=estimators)

    def _collect_member_outputs(self, X: pd.DataFrame, **kwargs) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Predict probabilities for the input data X and uncertainties for all estimators in the ensemble.

        Args:
            X (pd.DataFrame): DataFrame containing features.

        Returns:
            tuple[np.ndarray, np.ndarray, list[str]]: Tuple of array, if predict_uncertainty is True, containing:
                - Array of predicted probabilities, shape (n_estimators, n_samples, n_classes).
                - Array of predicted uncertainties, shape (n_estimators, n_samples, n_uncertainty_types).
                - List of available uncertainty types.
        """
        probas = []
        uncertainties = []
        expected_types = []

        for estimator in self.estimators:
            prediction = estimator.predict_uncertainty(X, **kwargs)
            member_probas = self._extract_member_probas(prediction)

            probas.append(member_probas)

            member_uncertainties = []
            available_types = []
            for type in UncertaintyType:
                uncertainty_type = type.value
                uncertainty_column = prediction.filter(like=f"{uncertainty_type}_uncertainty", axis=1).columns
                if len(uncertainty_column) != 1:
                    continue

                member_uncertainty = prediction[uncertainty_column[0]].to_numpy(dtype=float)

                if (
                    not np.isfinite(member_uncertainty).all()
                    or np.any(member_uncertainty < 0)
                    or np.isnan(member_uncertainty).any()
                ):
                    continue
                member_uncertainties.append(member_uncertainty)
                available_types.append(uncertainty_type)

            if expected_types:
                if available_types != expected_types:
                    raise ValueError("Inconsistent uncertainty types available across estimators")
            else:
                expected_types = available_types

            if len(available_types) > 0:
                uncertainties.append(np.stack(member_uncertainties, axis=-1))
            else:
                raise ValueError("No valid member uncertainties available for any type of {}".format(uncertainty_type))

        probas_array = np.stack(probas, axis=0)
        uncertainties_array = np.stack(uncertainties, axis=0)
        return probas_array, uncertainties_array, available_types

    def _aggregate_uncertainties(
        self, estimator_uncertainties: np.ndarray, available_uncertainties: list[str]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Aggregate uncertainties from estimators by averaging.

        Args:
            estimator_uncertainties (np.ndarray): Uncertainties from each estimator.
                Shape (n_estimators, n_samples, n_types).
            available_uncertainties (list[str]): Types of uncertainties available.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Knowledge, data, and total uncertainties.
            Each of them can be None, if not in available_uncertainties.
        """
        if estimator_uncertainties is not None:
            average_uncertainties = np.mean(estimator_uncertainties, axis=0)  # shape(n_samples, n_available_types)

            uncertainty_by_type = {
                uncertainty_type: average_uncertainties[:, type_index]
                for type_index, uncertainty_type in enumerate(available_uncertainties)
            }

            knowledge_uncertainty = uncertainty_by_type.get("knowledge", None)
            data_uncertainty = uncertainty_by_type.get("data", None)
            total_uncertainty = uncertainty_by_type.get("total", None)
        else:
            raise ValueError("No estimator uncertainties provided.")

        return knowledge_uncertainty, data_uncertainty, total_uncertainty

    def predict_uncertainty(
        self, X: pd.DataFrame, uncertainty_for_opt: bool = False, aggregation_strategy: str = "average", **kwargs
    ) -> pd.DataFrame:
        """
        Predicts label, probabilities and uncertainties for the input data X.

        Args:
            X (pd.DataFrame): Input data.
            uncertainty_for_opt (bool): If True, return only knowledge uncertainty.
            aggregation_strategy (str): How to aggregate uncertainties. Default is "average": Computes the average
            decomposed uncertainty estimated by ensemble members individually, per available uncertainty type in
            the member prediction. All uncertainty types are considered: knowledge, data, total.
            **kwargs: Additional arguments for member models.

        Returns:
            pd.DataFrame: Predicted with label, probabilities and uncertainties columns.

        """
        estimator_probas, estimator_uncertainties, available_uncertainties = self._collect_member_outputs(X, **kwargs)
        ensemble_probas = self._aggregate_probas(estimator_probas, strategy=aggregation_strategy)
        ensemble_pred = self._probas_to_preds(ensemble_probas)
        knowledge_uncertainty, data_uncertainty, total_uncertainty = self._aggregate_uncertainties(
            estimator_uncertainties, available_uncertainties
        )

        if uncertainty_for_opt:
            return pd.DataFrame(
                {"knowledge_uncertainty": knowledge_uncertainty},
                index=X.index if isinstance(X, pd.DataFrame) else None,
            )

        pred_res = self._build_result(
            X, ensemble_pred, ensemble_probas, knowledge_uncertainty, data_uncertainty, total_uncertainty
        )
        return pred_res
