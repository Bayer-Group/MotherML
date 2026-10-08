import logging

import numpy as np
import pandas as pd
from optuna.trial import Trial
from scipy.special import rel_entr
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from mother.ml.core import AbstractMotherPipeline

module_logger: logging.Logger = logging.getLogger(__name__)


class CVEnsembleClassifierMother(AbstractMotherPipeline, BaseEstimator):
    """
    Ensemble classifier that combines predictions from multiple pre-fitted estimators. As it inherits from
    `BaseEstimator`, it follows the scikit-learn estimator interface.

    Note:
        - Only single-target predictions are supported.
        - Assumes that estimators are pre-fitted and can provide predicted probabilities and uncertainties with the
          `predict_uncertainty` method.

    Attributes:
        estimators (list): List of pre-fitted estimators to be used in the ensemble. Each estimator must have a
        `classes_` attribute and a `predict_uncertainty` method.

    Methods:
        get_hyperparameter_space: Returns the hyperparameter search space for the ensemble (no tunable hyperparameters
        so far, so an empty dictionary is returned).
        set_params: Sets the parameters of the ensemble (not allowed so far, so self is returned unchanged).
        get_params: Gets the parameters of the ensemble.
        fit: Fits the ensemble (no-op as estimators are pre-fitted).
        predict: Predicts labels using the aggregated outputs of the ensemble members.
        predict_uncertainty: Predicts uncertainties associated with the ensemble's predictions.
    """

    def __init__(self, estimators: list):
        self.classes_ = self._validate_estimators(estimators)
        self.estimators = estimators

    @staticmethod
    def _validate_estimators(estimators: list) -> tuple[np.ndarray, list[list[int]]]:
        """
        Validate the provided estimators for the ensemble.

        Args:
            estimators (list): List of pre-fitted estimators to be used in the ensemble.

        Returns:
            np.ndarray: Array of class labels shared by all estimators (one-dimensional and distinct).

        Raises:
            ValueError: If the estimators list has fewer than two members, if any estimator is not fitted, if any
            estimator lacks the required attributes, or if class labels are inconsistent across estimators.
        """
        if len(estimators) < 2:
            raise ValueError("At least two estimators must be provided as ensemble members.")

        # Validate each estimator and collect class labels
        classes = None
        for i, estimator in enumerate(estimators):
            try:
                check_is_fitted(estimator)
            except NotFittedError as e:
                raise ValueError(f"Estimator {i} is not fitted.") from e

            if not hasattr(estimator, "classes_"):
                raise ValueError(f"Estimator {i} must have a 'classes_' attribute.")
            if not callable(getattr(estimator, "predict_uncertainty", None)):
                raise ValueError(f"Estimator {i} must have a 'predict_uncertainty' method.")

            member_classes = np.asarray(estimator.classes_)
            if (
                member_classes.ndim != 1
                or len(member_classes) == 0
                or len(np.unique(member_classes)) != len(member_classes)
            ):
                raise ValueError(f"Estimator {i} must have distinct one-dimensional class labels (classes_).")
            if classes is None:
                classes = member_classes
            if not (member_classes == classes).all():
                raise ValueError("All estimators must have the same class labels.")

        return classes

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

    def get_params(self, deep=True) -> dict:
        """
        Retrieves the parameters of the ensemble.

        Args:
            deep (bool): If True, will return the parameters of the estimators as well.

        Returns:
            dict: Dictionary containing the parameters of the ensemble and its estimators.
        """
        params = {"estimators": self.estimators}

        if deep:
            params.update({**{f"estimator_{i}": est.get_params(deep=deep) for i, est in enumerate(self.estimators)}})

        return params

    def fit(self, X, y, **fit_params):
        """
        Fits the ensemble. Since the ensemble members are pre-fitted, this method does not perform any refitting.

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

    def _aggregate_predictions(self, probas: np.ndarray, strategy: str = "average") -> np.ndarray:
        """
        Aggregates predicted probabilities from multiple estimators using the specified strategy.

        Args:
            probas (np.ndarray): Predicted probabilities from multiple estimators,
            shape (n_estimators, n_samples, n_classes).
            strategy (str): Strategy for aggregating predictions. Current options are:
                - "average": Aggregates predicted probabilities across estimators by computing the mean for each
                  sample and assigns the class label with highest mean probability.

        Returns:
            np.ndarray: Aggregated predicted probabilities for each sample, shape (n_samples, n_classes).
        """

        # input validation
        if probas.ndim != 3:  # shape (n_estimators, n_samples, n_classes)
            raise ValueError(
                f"Unexpected dimension for probas: {probas.ndim}. Should be 3 (n_estimators, n_samples, n_classes)."
            )

        if strategy == "average":
            mean_probas = np.mean(
                probas, axis=0
            )  # Mean across estimators for each sample -> shape (n_samples, n_classes)
            return mean_probas
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _estimate_knowledge_uncertainty(
        self, probas: np.ndarray, uncertainties: np.ndarray = None, method: str = "variance"
    ) -> np.ndarray:
        """
        Estimates uncertainty from predicted probabilities of multiple estimators using the specified method.

        Args:
            probas (np.ndarray): Predicted probabilities from multiple estimators,
              shape (n_estimators, n_samples, n_classes).
            uncertainties (np.ndarray, optional): Predicted knowledge uncertainties from multiple estimators,
              shape (n_estimators, n_samples).
            method (str): Method for uncertainty estimation. Current options are:
                - "variance": Computes the variance of predicted probabilities across estimators.
                - "disagreement": Computes disagreement among estimators using Kullback-Leibler divergence
                  (see Lakshminarayanan et al., 2017).
                - "member_average": Computes the average knowledge uncertainty estimated by ensemble members
                individually.

        Returns:
            np.ndarray: Estimated knowledge uncertainty for each sample, shape (n_samples,).
        """

        if method == "variance":
            return np.var(probas, axis=0).mean(
                axis=1
            )  # Variance across estimators for each sample -> shape (n_samples,)
        elif method == "disagreement":
            n_estimators, n_samples, _ = probas.shape

            # Compute prediction of ensemble per sample (by averaging member predictions)
            ensemble_probs = np.mean(probas, axis=0)  # shape: (n_samples, n_classes)

            # Compute KL divergence for each member's prediction against ensemble prediction
            kl_divs = np.zeros((n_estimators, n_samples))

            for e in range(n_estimators):
                kl_divs[e] = np.sum(rel_entr(probas[e], ensemble_probs), axis=1)  # shape: (n_samples,)

            # Sum up KL divergences across members to get disagreement per sample
            disagreement_score = np.sum(kl_divs, axis=0)  # shape: (n_samples,)

            return disagreement_score
        elif method == "member_average":
            member_average = np.mean(uncertainties, axis=0)  # shape (n_samples,)
            return member_average
        else:
            raise ValueError(f"Unknown uncertainty estimation method for ensemble model: {method}")

    def _predict_probas_and_uncertainties(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict probabilities for the input data X and knowledge uncertainties for all estimators in the ensemble.

        Args:
            X (pd.DataFrame): DataFrame containing features.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Array of predicted probabilities, shape (n_estimators, n_samples, n_classes).
                - Array of predicted knowledge uncertainties, shape (n_estimators, n_samples).
        """
        probas = []
        knowledge_uncertainties = []
        proba_columns = [f"proba_{i}" for i in range(len(self.classes_))]
        for i, estimator in enumerate(self.estimators):
            prediction = estimator.predict_uncertainty(X)
            member_probas = prediction.loc[:, proba_columns].to_numpy(dtype=float)
            if (
                not np.isfinite(member_probas).all()
                or np.any((member_probas < 0) | (member_probas > 1))
                or not np.allclose(np.sum(member_probas, axis=1), 1)
            ):
                raise ValueError(f"Estimator {i} returned invalid class probabilities.")
            probas.append(member_probas)

            knowledge_uncertainty_column = prediction.filter(like="knowledge_uncertainty", axis=1).columns
            if len(knowledge_uncertainty_column) != 1:
                raise ValueError(f"Estimator {i} must return exactly one knowledge uncertainty column.")

            member_knowledge_uncertainties = prediction[knowledge_uncertainty_column[0]].to_numpy(dtype=float)
            if not np.isfinite(member_knowledge_uncertainties).all() or np.any(member_knowledge_uncertainties < 0):
                raise ValueError(f"Estimator {i} returned invalid knowledge uncertainties.")
            knowledge_uncertainties.append(member_knowledge_uncertainties)

        probas = np.stack(probas, axis=0)
        knowledge_uncertainties = np.stack(knowledge_uncertainties, axis=0)

        return probas, knowledge_uncertainties

    def _convert_class_index_to_label(self, class_indices: np.ndarray) -> np.ndarray:
        """
        Convert class indices to class labels using the 'classes_' attribute.

        Args:
            class_indices (np.ndarray): Array of class indices.

        Returns:
            np.ndarray: Array of class labels corresponding to the input indices.
        """
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
        estimator_probas, _ = self._predict_probas_and_uncertainties(X)
        ensemble_probas = self._aggregate_predictions(estimator_probas, strategy=aggregation_strategy)
        pred_label_index = np.argmax(ensemble_probas, axis=1)
        predictions = self._convert_class_index_to_label(pred_label_index)
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
        estimator_probas, _ = self._predict_probas_and_uncertainties(X)
        return self._aggregate_predictions(estimator_probas, strategy=aggregation_strategy)

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        uncertainty_for_opt: bool = False,
        aggregation_strategy: str = "average",
        uncertainty_method: str = "variance",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict class probabilities and estimate uncertainty for the input data X using the ensemble.

        Args:
            X (pd.DataFrame): DataFrame containing features.
            uncertainty_for_opt (bool): Flag indicating if uncertainty is for optimization purposes. Default is False.
            aggregation_strategy (str): Strategy to aggregate predictions from ensemble members. Default is "average".
            uncertainty_method (str): Method to estimate uncertainty. Default is "variance".
            **kwargs: Additional keyword arguments for uncertainty estimation.

        Returns:
            pd.DataFrame: DataFrame containing predicted class probabilities and uncertainty estimates (if
            uncertainty_for_opt=False) or only estimates of knowledge uncertainty (if uncertainty_for_opt=True).
        """

        # 1. predict probabilities
        estimator_probas, estimator_knowledge_uncertainties = self._predict_probas_and_uncertainties(X)
        ensemble_probas = self._aggregate_predictions(estimator_probas, strategy=aggregation_strategy)
        ensemble_pred = self._convert_class_index_to_label(np.argmax(ensemble_probas, axis=1))

        # 2. estimate uncertainty using a specified method: variance, entropy, disagreement (extendable)
        knowledge_uncertainty = self._estimate_knowledge_uncertainty(
            probas=estimator_probas, uncertainties=estimator_knowledge_uncertainties, method=uncertainty_method
        )

        if uncertainty_for_opt:
            return pd.DataFrame(
                {"knowledge_uncertainty": knowledge_uncertainty},
                index=X.index if isinstance(X, pd.DataFrame) else None,
            )

        pred_res = pd.concat(
            [
                pd.DataFrame(ensemble_pred, columns=["pred"]),
                pd.DataFrame(ensemble_probas, columns=[f"proba_{i}" for i in range(ensemble_probas.shape[1])]),
                pd.DataFrame(knowledge_uncertainty, columns=["knowledge_uncertainty"]),
                pd.DataFrame(None, columns=["data_uncertainty"]),
                pd.DataFrame(None, columns=["total_uncertainty"]),
            ],  # add total / data uncertainties later
            axis=1,
        )

        return pred_res
