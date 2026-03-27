import inspect
import logging
from abc import ABC, abstractmethod

import pandas as pd
from optuna.trial import Trial
from sklearn import compose as skl_comp
from sklearn import pipeline as skl_pipe

module_logger = logging.getLogger(__name__)


class AbstractMotherPipeline(ABC):
    """
    The abstract Mother pipeline is a conventional sklearn estimator / transformer etc. but adds methods for
    hyperparameter definition. Furthermore, it ensures for non sklearn classes and derived classes that they are
    compatible to the sklearn pipeline interface. This is done by implementing the get_params and set_params methods.
    """

    @abstractmethod
    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        raise NotImplementedError

    def default_parameters(self, prefix: str = "") -> dict:
        return {}

    def get_all_params(self) -> dict:
        if hasattr(super, "get_all_params"):
            return super.get_all_params()  # type: ignore
        module_logger.warning("Could not find get_all_params method in the parent class (%s).", self.__class__.__name__)
        return {}

    @abstractmethod
    def set_params(self, **params):
        raise NotImplementedError(
            "set_params method is not implemented. Please implement this method in the derived class."
        )

    @abstractmethod
    def get_params(self, deep=True) -> dict:
        raise NotImplementedError(
            "get_params method is not implemented. Please implement this method in the derived class."
        )

    def predict_uncertainty(self, X: pd.DataFrame, **kwargs):
        """
        This method needs to be implemented for the models that supports uncertainty.
        Otherwise, the model will run self.predict(X).

        Args:
            X (pd.DataFrame): input features to predict target value
        """
        module_logger.warning(
            f"Uncertainty quantification is not implemented for {self.__class__.__name__}."
            "predict() function will be run."
        )
        pred_res = self.predict(X)

        # make a standardised output data frame
        # this only works for single-task output (not supporting multi-task learning)
        if len(pred_res.shape) == 1:
            # single dim output
            pred_res = pd.DataFrame(
                {
                    "mean_predictions": pred_res,
                    "knowledge_uncertainty": None,
                    "data_uncertainty": None,
                    "total_uncertainty": None,
                },
            )
        elif (len(pred_res.shape) == 2) and (pred_res.shape[-1] == 1):
            # 2D output with one col
            pred_res = pd.DataFrame(
                {
                    "mean_predictions": pred_res.iloc[:, -1] if isinstance(pred_res, pd.DataFrame) else pred_res[:, -1],
                    "knowledge_uncertainty": None,
                    "data_uncertainty": None,
                    "total_uncertainty": None,
                },
            )

        if isinstance(X, pd.DataFrame) and isinstance(pred_res, pd.DataFrame):
            pred_res.set_index(X.index, inplace=True)

        return pred_res


class _GroupIdArgInFitFunctionMixin:
    """helper class to pass the keyword 'group_id' to the fit function of a pipeline step"""

    def fit(self, X, y=None, **fit_params):
        """
        Fit the pipeline to the given data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs), optional
            The target values.
        **fit_params : keyword arguments
            Additional parameters to pass to the fit method of each step.
        """
        group_id = fit_params.pop("group_id", None)

        for step_name, step in getattr(self, self.iterative_steps):
            if hasattr(step, "fit"):
                if group_id is not None:
                    step.fit(X, y, group_id=group_id, **fit_params)
                else:
                    step.fit(X, y, **fit_params)

        return self


class _SklComposeWithHyperparameterRooting(AbstractMotherPipeline):
    """The _SklComposeWithHyperparameterRooting class is a utility class designed to facilitate
    the composition of hyperparameter spaces for pipeline and column transformer steps.
    It inherits from AbstractMotherPipeline, which defines the structure for hyperparameter-related methods.
    The class contains a class attribute iterative_steps, which specifies the attribute name that holds the steps
    of the pipeline or column transformer.
    The core functionality of this class is provided by the
    _dict_concatenation_on_pipelinestep_method_with_prefix method. This method takes a method name and a dictionary of
    method arguments, iterates over the steps defined in iterative_steps, and checks if each step object has the
    specified method. If the method exists, it is called with the provided arguments, and the results are concatenated
    into a single dictionary with prefixed keys to avoid naming conflicts.

    The class also defines two public methods: get_hyperparameter_space and default_parameters. These methods utilize
    _dict_concatenation_on_pipelinestep_method_with_prefix to gather and concatenate the results from the corresponding
    methods of each step object. This allows for the seamless integration and management of hyperparameters across
    different steps of a pipeline or column transformer.
    """

    iterative_steps: str = "steps"  # name of the attribute that contains the steps (different for Pipeline, and
    # ColumnTransformer)

    def _dict_concatenation_on_pipelinestep_method_with_prefix(
        self, method_name: str, method_kwargs: dict, prefix: str
    ) -> dict:
        out = {}
        for step_ in getattr(self, self.iterative_steps):
            stepname, obj = (step_[0], step_[1]) if len(step_) > 2 else step_
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)

                out = {**out, **method(**method_kwargs, prefix=prefix + stepname + "__")}

        return out

    def get_hyperparameter_space(self, X, y, trial: Trial, prefix: str = "") -> dict:
        suggested_params = self._dict_concatenation_on_pipelinestep_method_with_prefix(
            "get_hyperparameter_space", {"X": X, "y": y, "trial": trial}, prefix=prefix
        )
        return suggested_params

    def default_parameters(self, prefix: str = "") -> dict:
        return self._dict_concatenation_on_pipelinestep_method_with_prefix("default_parameters", {}, prefix=prefix)


class PipelineWithHyperparameterRooting(skl_pipe.Pipeline, _SklComposeWithHyperparameterRooting):
    iterative_steps: str = "steps"

    def predict_uncertainty(self, X, **kwargs):
        """Transform the data and apply predict_uncertainty with the final estimator.

        Call transform of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls predict_uncertainty.

        Step-specific parameters are passed to intermediate transformers and the final
        estimator using sklearn's naming convention (step_name__param_name). The method
        inspects the signature of each step's transform method to determine if it accepts
        keyword arguments (has **kwargs). Only steps that accept keyword arguments will
        receive their step-specific parameters.

        Parameters
        ----------
        X : array-like
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        **kwargs : dict
            Parameters passed to pipeline steps using sklearn's naming convention
            (step_name__param_name). For example:
            - ml_model__n_ensembles=100  -> passed to final estimator's predict_uncertainty
            - feature_selector__param=value -> passed to feature_selector's transform,
              but only if its transform method signature includes **kwargs

        Raises
        ------
        ValueError
            If any parameter does not follow the step_name__param_name convention.
        TypeError
            If any non-final pipeline step does not implement a transform method.

        Returns
        -------
        result : DataFrame or array-like
            Result of calling predict_uncertainty on the final estimator.
        """
        try:
            # Validate that all kwargs follow step__param convention (like sklearn)
            step_names = {step_name for step_name, _ in self.steps}
            unmatched_params = set()

            for key in kwargs:
                if "__" not in key:
                    unmatched_params.add(key)
                else:
                    step_prefix = key.split("__")[0]
                    if step_prefix not in step_names:
                        unmatched_params.add(key)

            if unmatched_params:
                raise ValueError(
                    f"predict_uncertainty() got unexpected keyword argument(s): {sorted(unmatched_params)}. "
                    f"Parameters must follow the pattern 'step_name__param_name' where step_name is one of: "
                    f"{sorted(step_names)}. For example: ml_model__n_ensembles=100"
                )

            # Separate kwargs by step name prefix (sklearn compatibility)
            step_kwargs = {}
            for step_name, _ in self.steps:
                step_kwargs[step_name] = {}
                prefix = step_name + "__"
                for key, value in kwargs.items():
                    if key.startswith(prefix):
                        # Remove the prefix from the parameter name
                        param_name = key[len(prefix) :]
                        step_kwargs[step_name][param_name] = value

            # Transform through all intermediate steps
            Xt = X
            for idx in range(len(self.steps) - 1):
                step_name, step = self.steps[idx]

                if not hasattr(step, "transform"):
                    raise TypeError(
                        f"Pipeline step {idx} ({type(step).__name__}) does not implement a 'transform' method. "
                        "All non-final steps must be transformers."
                    )

                # Check if transform method accepts **kwargs by inspecting its signature
                transform_signature = inspect.signature(step.transform)
                accepts_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD for param in transform_signature.parameters.values()
                )

                # Pass step-specific kwargs to transform only if the method accepts them
                if accepts_kwargs and step_kwargs[step_name]:
                    Xt = step.transform(Xt, **step_kwargs[step_name])
                else:
                    Xt = step.transform(Xt)

            final_step_name, final_estimator = self.steps[-1]

            return final_estimator.predict_uncertainty(Xt, **step_kwargs[final_step_name])
        except AttributeError as e:
            module_logger.error(
                "Error during predict_uncertainty. "
                "Ensure that all non-final steps implement .transform() "
                "and the final estimator implements .predict_uncertainty()."
            )
            raise e


class ColumnTransformerWithHyperparameterRooting(skl_comp.ColumnTransformer, _SklComposeWithHyperparameterRooting):
    """adds hyperparameter spaces from all column-transformer-steps together, and adds prefixes as necessary
    to parameter names"""

    iterative_steps: str = "transformers"


class FeatureUnionWithHyperparameterRooting(skl_pipe.FeatureUnion, _SklComposeWithHyperparameterRooting):
    iterative_steps: str = "transformer_list"
