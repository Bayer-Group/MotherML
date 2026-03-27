import importlib
import inspect
import logging
from pathlib import Path
from typing import List, Optional, Type

from mother.ml.config import ModelConfig
from mother.ml.core import (
    AbstractMotherPipeline,
    ColumnTransformerWithHyperparameterRooting,
    FeatureUnionWithHyperparameterRooting,
    PipelineWithHyperparameterRooting,
)
from mother.ml.models.m_catboost import (
    CatboostClassifierMother,
    CatboostGaussianProcessRegressorMother,
    CatboostRankerMother,
    CatboostRegressorMother,
)
from mother.ml.utils import avg_ndcg_score

module_logger: logging.Logger = logging.getLogger(__name__)
# Initialize containers for automatically discovered model classes and algorithms


class MotherModelRegistry:
    """
    Singleton registry for dynamically discovering and managing model classes in the 'mother.ml.models' package.

    This class scans the models directory for Python files matching the pattern 'm_*.py', imports them,
    and registers all classes that inherit from AbstractMotherPipeline. It provides mappings for model class names,
    lower-case lookups, and a list of supported algorithms. The registry is used to facilitate model discovery,
    retrieval, and algorithm support checks throughout the mother.ml package.

    Attributes:
        models_dir (Path): Path to the directory containing model modules.
        model_classes (dict): Mapping of model class names to their class objects.
        model_classes_lower (dict): Mapping of lower-case model class names to their canonical names.
        supported_algorithms (list): List of supported algorithm names discovered from model files.
    """

    _instance = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            module_logger.debug("Initializing MotherModelRegistry")
            self.models_dir: Path = Path(__file__).parent.joinpath("models")
            self.model_classes: dict[str, type[AbstractMotherPipeline]] = {}
            self.model_classes_lower: dict[str, str] = {}
            self.supported_algorithms: dict[str, set[str]] = {}

    def _load_models(self) -> None:
        """Dynamically load model classes from the models directory."""
        if self._initialized:
            return  # Prevent re-initialization
        importlib.invalidate_caches()  # Ensure fresh import cache
        module_logger.debug("Loading models from directory: %s", self.models_dir)
        # Get all Python files in the models directory (excluding __init__.py)
        model_files: list = [f.stem for f in self.models_dir.glob("m_*.py")]

        # Import each model module and collect AbstractMotherPipeline implementations
        for model_file in model_files:
            try:
                module_logger.debug(f"Loading model module: {model_file}")
                module = importlib.import_module(f"mother.ml.models.{model_file}")
                # Ensure the module is loaded correctly
                if not hasattr(module, "__name__"):
                    module_logger.error(f"Module {model_file} does not have a __name__ attribute.")
                    continue
                # Find all classes in the module that inherit from AbstractMotherPipeline
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, AbstractMotherPipeline)
                        and obj.__module__ == module.__name__
                        and obj != AbstractMotherPipeline
                        and not name.startswith("_")
                    ):
                        self.model_classes[name] = obj
                        self.model_classes_lower[name.lower()] = name  # Add lower-case mapping

                        algo: str = model_file.lower().lstrip("m_")

                        if algo not in self.supported_algorithms:
                            self.supported_algorithms[algo] = set()
                        self.supported_algorithms[algo].add(name)
            except Exception as e:
                module_logger.warning(
                    (
                        f"Warning: Failed to load model from {model_file}: {str(e)} "
                        f"(Please, ignore this message if you don't need {model_file.split('m_')[1]})"
                    )
                )
        module_logger.debug("Model loading complete.")
        module_logger.info(f"Loaded {len(self.model_classes)} model classes: {', '.join(self.model_classes.keys())}")
        self._initialized = True

    def register_model(self, model_class: Type[AbstractMotherPipeline], algorithm: Optional[str] = None) -> None:
        """Register a model class manually.

        Args:
            model_class: The model class to register
            algorithm: Optional algorithm name. If not provided, will be derived from class name
        """
        if not issubclass(model_class, AbstractMotherPipeline):
            raise ValueError(f"Model class {model_class.__name__} must inherit from AbstractMotherPipeline")

        class_name = model_class.__name__

        # Add to model classes
        self.model_classes[class_name] = model_class
        self.model_classes_lower[class_name.lower()] = class_name

        # Determine algorithm name
        if algorithm is None:
            # Try to extract algorithm from class name
            if class_name.lower().startswith("mother"):
                algorithm = class_name[6:].lower()  # Remove "Mother" prefix
            elif class_name.lower().endswith("mother"):
                algorithm = class_name[:-6].lower()  # Remove "Mother" suffix
            else:
                algorithm = class_name.lower()

        # Add algorithm if not already present
        if algorithm not in self.supported_algorithms:
            self.supported_algorithms[algorithm] = set()
        self.supported_algorithms[algorithm].add(class_name)

        module_logger.info(f"Registered model class: {class_name} with algorithm: {algorithm}")

    def unregister_model(self, model_class_name: str) -> None:
        """Unregister a model class.

        Args:
            model_class_name: Name of the model class to unregister
        """
        if model_class_name in self.model_classes:
            del self.model_classes[model_class_name]
            if model_class_name.lower() in self.model_classes_lower:
                del self.model_classes_lower[model_class_name.lower()]
            module_logger.info(f"Unregistered model class: {model_class_name}")
        else:
            module_logger.warning(f"Model class {model_class_name} not found for unregistration")

    def list_registered_models(self) -> dict:
        """List all registered models with their algorithms."""
        register_models = {}
        for name, classes in self.supported_algorithms.items():
            for cls in classes:
                register_models[cls] = {
                    "class": self.model_classes[cls],
                    "algorithm": name,
                }
        # return {
        #     name: {
        #         "class": cls,
        #         "algorithm": next(
        #             (algo for algo in self.supported_algorithms.keys() if name.lower().startswith(algo.lower())),
        #             "unknown",
        #         ),
        #     }
        #     for name, classes in self.supported_algorithms.items()
        # }
        return register_models


def register_model(algorithm: Optional[str] = None):
    """Decorator to register a model class.

    Args:
        algorithm: Optional algorithm name

    Example:
        @register_model("my_algorithm")
        class MyCustomMother(AbstractMotherPipeline):
            pass
    """

    def decorator(model_class: Type[AbstractMotherPipeline]):
        _registry.register_model(model_class, algorithm)
        return model_class

    return decorator


# Create singleton instance
_registry: MotherModelRegistry = MotherModelRegistry()

__all__ = [
    "AbstractMotherPipeline",
    "CatboostRegressorMother",
    "CatboostClassifierMother",
    "CatboostGaussianProcessRegressorMother",
    "CatboostRankerMother",
    "ColumnTransformerWithHyperparameterRooting",
    "FeatureUnionWithHyperparameterRooting",
    "ModelConfig",
    "avg_ndcg_score",
    "PipelineWithHyperparameterRooting",
] + list(_registry.model_classes.keys())


def get_model_class_by_algorithm(algorithm: str) -> list[Type[AbstractMotherPipeline]]:
    """Get model classes by algorithm name.

    Args:
        algorithm: Name of the algorithm

    Returns:
        Type: The model class

    """
    if _registry._initialized is False:
        _registry._load_models()
    return [
        _registry.model_classes[class_]
        for name, class_set in _registry.supported_algorithms.items()
        if name.lower().startswith(algorithm.lower())
        for class_ in class_set
    ]


def get_model_class_by_algorithm_and_type(algorithm: str, model_type: str) -> type[AbstractMotherPipeline]:
    """
    Returns the appropriate model class based on the algorithm and model type.
    """
    models: List[str] = [m.lower() for m in get_supported_models()]
    __estimator_type: str
    __model_type: Optional[str] = None

    res = model_type.split("_", 1)
    if len(res) == 2:
        __estimator_type, __model_type = res
    else:
        __estimator_type = res[0]

    if __estimator_type == "classification":
        __estimator_type = "classifier"
    elif __estimator_type == "regression":
        __estimator_type = "regressor"
    elif __estimator_type == "ranking":
        __estimator_type = "ranker"
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'classification' or 'regression'.")

    selected_models: List[str] = [m for m in models if m.startswith(algorithm.lower()) and __estimator_type in m]
    if __model_type is not None:
        if len(selected_models) != 1:
            selected_models = [m for m in selected_models if __model_type in m]
        elif __model_type not in ["binary", "multiclass"]:
            selected_models = []
    elif algorithm == "catboost":
        # exclude gaussian process regression
        selected_models = [m for m in selected_models if "gaussian" not in m]

    if len(selected_models) != 1:
        raise ValueError(f"Unsupported algorithm '{algorithm}' or model type '{model_type}'. ")
    return get_model_class(name=selected_models[0])


def algo_is_supported(algorithm: str) -> bool:
    """Check if the specified algorithm is supported.

    Args:
        algorithm: Name of the algorithm to check

    Returns:
        bool: True if supported, False otherwise
    """
    if _registry._initialized is False:
        _registry._load_models()
    return algorithm.lower() in _registry.supported_algorithms


def get_available_algorithms() -> List[str]:
    """Get a list of all supported algorithms.

    Returns:
        List[str]: Names of supported algorithms
    """
    if _registry._initialized is False:
        _registry._load_models()
    return list(_registry.supported_algorithms.keys())


def get_model_class(name: str) -> type[AbstractMotherPipeline]:
    """Get a model class by name.

    Args:
        name: Name of the model class

    Returns:
        Type: The model class

    Raises:
        KeyError: If the model class is not found
    """
    if _registry._initialized is False:
        _registry._load_models()
    if name in _registry.model_classes:
        return _registry.model_classes[name]
    # Try lower-case mapping
    lower_name = name.lower()
    if lower_name in _registry.model_classes_lower:
        return _registry.model_classes[_registry.model_classes_lower[lower_name]]
    raise KeyError(f"Model class '{name}' not found")


def get_supported_models() -> List[str]:
    """Get a list of all supported model class names.

    Returns:
        List[str]: Names of supported model classes
    """
    if _registry._initialized is False:
        _registry._load_models()
    return list(_registry.model_classes.keys())


def describe_model(name: str) -> str:
    """Get help text for a model class.

    Args:
        name: Name of the model class

    Returns:
        str: Help text for the model class
    """
    if _registry._initialized is False:
        _registry._load_models()
    # Instead of calling help() directly (which prints to stdout)
    # Use the inspect module to get the docstring
    if name in _registry.model_classes:
        model_class = _registry.model_classes[name]
        import inspect

        docstring = inspect.getdoc(model_class) or "No documentation available"

        # Format for better display
        result = f"## {name}\n\n{docstring}\n\n"

        # Add method documentation if desired
        methods = ["get_hyperparameter_space", "default_parameters"]
        for method_name in methods:
            if hasattr(model_class, method_name):
                method = getattr(model_class, method_name)
                method_doc = inspect.getdoc(method) or "No documentation available"
                result += f"### {method_name}\n\n{method_doc}\n\n"

        result += f"For more information on the parent class just use 'help(ml.get_model_class(\"{name}\")'"
        return result
    raise KeyError(f"Model class '{name}' not found")
