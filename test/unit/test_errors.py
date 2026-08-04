import builtins
import importlib
import sys

import pytest

from mother.errors import ConfigurationError, ExtrasDependencyImportError


def test_extras_dependency_import_error():
    nested_error = ImportError("No module named 'example'")
    error = ExtrasDependencyImportError("example", nested_error)

    assert isinstance(error, ExtrasDependencyImportError)
    assert str(nested_error) in str(error)
    assert "pip install 'mother-ml[example]'" in str(error)
    assert "uv add mother-ml --extra example" in str(error)
    assert "uv sync --extra example" in str(error)


def test_configuration_error():
    error = ConfigurationError("Configuration is invalid")

    assert isinstance(error, ConfigurationError)
    assert str(error) == "Configuration is invalid"


@pytest.mark.parametrize(
    ("module_name", "import_error"),
    [
        ("torch", ModuleNotFoundError("No module named 'torch'")),
        ("tabpfn", ImportError("tabpfn is incompatible with scikit-learn")),
    ],
    ids=["missing-torch", "tabpfn-import-error"],
)
def test_tabpfn_optional_dependency_imports_raise_extras_error(monkeypatch, module_name, import_error):
    target_module = "mother.ml.models.m_tabpfn"
    sys.modules.pop(target_module, None)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            raise import_error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ExtrasDependencyImportError) as exc_info:
        importlib.import_module(target_module)

    assert str(import_error) in str(exc_info.value)
    assert "mother-ml[tabpfn]" in str(exc_info.value)
