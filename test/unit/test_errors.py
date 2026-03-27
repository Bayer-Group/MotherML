from mother.errors import ConfigurationError, ExtrasDependencyImportError


def test_extras_dependency_import_error():
    nested_error = ImportError("No module named 'example'")
    error = ExtrasDependencyImportError("example", nested_error)

    assert isinstance(error, ExtrasDependencyImportError)
    assert str(nested_error) in str(error)
    assert "pip install 'mother[example]'" in str(error)
    assert "uv add 'mother[example]'" in str(error)


def test_configuration_error():
    error = ConfigurationError("Configuration is invalid")

    assert isinstance(error, ConfigurationError)
    assert str(error) == "Configuration is invalid"
