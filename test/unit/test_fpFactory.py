import pytest
from rdkit.Chem import rdFingerprintGenerator as rdFG

from mother.feature_generation.fp_gen import FingerprintFactory


def test_initialization() -> None:
    # Test valid initialization
    params = {"n_bits": 1024, "countSimulation": True}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    assert factory.fp_type == "MorganFP"
    assert factory.parameters["fpSize"] == 1024
    assert factory.parameters["countSimulation"] is True

    # Test renaming of n_bits to fpSize
    assert "n_bits" not in factory.parameters
    assert "fpSize" in factory.parameters

    # Test default values
    params = {}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    assert factory.parameters["fpSize"] == 2048
    assert factory.parameters["countSimulation"] is False

    # Test invalid fingerprint type
    with pytest.raises(ValueError, match="Unknown fingerprint type: InvalidFP"):
        FingerprintFactory(fp_type="InvalidFP", parameters=params)


@pytest.mark.parametrize("fp_type", ["MorganFP", "AtomPairFP", "RDKitFP", "TopologicalTorsionFP"])
def test_get_fingerprint_generator(fp_type) -> None:
    params = {"fpSize": 1024, "countSimulation": True}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    generator = factory.get_fingerprint_generator()
    assert isinstance(generator, rdFG.FingerprintGenerator64)


def test_get_fingerprint_generator_raises() -> None:
    with pytest.raises(
        ValueError,
    ):
        params = {"fpSize": 1024, "countSimulation": True}
        FingerprintFactory(fp_type="InvalidFP", parameters=params)


def test_fpSize_property() -> None:
    params = {"fpSize": 1024}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    assert factory.fpSize == 1024


def test_countSimulation_property() -> None:
    params = {"countSimulation": True}
    factory = FingerprintFactory(fp_type="MorganFP", parameters=params)
    assert factory.countSimulation is True


def test_is_supported() -> None:
    assert FingerprintFactory.is_supported("MorganFP") is True
    assert FingerprintFactory.is_supported("InvalidFP") is False


if __name__ == "__main__":
    pytest.main()
