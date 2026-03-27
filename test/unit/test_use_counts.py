"""
Tests for the use_counts feature in FingerprintsGeneric and MorganFingerprints.

Red-Green approach:
- These tests verify that use_counts=True produces count-based fingerprints
  (values > 1 possible) while use_counts=False produces binary fingerprints
  (values only 0 or 1).
- If use_counts logic were removed and everything used binary fingerprints,
  tests asserting count values > 1 or differing outputs would FAIL (RED).
"""

import numpy as np
import pytest
from rdkit.Chem import MolFromSmiles

from mother.feature_generation.core import FingerprintsGeneric, MorganFingerprints


@pytest.fixture
def mols_with_repeated_substructures():
    """Molecules chosen to have repeated substructures, producing count values > 1."""
    smiles = [
        "c1ccc(cc1)c2ccccc2",  # Biphenyl — repeated aromatic rings
        "C1CCCCC1C1CCCCC1",  # Bicyclohexyl — repeated ring substructures
        "CCCCCCCCCCCCCCCC",  # Hexadecane — many repeated CH2 fragments
        "c1ccc2c(c1)ccc3ccccc32",  # Phenanthrene — fused rings
    ]
    return np.array([MolFromSmiles(s) for s in smiles])


class TestUseCountsFingerprintsGeneric:
    """Tests for use_counts on FingerprintsGeneric."""

    def test_binary_fingerprints_only_zero_and_one(self, mols_with_repeated_substructures):
        """RED if use_counts default were True: binary FPs should contain only 0s and 1s."""
        fg = FingerprintsGeneric(
            fp_type="MorganFP",
            parameters={"radius": 2, "fpSize": 2048},
            use_counts=False,
        )
        fg.fit()
        result = fg.transform(mols_with_repeated_substructures)

        unique_values = np.unique(result[~np.isnan(result)])
        assert set(unique_values).issubset({0, 1}), (
            f"Binary fingerprints should only contain 0 and 1, got {unique_values}"
        )

    def test_count_fingerprints_can_exceed_one(self, mols_with_repeated_substructures):
        """RED if use_counts were ignored: count FPs should have values > 1 for molecules
        with repeated substructures."""
        fg = FingerprintsGeneric(
            fp_type="MorganFP",
            parameters={"radius": 2, "fpSize": 2048},
            use_counts=True,
        )
        fg.fit()
        result = fg.transform(mols_with_repeated_substructures)

        max_val = np.nanmax(result)
        assert max_val > 1, (
            f"Count fingerprints should have values > 1 for molecules with repeated "
            f"substructures, but max value was {max_val}"
        )

    def test_count_and_binary_differ(self, mols_with_repeated_substructures):
        """RED if use_counts had no effect: binary and count outputs must differ."""
        fg_binary = FingerprintsGeneric(
            fp_type="MorganFP",
            parameters={"radius": 2, "fpSize": 2048},
            use_counts=False,
        )
        fg_count = FingerprintsGeneric(
            fp_type="MorganFP",
            parameters={"radius": 2, "fpSize": 2048},
            use_counts=True,
        )

        fg_binary.fit()
        fg_count.fit()

        binary_result = fg_binary.transform(mols_with_repeated_substructures)
        count_result = fg_count.transform(mols_with_repeated_substructures)

        assert binary_result.shape == count_result.shape
        assert not np.array_equal(binary_result, count_result), (
            "Count and binary fingerprints should produce different outputs"
        )

    def test_use_counts_default_is_false(self):
        """RED if default changed: use_counts should default to False."""
        fg = FingerprintsGeneric(
            fp_type="MorganFP",
            parameters={"radius": 2, "fpSize": 2048},
        )
        assert fg.use_counts is False

    def test_output_shape_consistent(self, mols_with_repeated_substructures):
        """Both modes should produce the same output shape."""
        fp_size = 512
        fg_binary = FingerprintsGeneric(
            fp_type="MorganFP",
            parameters={"radius": 2, "fpSize": fp_size},
            use_counts=False,
        )
        fg_count = FingerprintsGeneric(
            fp_type="MorganFP",
            parameters={"radius": 2, "fpSize": fp_size},
            use_counts=True,
        )

        fg_binary.fit()
        fg_count.fit()

        binary_result = fg_binary.transform(mols_with_repeated_substructures)
        count_result = fg_count.transform(mols_with_repeated_substructures)

        n_mols = len(mols_with_repeated_substructures)
        assert binary_result.shape == (n_mols, fp_size)
        assert count_result.shape == (n_mols, fp_size)


class TestUseCountsMorganFingerprints:
    """Tests for use_counts on the MorganFingerprints convenience class."""

    def test_morgan_binary_default(self, mols_with_repeated_substructures):
        """MorganFingerprints with default use_counts=False should produce binary output."""
        fg = MorganFingerprints(radius=2, fpSize=1024)
        fg.fit()
        result = fg.transform(mols_with_repeated_substructures)

        unique_values = np.unique(result[~np.isnan(result)])
        assert set(unique_values).issubset({0, 1})

    def test_morgan_count_mode(self, mols_with_repeated_substructures):
        """RED if use_counts not wired through MorganFingerprints: should produce counts > 1."""
        fg = MorganFingerprints(radius=2, fpSize=1024, use_counts=True)
        fg.fit()
        result = fg.transform(mols_with_repeated_substructures)

        max_val = np.nanmax(result)
        assert max_val > 1, f"MorganFingerprints with use_counts=True should produce values > 1, got max={max_val}"

    def test_morgan_use_counts_preserved_after_set_params(self, mols_with_repeated_substructures):
        """use_counts should persist after set_params on other parameters."""
        fg = MorganFingerprints(radius=2, fpSize=1024, use_counts=True)
        fg.set_params(radius=3)
        fg.fit()
        result = fg.transform(mols_with_repeated_substructures)

        max_val = np.nanmax(result)
        assert max_val > 1, "use_counts should still be True after set_params(radius=3)"

    def test_morgan_use_counts_toggled_via_set_params(self, mols_with_repeated_substructures):
        """RED if use_counts isn't a proper sklearn param: set_params should toggle it."""
        fg = MorganFingerprints(radius=2, fpSize=1024, use_counts=False)
        fg.set_params(use_counts=True)
        fg.fit()
        result = fg.transform(mols_with_repeated_substructures)

        max_val = np.nanmax(result)
        assert max_val > 1, "set_params(use_counts=True) should enable count mode"

    def test_morgan_clone_preserves_use_counts(self):
        """sklearn.clone should preserve use_counts parameter."""
        from sklearn.base import clone

        fg = MorganFingerprints(radius=2, fpSize=1024, use_counts=True)
        fg_cloned = clone(fg)

        assert fg_cloned.use_counts is True
        assert fg_cloned.get_params()["use_counts"] is True

    def test_morgan_get_params_includes_use_counts(self):
        """use_counts should be visible in get_params()."""
        fg = MorganFingerprints(radius=2, fpSize=1024, use_counts=True)
        params = fg.get_params()
        assert "use_counts" in params
        assert params["use_counts"] is True
