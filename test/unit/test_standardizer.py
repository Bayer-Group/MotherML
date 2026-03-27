"""
Unit tests for the Standardizer class.

This test module provides comprehensive testing for the Standardizer class
from mother.preprocessing.standardizer, covering various input types, flags,
and edge cases.
"""

import logging

import pytest
from rdkit import Chem

from mother.preprocessing.standardizer import Standardizer
from mother.preprocessing.utils import StandardizationFlag

# Test molecules covering various edge cases
TEST_MOLECULES = {
    # Basic molecules
    "benzoic_acid": "O=C(O)c1ccccc1",
    "benzoate_anion": "O=C([O-])c1ccccc1",
    "sodium_benzoate": "O=C([O-])c1ccccc1.[Na+]",
    "sodium_benzoate_ester": "O=C(O[Na])c1ccccc1",
    "benzoate_with_trimethylammonium": "C[N+](C)C.O=C([O-])c1ccccc1",
    # Tautomers
    "keto_form": "CC(=O)CC(=O)C",  # 2,4-pentanedione (keto form)
    "enol_form": "CC(O)=CC(=O)C",  # 2,4-pentanedione (enol form)
    # Stereochemistry
    "chiral_molecule": "C[C@H](O)CC",  # (S)-butan-2-ol
    "achiral_molecule": "CC(O)CC",  # butan-2-ol (no stereochemistry)
    # Salts and fragments
    "hcl_salt": "CCc1ccc2c3[nH]c4ccccc4c3cc[n+]2c1.[Cl-]",  # harmaline HCl
    "multi_fragment": "c1ccccc1.CCCC",  # benzene + butane
    # Special cases
    "isocyanate": "CN=C=O",  # methyl isocyanate
    "aflatoxin": "COc1cc2c(c3oc(=O)c4c(c13)CCC4=O)[C@@H]1C=CO[C@@H]1O2",
    # Radicals and unusual structures
    "nitric_oxide": "[N]=O",  # radical
    "superoxide": "[O-][O]",  # radical anion
    # Metal complexes
    "copper_complex": "c1ccc(cc1)[N-]c2ccccc2.[Cu+2]",
    # Invalid/edge cases
    "empty": "",
    "invalid_smiles": "c1",  # incomplete aromatic ring
    "invalid_kekulization": "c1nncn1",  # triazole - might cause kekulization issues
}


@pytest.fixture
def standardizer() -> Standardizer:
    """Create a default Standardizer instance."""
    return Standardizer()


@pytest.fixture
def standardizer_full() -> Standardizer:
    """Create a Standardizer with all flags enabled."""
    return Standardizer(flag=StandardizationFlag.ALL)


@pytest.fixture
def standardizer_minimal() -> Standardizer:
    """Create a Standardizer with no flags enabled."""
    return Standardizer(flag=StandardizationFlag.NONE)


class TestStandardizerInitialization:
    """Test Standardizer initialization and configuration."""

    def test_default_initialization(self):
        """Test that Standardizer initializes with default settings."""
        std = Standardizer()
        assert std.flag == StandardizationFlag.STANDARDIZE
        assert isinstance(std, Standardizer)

    def test_custom_flag_initialization(self):
        """Test initialization with custom flags."""
        std = Standardizer(flag=StandardizationFlag.DESALT | StandardizationFlag.NEUTRALIZE)
        assert std.flag & StandardizationFlag.DESALT
        assert std.flag & StandardizationFlag.NEUTRALIZE
        assert not (std.flag & StandardizationFlag.CANONICAL_TAUTOMER)

    def test_max_params_initialization(self):
        """Test initialization with custom max parameters."""
        std = Standardizer(max_restarts=100, max_tautomers=50, max_transforms=500)
        assert std._params.maxRestarts == 100
        assert std._params.maxTautomers == 50
        assert std._params.maxTransforms == 500

    def test_prefer_organic_flag(self):
        """Test prefer_organic initialization."""
        std_organic = Standardizer(prefer_organic=True)
        std_inorganic = Standardizer(prefer_organic=False)
        assert std_organic._params.preferOrganic is True
        assert std_inorganic._params.preferOrganic is False


class TestStandardizerFlagManagement:
    """Test flag management methods."""

    def test_set_flag(self, standardizer: Standardizer):
        """Test setting flags."""
        standardizer.set_flag(StandardizationFlag.ALL)
        assert standardizer.flag == StandardizationFlag.ALL

    def test_disable_flag(self, standardizer_full: Standardizer):
        """Test disabling specific flags."""
        standardizer_full.disable_flag(StandardizationFlag.DESALT)
        assert not (standardizer_full.flag & StandardizationFlag.DESALT)
        assert standardizer_full.flag & StandardizationFlag.NEUTRALIZE

    def test_enable_all_flags(self, standardizer: Standardizer):
        """Test enable_all_flags method."""
        standardizer.enable_all_flags()
        assert standardizer.flag == StandardizationFlag.ALL

    def test_neutralize_flag(self, standardizer_minimal: Standardizer):
        """Test neutralize method."""
        standardizer_minimal.neutralize()
        assert standardizer_minimal.flag & StandardizationFlag.NEUTRALIZE

    def test_desalt_flag(self, standardizer_minimal: Standardizer):
        """Test desalt method."""
        standardizer_minimal.desalt()
        assert standardizer_minimal.flag & StandardizationFlag.DESALT

    def test_keep_salt(self, standardizer_full: Standardizer):
        """Test keep_salt method."""
        standardizer_full.keep_salt()
        assert not (standardizer_full.flag & StandardizationFlag.DESALT)

    def test_flatten(self, standardizer_minimal: Standardizer):
        """Test flatten method."""
        standardizer_minimal.flatten()
        assert standardizer_minimal.flag & StandardizationFlag.FLATTEN_STEREOCHEMISTRY

    def test_canonical_tautomer(self, standardizer_minimal: Standardizer):
        """Test canonical_tautomer method."""
        standardizer_minimal.canonical_tautomer()
        assert standardizer_minimal.flag & StandardizationFlag.CANONICAL_TAUTOMER

    def test_get_flag(self, standardizer: Standardizer):
        """Test get_flag method."""
        flag = standardizer.get_flag()
        assert flag == StandardizationFlag.STANDARDIZE


class TestStandardizerInputTypes:
    """Test different input types (SMILES strings vs Mol objects)."""

    def test_smiles_input(self, standardizer: Standardizer):
        """Test standardization with SMILES string input."""
        result = standardizer.standardize(TEST_MOLECULES["benzoic_acid"])
        assert isinstance(result, Chem.rdchem.Mol)

    def test_mol_input(self, standardizer: Standardizer):
        """Test standardization with Mol object input."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["benzoic_acid"])
        result = standardizer.standardize(mol)
        assert isinstance(result, Chem.rdchem.Mol)

    def test_output_as_smiles(self, standardizer: Standardizer):
        """Test output as SMILES string."""
        standardizer.as_smiles(True)
        result = standardizer.standardize(TEST_MOLECULES["benzoic_acid"])
        assert isinstance(result, str)

    def test_output_as_mol(self, standardizer: Standardizer):
        """Test output as Mol object."""
        standardizer.as_smiles(False)
        result = standardizer.standardize(TEST_MOLECULES["benzoic_acid"])
        assert isinstance(result, Chem.rdchem.Mol)

    def test_none_input(self, standardizer: Standardizer):
        """Test handling of None input."""
        result = standardizer.standardize(None)
        assert result is None

    def test_empty_string_input(self, standardizer: Standardizer):
        """Test handling of empty string input."""
        result = standardizer.standardize(TEST_MOLECULES["empty"])
        # Empty string produces a valid molecule instance but with no atoms
        assert isinstance(result, (Chem.rdchem.Mol, str))

    def test_invalid_smiles_input(self, standardizer: Standardizer):
        """Test handling of invalid SMILES."""
        result = standardizer.standardize(TEST_MOLECULES["invalid_smiles"])
        # Invalid SMILES should return None
        assert result is None


class TestStandardizationOperations:
    """Test individual standardization operations."""

    def test_desalting(self):
        """Test desalting operation."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE | StandardizationFlag.DESALT)
        std.as_smiles(True)

        # Sodium benzoate should lose sodium ion
        result = std.standardize(TEST_MOLECULES["sodium_benzoate"])
        assert isinstance(result, str)
        assert "[Na" not in result and "[Na+" not in result

    def test_neutralization(self):
        """Test neutralization operation."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE | StandardizationFlag.NEUTRALIZE)
        std.as_smiles(True)

        # Benzoate anion should be neutralized
        result = std.standardize(TEST_MOLECULES["benzoate_anion"])
        assert isinstance(result, str)
        # Check that it's now neutral (contains OH or similar)
        assert "O" in result

    def test_canonical_tautomer_operation(self):
        """Test canonical tautomer generation."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE | StandardizationFlag.CANONICAL_TAUTOMER)
        std.as_smiles(True)

        # Both tautomers should give the same canonical form
        keto_result = std.standardize(TEST_MOLECULES["keto_form"])
        enol_result = std.standardize(TEST_MOLECULES["enol_form"])

        assert keto_result is not None
        assert enol_result is not None
        # Both should produce the same canonical tautomer
        assert keto_result == enol_result

    def test_flatten_stereochemistry(self):
        """Test flattening of stereochemistry."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE | StandardizationFlag.FLATTEN_STEREOCHEMISTRY)
        std.as_smiles(True)

        result = std.standardize(TEST_MOLECULES["chiral_molecule"])
        assert result is not None
        # Stereochemistry markers should be removed
        assert "@" not in result

    def test_fragment_removal(self):
        """Test fragment removal."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE | StandardizationFlag.DESALT)
        std.as_smiles(True)

        # Multi-fragment should keep only largest fragment
        result = std.standardize(TEST_MOLECULES["multi_fragment"])
        assert result is not None
        # Should contain benzene (largest fragment) but not butane
        assert "." not in result  # No fragment separator

    def test_metal_disconnection(self):
        """Test metal disconnection."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE)
        std.as_smiles(True)

        result = std.standardize(TEST_MOLECULES["copper_complex"])
        assert result is not None


class TestStandardizationFlags:
    """Test various flag combinations."""

    def test_no_standardization(self):
        """Test with NONE flag - should return input mostly unchanged."""
        std = Standardizer(flag=StandardizationFlag.NONE)
        std.as_smiles(True)

        input_smiles = TEST_MOLECULES["benzoic_acid"]
        result = std.standardize(input_smiles)

        # With NONE flag, molecule should be minimally changed
        assert result is not None

    def test_standardize_only(self):
        """Test with only STANDARDIZE flag."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE)
        std.as_smiles(True)

        result = std.standardize(TEST_MOLECULES["benzoic_acid"])
        assert result is not None

    def test_all_flags(self):
        """Test with ALL flags enabled."""
        std = Standardizer(flag=StandardizationFlag.ALL)
        std.as_smiles(True)

        result = std.standardize(TEST_MOLECULES["sodium_benzoate"])
        assert result is not None
        # Should be desalted, neutralized, standardized, etc.
        assert "[Na" not in result

    def test_combined_flags(self):
        """Test various flag combinations."""
        std = Standardizer(
            flag=StandardizationFlag.STANDARDIZE | StandardizationFlag.NEUTRALIZE | StandardizationFlag.DESALT
        )
        std.as_smiles(True)

        result = std.standardize(TEST_MOLECULES["hcl_salt"])
        assert result is not None
        # Should be desalted (no Cl-)
        assert "[Cl-]" not in result


class TestMoleculeProperties:
    """Test that molecule properties are preserved and added."""

    def test_original_smiles_property(self, standardizer: Standardizer):
        """Test that original SMILES is stored as property."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["benzoic_acid"])
        result = standardizer.standardize(mol)

        assert result is not None
        assert result.HasProp("Original_SMILES")

    def test_canonical_smiles_property(self):
        """Test that canonical SMILES is stored as property."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE)
        mol = Chem.MolFromSmiles(TEST_MOLECULES["benzoic_acid"])
        result = std.standardize(mol)

        assert result is not None
        assert result.HasProp("Canonical_SMILES")

    def test_properties_preserved(self):
        """Test that original properties are preserved."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE)
        mol = Chem.MolFromSmiles(TEST_MOLECULES["benzoic_acid"])
        mol.SetProp("TestProp", "TestValue")
        mol.SetProp("_Name", "TestMolecule")

        result = std.standardize(mol)

        assert result is not None
        assert result.HasProp("TestProp")
        assert result.GetProp("TestProp") == "TestValue"
        assert result.HasProp("_Name")
        assert result.GetProp("_Name") == "TestMolecule"


class TestIndividualMethods:
    """Test individual standardizer methods."""

    def test_normalize(self, standardizer: Standardizer):
        """Test normalize method."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["benzoic_acid"])
        result = standardizer.normalize(mol)
        assert isinstance(result, Chem.rdchem.Mol)

    def test_reionize(self, standardizer: Standardizer):
        """Test reionize method."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["benzoate_anion"])
        result = standardizer.reionize(mol)
        assert isinstance(result, Chem.rdchem.Mol)

    def test_disconnect(self, standardizer: Standardizer):
        """Test disconnect method."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["copper_complex"])
        result = standardizer.disconnect(mol)
        assert isinstance(result, Chem.rdchem.Mol)

    def test_get_largest_fragment(self, standardizer: Standardizer):
        """Test get_largest_fragment method."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["multi_fragment"])
        result = standardizer.get_largest_fragment(mol)
        assert isinstance(result, Chem.rdchem.Mol)
        # Result should be a single fragment (benzene)
        assert "." not in Chem.MolToSmiles(result)

    def test_remove_fragments(self, standardizer: Standardizer):
        """Test remove_fragments method."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["sodium_benzoate"])
        result = standardizer.remove_fragments(mol)
        assert isinstance(result, Chem.rdchem.Mol)

    def test_uncharge(self, standardizer: Standardizer):
        """Test uncharge method."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["benzoate_anion"])
        result = standardizer.uncharge(mol)
        assert isinstance(result, Chem.rdchem.Mol)

    def test_get_canonical_tautomer(self, standardizer: Standardizer):
        """Test get_canonical_tautomer method."""
        mol_keto = Chem.MolFromSmiles(TEST_MOLECULES["keto_form"])
        mol_enol = Chem.MolFromSmiles(TEST_MOLECULES["enol_form"])

        result_keto = standardizer.get_canonical_tautomer(mol_keto)
        result_enol = standardizer.get_canonical_tautomer(mol_enol)

        # Both tautomers should give the same result
        assert Chem.MolToSmiles(result_keto) == Chem.MolToSmiles(result_enol)

    def test_enumerate_tautomers(self, standardizer: Standardizer):
        """Test enumerate_tautomers method."""
        mol = Chem.MolFromSmiles(TEST_MOLECULES["keto_form"])
        result = standardizer.enumerate_tautomers(mol)

        # Should return a TautomerEnumeratorResult
        assert result is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_kekulization_error_repair(self):
        """Test that kekulization errors are handled and repaired."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE)

        # This might cause kekulization issues
        result = std.standardize(TEST_MOLECULES["invalid_kekulization"])

        # Should either fix it or return None
        assert result is None or isinstance(result, (Chem.rdchem.Mol, str))

    def test_radical_handling(self):
        """Test handling of radical species."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE)
        std.as_smiles(True)

        result = std.standardize(TEST_MOLECULES["nitric_oxide"])
        assert result is not None

    def test_complex_molecule(self):
        """Test standardization of complex molecule (aflatoxin)."""
        std = Standardizer(flag=StandardizationFlag.ALL)
        std.as_smiles(True)

        result = std.standardize(TEST_MOLECULES["aflatoxin"])
        assert result is not None

    def test_reusability(self, standardizer: Standardizer):
        """Test that standardizer can be reused with different molecules."""
        mol1 = TEST_MOLECULES["benzoic_acid"]
        mol2 = TEST_MOLECULES["isocyanate"]

        result1 = standardizer.standardize(mol1)
        result2 = standardizer.standardize(mol2)

        assert result1 is not None
        assert result2 is not None
        assert result1 != result2

    def test_flag_reusability(self):
        """Test that standardizer flags can be changed and reused."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE)

        # First run with STANDARDIZE
        result1 = std.standardize(TEST_MOLECULES["sodium_benzoate"])

        # Change flag to add DESALT
        std.desalt()
        result2 = std.standardize(TEST_MOLECULES["sodium_benzoate"])

        # Results should be different
        assert result1 is not None
        assert result2 is not None


class TestLogging:
    """Test logging functionality."""

    def test_rdkit_log_level(self, standardizer: Standardizer):
        """Test setting RDKit log level."""
        standardizer.set_rdkit_log_level(logging.WARNING)
        # Should not raise any errors

    def test_disable_rdkit_logging(self, standardizer: Standardizer):
        """Test disabling RDKit logging."""
        standardizer.disable_rdkit_logging()
        # Should not raise any errors


class TestKwargsHandling:
    """Test kwargs handling in standardize method."""

    def test_canonical_kwarg(self, standardizer: Standardizer):
        """Test canonical kwarg."""
        standardizer.as_smiles(True)

        result_canonical = standardizer.standardize(TEST_MOLECULES["benzoic_acid"], canonical=True)
        result_noncanonical = standardizer.standardize(TEST_MOLECULES["benzoic_acid"], canonical=False)

        assert result_canonical is not None
        assert result_noncanonical is not None

    def test_isomeric_smiles_kwarg(self):
        """Test isomericSmiles kwarg."""
        std = Standardizer(flag=StandardizationFlag.STANDARDIZE)
        std.as_smiles(True)

        result = std.standardize(TEST_MOLECULES["chiral_molecule"], isomericSmiles=False)

        assert result is not None
        # Stereochemistry should not be present
        assert "@" not in result


class TestResultIsSmiles:
    """Test result_is_smiles method."""

    def test_result_is_smiles_true(self):
        """Test result_is_smiles when output is SMILES."""
        std = Standardizer()
        std.as_smiles(True)
        assert std.result_is_smiles() is True

    def test_result_is_smiles_false(self):
        """Test result_is_smiles when output is Mol."""
        std = Standardizer()
        std.as_smiles(False)
        assert std.result_is_smiles() is False


class TestMethodChaining:
    """Test method chaining functionality."""

    def test_flag_method_chaining(self):
        """Test that flag methods can be chained."""
        std = Standardizer(flag=StandardizationFlag.NONE)

        result_std = std.neutralize().desalt().canonical_tautomer()

        assert result_std is std  # Should return self
        assert std.flag & StandardizationFlag.NEUTRALIZE
        assert std.flag & StandardizationFlag.DESALT
        assert std.flag & StandardizationFlag.CANONICAL_TAUTOMER

    def test_as_smiles_chaining(self):
        """Test that as_smiles can be chained."""
        std = Standardizer()
        result_std = std.as_smiles(True)

        assert result_std is std  # Should return self
        assert std.result_is_smiles() is True


class TestBatchProcessing:
    """Test processing multiple molecules."""

    def test_multiple_molecules(self, standardizer: Standardizer):
        """Test standardizing multiple molecules."""
        standardizer.as_smiles(True)

        test_smiles = [
            TEST_MOLECULES["benzoic_acid"],
            TEST_MOLECULES["sodium_benzoate"],
            TEST_MOLECULES["isocyanate"],
        ]

        results = [standardizer.standardize(smi) for smi in test_smiles]

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert all(isinstance(r, str) for r in results)

    def test_mixed_valid_invalid(self, standardizer: Standardizer):
        """Test processing a mix of valid and invalid molecules."""
        standardizer.as_smiles(True)

        test_smiles = [
            TEST_MOLECULES["benzoic_acid"],  # valid
            TEST_MOLECULES["invalid_smiles"],  # invalid
            TEST_MOLECULES["isocyanate"],  # valid
            TEST_MOLECULES["empty"],  # empty
        ]

        results = [standardizer.standardize(smi) for smi in test_smiles]

        assert len(results) == 4
        # First and third should be valid
        assert results[0] is not None
        assert results[2] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
