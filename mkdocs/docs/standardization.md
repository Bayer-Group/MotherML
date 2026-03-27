# Chemical Molecule Standardisation

Standardisation ensures that molecular representations are consistent and comparable by transforming chemical structures into a canonical, uniform form.

This helps to:

- Remove ambiguity in molecular representations.
- Ensure that the same molecule is always represented in the same way.
- Facilitate accurate feature extraction, similarity searches, and model training.

---

## Why is Standardisation Important?

- **Consistency**: Different sources may represent the same molecule differently (e.g., different tautomers, salts, or stereochemistry).
- **Data Quality**: Standardisation removes unwanted fragments, normalises charges, and corrects valence issues.
- **Reproducibility**: Ensures that results are not affected by inconsistent molecular input.

---

## Standardisation Steps Supported by Mother

The **Mother** framework provides tools to automate standardisation largely following [RDKit MolStandardize](https://github.com/rdkit/rdkit/blob/master/Docs/Notebooks/MolStandardize.ipynb).

### Available Standardization Flags

0. **Base Standardisation `STANDARDIZE`**
    - Baseline initial standardisation steps at the beginning and the end of the entire standardisation process
        - **Beginning**: Metal disconnection and molecule sanitisation
        - **End**: Molecule normalisation and reionisation
    - This flag is typically always included as it provides essential cleanup steps

1. **Salt Stripping (Desalting) `DESALT`**
    - Removes counterions and keeps only the main organic molecule
    - Includes fragment removal and largest fragment selection
    - Example: Aspirin sodium salt → Aspirin
    - Example: Hydrochloride salt → Free base

2. **Charge Neutralisation `NEUTRALIZE`**
    - Adjusts formal charges to standard states
    - Attempts to generate a neutral form of the molecule
    - Example: Benzoate anion → Benzoic acid

3. **Canonical Tautomer `CANONICAL_TAUTOMER`**
    - Converts tautomers to a preferred canonical form
    - Ensures the same tautomer is always generated for a given molecule
    - Example: Keto-enol tautomers → Canonical form

4. **Flatten Stereochemistry `FLATTEN_STEREOCHEMISTRY`**
    - Removes stereochemical information from the molecule
    - Useful when stereochemistry is not relevant to the analysis
    - Example: (S)-butan-2-ol → butan-2-ol (no stereochemistry)

### Predefined Flag Combinations

- **`NONE`**: No standardisation (molecule unchanged)
- **`KEEP_SALT`**: `STANDARDIZE | NEUTRALIZE` (standardise and neutralise but keep salts)
- **`ALL`**: `STANDARDIZE | NEUTRALIZE | DESALT | FLATTEN_STEREOCHEMISTRY | CANONICAL_TAUTOMER` (all standardisation steps)

---

## Usage Examples

### Using StandardizerTransformer (Scikit-learn compatible)

The `StandardizerTransformer` is designed for use in scikit-learn pipelines:

```python
from mother.preprocessing import StandardizerTransformer

# Example: Standardise and desalt SMILES strings
standardizer = StandardizerTransformer(flags=["STANDARDIZE", "DESALT", "NEUTRALIZE"])
standardised_smiles = standardizer.fit_transform(smiles_data)
```

### Using the Standardizer class directly

For more control, you can use the `Standardizer` class directly:

```python
from mother.preprocessing.standardizer import Standardizer
from mother.preprocessing.utils import StandardizationFlag

# Create a standardizer with specific flags
std = Standardizer(
    flag=StandardizationFlag.STANDARDIZE | StandardizationFlag.DESALT | StandardizationFlag.NEUTRALIZE,
    as_smiles=True  # Output as SMILES strings instead of Mol objects
)

# Standardize a single molecule
result = std.standardize("O=C([O-])c1ccccc1.[Na+]")
print(result)  # Output: "O=C(O)c1ccccc1" (benzoic acid)
```

### Method Chaining

The `Standardizer` class supports method chaining for convenient configuration:

```python
from mother.preprocessing.standardizer import Standardizer

# Build standardizer using method chaining
std = Standardizer().neutralize().desalt().canonical_tautomer().as_smiles(True)

# Standardize molecules
result = std.standardize("CC(=O)CC(=O)C")  # Keto form
print(result)  # Canonical tautomer
```

### Processing Multiple Molecules

```python
from mother.preprocessing.standardizer import Standardizer
from mother.preprocessing.utils import StandardizationFlag

std = Standardizer(flag=StandardizationFlag.ALL, as_smiles=True)

smiles_list = [
    "O=C(O)c1ccccc1",           # Benzoic acid
    "O=C([O-])c1ccccc1.[Na+]",  # Sodium benzoate
    "CCc1ccc2c3[nH]c4ccccc4c3cc[n+]2c1.[Cl-]",  # HCl salt
]

standardized = [std.standardize(smi) for smi in smiles_list]
```

### Advanced Configuration

```python
from mother.preprocessing.standardizer import Standardizer
from mother.preprocessing.utils import StandardizationFlag

# Create standardizer with custom parameters
std = Standardizer(
    flag=StandardizationFlag.STANDARDIZE | StandardizationFlag.NEUTRALIZE,
    max_restarts=200,          # Maximum normalization restarts
    max_tautomers=100,         # Maximum tautomers to consider
    max_transforms=1000,       # Maximum transformations
    prefer_organic=True,       # Prefer organic fragments when choosing largest
    as_smiles=False            # Return Mol objects
)

# Get result as Mol object
mol = std.standardize("O=C(O)c1ccccc1")

# Access molecule properties
if mol.HasProp("Original_SMILES"):
    print(f"Original: {mol.GetProp('Original_SMILES')}")
if mol.HasProp("Canonical_SMILES"):
    print(f"Canonical: {mol.GetProp('Canonical_SMILES')}")
```

### Dynamic Flag Management

```python
from mother.preprocessing.standardizer import Standardizer
from mother.preprocessing.utils import StandardizationFlag

# Start with basic standardization
std = Standardizer(flag=StandardizationFlag.STANDARDIZE)

# Process first batch
batch1 = [std.standardize(smi) for smi in smiles_batch1]

# Add desalting for second batch
std.desalt()
batch2 = [std.standardize(smi) for smi in smiles_batch2]

# Use all flags for third batch
std.enable_all_flags()
batch3 = [std.standardize(smi) for smi in smiles_batch3]

# Disable a specific flag
std.disable_flag(StandardizationFlag.FLATTEN_STEREOCHEMISTRY)
batch4 = [std.standardize(smi) for smi in smiles_batch4]
```

---

## Standardization Pipeline

The standardization process follows this sequence when flags are enabled:

1. **Initial validation and sanitization** (if `STANDARDIZE` flag is set)
   - Validate input SMILES
   - Sanitize molecule
   - Remove hydrogens
   - Disconnect metals

2. **Desalting** (if `DESALT` flag is set)
   - Remove fragments
   - Strip salts using salt list
   - Select largest fragment

3. **Neutralization** (if `NEUTRALIZE` flag is set)
   - Uncharge the molecule to generate neutral form

4. **Tautomerization** (if `CANONICAL_TAUTOMER` flag is set)
   - Generate canonical tautomer

5. **Stereochemistry handling** (if `FLATTEN_STEREOCHEMISTRY` flag is set)
   - Remove stereochemistry information

6. **Final normalization** (if `STANDARDIZE` flag is set)
   - Apply normalization rules
   - Reionize molecule
   - Assign stereochemistry
   - Generate canonical SMILES

### Molecule Properties

After standardization, molecules contain additional properties:

- **`Original_SMILES`**: The original SMILES string before standardization
- **`Canonical_SMILES`**: The canonical SMILES after standardization
- **`_Name`**: Molecule name (if provided)

---

## API Reference

### Standardizer Class

```python
class Standardizer:
    def __init__(
        self,
        flag: StandardizationFlag = StandardizationFlag.STANDARDIZE,
        max_restarts: int = 200,
        max_tautomers: int = 100,
        max_transforms: int = 1000,
        prefer_organic: bool = True,
        as_smiles: bool = False,
        **kwargs
    )
```

**Parameters:**

- `flag`: Standardization flags to apply (default: `STANDARDIZE`)
- `max_restarts`: Maximum normalization restarts (default: 200)
- `max_tautomers`: Maximum tautomers to enumerate (default: 100)
- `max_transforms`: Maximum transformations to apply (default: 1000)
- `prefer_organic`: Prefer organic fragments when choosing largest (default: True)
- `as_smiles`: Return SMILES strings instead of Mol objects (default: False)

**Key Methods:**

- `standardize(mol)`: Standardize a molecule (accepts SMILES string or Mol object)
- `as_smiles(bool)`: Set output format (returns self for chaining)
- `set_flag(flag)`: Set standardization flags (returns self)
- `neutralize()`: Enable neutralization (returns self)
- `desalt()`: Enable desalting (returns self)
- `keep_salt()`: Disable desalting (returns self)
- `flatten()`: Enable stereochemistry flattening (returns self)
- `canonical_tautomer()`: Enable canonical tautomer (returns self)
- `enable_all_flags()`: Enable all flags (returns self)
- `disable_flag(flag)`: Disable specific flag (returns self)
- `get_flag()`: Get current flags
- `result_is_smiles()`: Check if output format is SMILES

**Individual Operations:**

- `normalize(mol)`: Apply normalization rules
- `reionize(mol)`: Reionize molecule
- `disconnect(mol)`: Disconnect metals
- `get_largest_fragment(mol)`: Get largest fragment
- `remove_fragments(mol)`: Remove small fragments
- `uncharge(mol)`: Neutralize charges
- `get_canonical_tautomer(mol)`: Get canonical tautomer
- `enumerate_tautomers(mol)`: Enumerate all tautomers

### StandardizerTransformer Class

Scikit-learn compatible transformer for pipeline integration.

```python
class StandardizerTransformer:
    def __init__(
        self,
        flags: List[str],
        smiles_col: Optional[str] = None,
        parallel: bool = False,
        error: str = "ignore"
    )
```

**Parameters:**

- `flags`: List of flag names as strings (e.g., `["STANDARDIZE", "DESALT"]`)
- `smiles_col`: Column name for SMILES (for DataFrames)
- `parallel`: Enable parallel processing
- `error`: Error handling strategy ("ignore" or "raise")

---

## Best Practices

1. **Always standardise molecules** before feature generation or model training
2. **Use `STANDARDIZE` flag** as the base - it provides essential cleanup
3. **Combine flags appropriately**:
   - For most applications: `STANDARDIZE | NEUTRALIZE | DESALT`
   - For tautomer-invariant models: Add `CANONICAL_TAUTOMER`
   - For 2D descriptors: Consider adding `FLATTEN_STEREOCHEMISTRY`
4. **Document the standardisation steps** used for reproducibility
5. **Test standardization** on your dataset to understand its impact
6. **Handle errors appropriately** - some molecules may fail standardization
7. **Preserve original data** - standardization is lossy, keep original SMILES

### Common Pitfalls

- **Over-standardization**: Removing too much information (e.g., removing stereochemistry when it's important)
- **Under-standardization**: Not removing enough variation (e.g., keeping salts when they're not relevant)
- **Inconsistent application**: Applying different standardization to training vs. test data
- **Lost information**: Not preserving original SMILES for reference

---

## Performance Considerations

- Standardization can be computationally intensive for large datasets
- The `as_smiles=True` option can be faster than returning Mol objects
- Metal disconnection and tautomer enumeration are the most expensive operations

---

## Further Reading

- [RDKit Standardizer Documentation](https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.html)

<div align="right">
  <a href="../feature_generation">Next: Feature Generation &rarr;</a>
</div>
