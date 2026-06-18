from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import download_model, get_cache_dir

# Determine cache directory and ensure it exists
cache_dir = get_cache_dir()
cache_dir.mkdir(parents=True, exist_ok=True)

# Resolve the exact file paths TabPFN will use at runtime for V2 weights.
# V2 is the commercially-licensed version; TabPFN 3 (default in 8.x) is non-commercial.
# download_model() expects the full destination file path, not a directory.
_MODELS = [
    (TabPFNRegressor.create_default_for_version(ModelVersion.V2).model_path, "regressor"),
    (TabPFNClassifier.create_default_for_version(ModelVersion.V2).model_path, "classifier"),
]

print(f"Downloading TabPFN V2 models to {cache_dir}")
for model_path, which in _MODELS:
    print(f"  {which}: {model_path}")
    result = download_model(model_path, version=ModelVersion.V2, which=which)  # type: ignore[arg-type]
    if result != "ok":
        raise RuntimeError(f"Failed to download TabPFN V2 {which}: {result}")
print("TabPFN V2 models downloaded successfully")
