import sys
from pathlib import Path

from tabpfn.model.loading import _user_cache_dir, download_all_models

# Determine cache directory
cache_dir: Path = _user_cache_dir(platform=sys.platform, appname="tabpfn")
cache_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading all models to {cache_dir}")
download_all_models(cache_dir)
print(f"All models downloaded to {cache_dir}")
