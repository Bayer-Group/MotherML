#!/usr/bin/env bash
#
# This script creates a virtual environment and installs a distribution of
# this project into the environment and then executes the unit tests. This
# provides a crude check that the distribution is installable and that the
# package is minimally functional.
#

# Exit immediately if a command exits with a non-zero status
set -e
# Exit if any command in a pipeline fails (not just the last one)
set -o pipefail
# Treat unset variables as an error
set -u

if [ -z "$1" ]; then
  echo "usage: $0 mother_ml-YY.MM.MICRO-py3-none-any.whl"
  exit
fi

RELEASE_ARCHIVE="$1"

echo "Release archive: $RELEASE_ARCHIVE"

echo "Removing any old artefacts"
rm -rf test_venv

echo "Creating test virtual environment"
python -m venv test_venv

echo "Entering test virtual environment"
source test_venv/bin/activate

echo "Upgrading pip"
pip install pip --upgrade && pip install pytest

echo "Installing $RELEASE_ARCHIVE"
pip install "$RELEASE_ARCHIVE"

echo "Checking installed extras..."
python -c "
import sys
extras_to_check = {
    'tabpfn': 'tabpfn',
    'torch': 'torch',
    'rna': 'anndata',
    'report': 'seaborn',
    'clustering': 'hdbscan',
    'uncertainty': 'quantile_forest',
}

print('\\nInstalled extras:')
for extra_name, package_name in extras_to_check.items():
    try:
        __import__(package_name)
        print(f'  ✓ {extra_name}')
    except ImportError:
        print(f'  ✗ {extra_name}')
print()
"

echo "Running tests"
cd ../test || exit
pytest unit/test_ml.py

echo "Exiting test virtual environment"
deactivate
