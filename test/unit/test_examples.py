"""
To avoid bit-rot in the examples they are tested as part of the unit tests
suite.
"""

import json
import os
import typing
from pathlib import Path

import pandas as pd
import pytest

path_to_test_dir = Path(__file__).parent
# Use the current virtual environment when executing the example scripts.
VENV_DIR = os.environ.get("VIRTUAL_ENV")

REPO_DIR: Path = Path(__file__).parent.parent.parent

# Collect all Python script names in the 'examples' folder
example_scripts: typing.List[str] = [script.name for script in REPO_DIR.joinpath("examples").glob("high*.py")]


@pytest.mark.parametrize("working_dir", [".", "examples/notebooks/01_basics"])
def test_clustering_notebook_loads_freesolv(monkeypatch, working_dir) -> None:
    notebook_path = REPO_DIR / "examples/notebooks/01_basics/05_compound_clustering.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "data = pd.read_csv(" in "".join(cell["source"])
    )
    # monkeypatch.chdir(REPO_DIR / working_dir)
    namespace = {"Path": Path, "pd": pd}
    exec(compile(source, filename=str(notebook_path), mode="exec"), globals=namespace)
    expected = pd.read_csv(REPO_DIR / "examples/notebooks/freesolv_train.csv")
    expected = expected[["smiles", "expt"]].rename(columns={"smiles": "SMILES"})
    pd.testing.assert_frame_equal(namespace["data"], expected)


def test_multitask_notebook_loads_public_adme(monkeypatch) -> None:
    notebook_path = REPO_DIR / "examples/notebooks/02_regression/04_multitask_random_forest.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "df = pd.read_csv(" in "".join(cell["source"])
    )
    source_data = pd.DataFrame(
        {
            "SMILES": ["CCO"],
            "LOG HLM_CLint (mL/min/kg)": [0.5],
            "LOG RLM_CLint (mL/min/kg)": [0.7],
        }
    )

    def read_csv(url):
        assert url.startswith("https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/")
        assert url.endswith("/ADME_public_set_3521.csv")
        return source_data.copy()

    monkeypatch.setattr(pd, "read_csv", read_csv)
    namespace = {"pd": pd}
    exec(compile(source, str(notebook_path), "exec"), namespace)
    expected = pd.DataFrame({"SMILES": ["CCO"], "LOG_HLM_CLint": [0.5], "LOG_RLM_CLint": [0.7]})
    pd.testing.assert_frame_equal(namespace["df"], expected)
