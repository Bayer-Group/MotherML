import pickle
from datetime import datetime

import numpy as np
import pytest
import sklearn.base as skl_base
from rdkit import Chem

import mother.cv as cv_module
from mother.cv import core, cv_methods


def test_np_to_bv():
    sample_array = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    bit_vector = cv_methods.np_to_bv(sample_array)
    assert len(bit_vector) == len(sample_array), "Bit vector length mismatch."
    expected_bits = [1, 3, 4, 7]
    for index in expected_bits:
        assert bit_vector[index], f"Bit at index {index} should be set."


@pytest.fixture
def smiles_test_cases():
    return {
        "COc1ccc(C(=O)N(C)C)cc1": {
            "Murcko": "c1ccccc1",
            "GenericMurcko": "CCC1CCC(C(C)C(C)C)CC1",
            "NoScaffold": "COc1ccc(C(=O)N(C)C)cc1",
        },
        "CCc1cnccn1": {
            "Murcko": "c1cnccn1",
            "GenericMurcko": "CCC1CCCCC1",
            "NoScaffold": "CCc1cnccn1",
        },
        "CCCCCCCO": {
            "Murcko": "CCCCCCCO",
            "GenericMurcko": "CCCCCCCC",
            "NoScaffold": "CCCCCCCO",
        },
        "CNC(=O)c1ccccc1": {
            "Murcko": "c1ccccc1",
            "GenericMurcko": "CCC(C)C1CCCCC1",
            "NoScaffold": "CNC(=O)c1ccccc1",
        },
    }


@pytest.mark.parametrize("scaffold_option", ["Murcko", "GenericMurcko", "NoScaffold"])
def test_murcko_scaffold_reduction(
    smiles_test_cases,
    scaffold_option,
):
    for smiles, expected in smiles_test_cases.items():
        mol = Chem.MolFromSmiles(smiles)
        result = cv_methods.murcko_scaffold_reduction([mol], scaffold=scaffold_option)[0]
        expected_smiles = expected[scaffold_option]
        expected_mol = Chem.MolFromSmiles(expected_smiles)

        assert Chem.MolToSmiles(result) == Chem.MolToSmiles(expected_mol)


def test_empty_input():
    result = cv_methods.murcko_scaffold_reduction([], scaffold="Murcko")
    assert result == []


class TestKMedoidsClustering:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method to initialize sample fingerprints for testing."""
        self.fingerprints = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0]])

    def test_default_parameters(self):
        """Test k-medoids clustering with default parameters."""
        clusters = cv_methods.kmedoids_clustering(self.fingerprints)
        assert len(clusters) == 2  # Expecting 2 clusters (default)
        assert all(isinstance(k, int) for k in clusters.keys())  # Check that keys are integers

    def test_custom_cluster_number(self):
        """Test k-medoids clustering with a custom number of clusters."""
        clusters = cv_methods.kmedoids_clustering(self.fingerprints, clusters_number=3)
        assert len(clusters) == 3  # Expecting 3 clusters
        assert all(isinstance(k, int) for k in clusters.keys())

    def test_random_state_reproducibility(self):
        """Test that using a random state produces the same clusters."""
        clusters1 = cv_methods.kmedoids_clustering(self.fingerprints, random_state=42)
        clusters2 = cv_methods.kmedoids_clustering(self.fingerprints, random_state=42)
        assert clusters1 == clusters2  # Clusters should be the same


@pytest.fixture
def test_cases():
    return [
        Chem.MolFromSmiles(mol)
        for mol in [
            "c1ccccc1",
            "c1ccncc1",
            "c1ccoc1",
            "c1ccsc1",
            "CCCCCC",
            "CC(C)CC(C)C",
            "CC(C)C1CCC(C(C)C)CC1",
            "Nc1cccc(C(=O)Nc2ccnc(NC(=O)CO)c2)c1F",
            "Cc1c(N)cccc1C(=O)Nc1ccnc(NC(=O)CC(C)C)c1",
            "COc1cccc(C(=O)Nc2ccnc(Nc3cc(C)cc(C)c3)c2)c1",
            "Cc1c(N)cccc1C(=O)Nc1ccnc(NC(=O)NCc2ccccc2)c1",
            "Cc1cc(Nc2cc(NC(=O)c3ccc(O)cc3Cl)ccn2)nc(N(C)C)n1",
            "C1CCCCC1C2=CC=CC=C2",
            "O=C(Nc1ccnc(NC(=O)C2CCC2)c1)c1c(Cl)cccc1Cl",
            "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",
            "CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCC",
            "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
            "Brc1cccc(Nc2ncnc3cc4[nH]cnc4cc23)c1",
            "CNc1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
        ]
    ]


@pytest.mark.parametrize("scaffold_option", ["NoScaffold", "Murcko", "GenericMurcko"])
@pytest.mark.parametrize("iteration_method", ["random", "build"])
@pytest.mark.parametrize("max_iter", [10, 100])
@pytest.mark.parametrize("clusters_number", [2, 5])
class TestKMedoids:
    def test_clustering_on_mols(self, test_cases, scaffold_option, iteration_method, max_iter, clusters_number):
        groups_engine = core.KMedoidsGroupingFromMols(
            clusters_number=clusters_number,
            scaffold=scaffold_option,
            iteration_method=iteration_method,
            max_iter=max_iter,
            random_state=42,
        )

        groups = groups_engine.set_output(transform="pandas").fit_transform(test_cases)

        assert groups is not None, "Groups attribute is not initialized"
        assert "kmedoids-group" in groups.columns, "Failed to generate clustering data"
        assert groups["kmedoids-group"].nunique() >= 1, "Unexpected number of clusters"
        assert not groups.empty, "Clustering result is empty"

        scaffolds = groups_engine.get_murcko_scaffolds_out()

        assert len(scaffolds) == len(test_cases)
        assert scaffolds is not None, "Scaffolds attribute is not initialized"


@pytest.mark.parametrize("scaffold_option", ["NoScaffold", "Murcko", "GenericMurcko"])
class TestHdbscan:
    def test_clustering_on_mols(self, test_cases, scaffold_option):
        groups_engine = core.HdbscanGroupingFromMols(scaffold=scaffold_option)
        groups = groups_engine.set_output(transform="pandas").fit_transform(test_cases)
        if groups["hdbscan-group"].nunique() <= 1:
            pytest.skip(f"HDBSCAN failed to find multiple clusters for scaffold option: {scaffold_option}")

        assert groups is not None, "Groups attribute is not initialized"
        assert "hdbscan-group" in groups.columns, "Failed to generate clustering data"
        assert groups["hdbscan-group"].isnull().sum() == 0, "Failed to assign compounds to the clusters"
        assert groups["hdbscan-group"].nunique() >= 1, "Unexpected number of clusters"
        assert not groups.empty, "Clustering result is empty"

        scaffolds = groups_engine.get_murcko_scaffolds_out()
        assert scaffolds is not None, "Scaffolds attribute is not initialized"
        assert len(scaffolds) == len(test_cases)


class TestDefaultGrouping:
    @pytest.fixture
    def grouping(self):
        """Create a DefaultGrouping instance for testing"""
        return core.DefaultGrouping()

    @pytest.fixture
    def valid_input(self):
        """Create valid numeric input data"""
        return np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

    def test_initialization(self, grouping: cv_module.DefaultGrouping):
        """Test the initialization of DefaultGrouping"""
        assert grouping.name == "groups"

    def test_fit_returns_self(self, grouping: cv_module.DefaultGrouping, valid_input):
        """Test that fit returns self and sets is_fitted"""
        result = grouping.fit(valid_input)
        assert result is grouping
        assert hasattr(result, "is_fitted")
        assert result.is_fitted is True

    def test_get_output_dimension(self, grouping: cv_module.DefaultGrouping):
        """Test get_output_dimension returns 1"""
        assert grouping.get_output_dimension() == 1

    def test_get_feature_names_out(self, grouping: cv_module.DefaultGrouping):
        """Test get_feature_names_out returns correct name"""
        names = grouping.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) == 1
        assert names[0] == "groups"

    def test_fit_with_valid_input(self, grouping: cv_module.DefaultGrouping, valid_input):
        """Test fit method with valid input"""
        result = grouping.fit(valid_input)
        assert result.is_fitted is True  # Check if the instance is fitted

    def test_transform_with_valid_input(self, grouping: cv_module.DefaultGrouping, valid_input):
        """Test transform with valid input after fitting"""
        grouping.fit(valid_input)  # Fit the model first
        transformed = grouping.transform(valid_input)
        assert np.array_equal(transformed, valid_input)  # Check if the transformation is as expected

    def test_transform_with_non_numeric_input(self, grouping: cv_module.DefaultGrouping):
        """Test transform raises ValueError with non-numeric input"""
        grouping.fit(np.array([[1.0], [2.0], [3.0]]))  # Fit the model first
        non_numeric_input = np.array([["a"], ["b"], ["c"]])  # Non-numeric input
        with pytest.raises(ValueError, match="Input contains non-numeric data"):
            grouping.transform(non_numeric_input)

    def test_transform_with_na_value(self, grouping: cv_module.DefaultGrouping):
        """Test transform raises ValueError with NA values"""
        grouping.fit(np.array([[1.0], [2.0], [3.0]]))  # Fit the model first
        na_input = np.array([[1.0], [np.nan], [3.0]])  # Input with NA
        with pytest.raises(ValueError, match="'NA' value found in provided list"):
            grouping.transform(na_input)


class TestTanimoto:
    def test_tanimoto_on_mols(self, test_cases, grouping_params: cv_module.TanimotoSimilarityParams) -> None:
        tanimoto_engine = core.TanimotoGroupingFromMols(similarity_threshold=grouping_params.similarity_threshold)
        groups = tanimoto_engine.set_output(transform="pandas").fit_transform(test_cases)

        if groups["tanimoto-group"].nunique() <= 1:
            pytest.skip("Tanimoto failed to find clusters")

        assert groups is not None, "Groups attribute is not initialized"
        assert "tanimoto-group" in groups.columns, "Failed to generate clustering data"
        assert groups["tanimoto-group"].isnull().sum() == 0, "Failed to assign compounds to the clusters"
        assert not groups.empty, "Clustering result is empty"


class TestTimeSeriesGrouping:
    @pytest.fixture
    def grouping(self):
        """Create a TimeSeriesGrouping instance for testing"""
        return core.TimeSeriesGrouping(datetime_fmt="%Y-%m-%d")

    def test_initialization(self, grouping: cv_module.TimeSeriesGrouping):
        """Test the initialization of TimeSeriesGrouping"""
        assert grouping.datetime_fmt == "%Y-%m-%d"

    def test_fit_valid_datetime(self, grouping: cv_module.TimeSeriesGrouping):
        """Test fitting with valid datetime objects"""
        valid_dates = [datetime(2022, 1, 1), datetime(2022, 1, 2)]
        result = grouping.fit(valid_dates)
        assert result.is_fitted is True

    def test_fit_valid_string_datetime(self, grouping: cv_module.TimeSeriesGrouping):
        """Test fitting with valid datetime strings"""
        valid_dates = ["2022-01-01", "2022-01-02"]
        result = grouping.fit(valid_dates)
        assert result.is_fitted is True

    def test_fit_invalid_string_datetime(self, grouping: cv_module.TimeSeriesGrouping):
        """Test fitting with an invalid datetime string"""
        invalid_dates = ["2022-01-01", "invalid_date"]
        with pytest.raises(ValueError, match="Element 'invalid_date' does not match the datetime format '%Y-%m-%d'"):
            grouping.fit(invalid_dates)

    def test_transform_sorted_dates(self, grouping: cv_module.TimeSeriesGrouping):
        """Test transform with sorted datetime objects"""
        valid_dates = [datetime(2022, 1, 1), datetime(2022, 1, 2)]
        grouping.fit(valid_dates)
        transformed = grouping.transform(valid_dates)
        assert np.array_equal(transformed, np.array(valid_dates, dtype="datetime64"))

    def test_transform_sorted_string_dates(self, grouping: cv_module.TimeSeriesGrouping):
        """Test transform with sorted datetime strings"""
        valid_dates = ["2022-01-01", "2022-01-02"]
        grouping.fit(valid_dates)
        transformed = grouping.transform(valid_dates)
        assert np.array_equal(transformed, np.array([datetime(2022, 1, 1), datetime(2022, 1, 2)], dtype="datetime64"))

    def test_transform_unsorted_dates(self, grouping: cv_module.TimeSeriesGrouping):
        """Test transform with unsorted datetime objects"""
        unsorted_dates = [datetime(2022, 1, 2), datetime(2022, 1, 1)]
        grouping.fit(unsorted_dates)
        with pytest.raises(ValueError, match="Provided data are not sorted and can not be used for Time Series Split."):
            grouping.transform(unsorted_dates)

    def test_transform_unsorted_string_dates(self, grouping: cv_module.TimeSeriesGrouping):
        """Test transform with unsorted datetime strings"""
        unsorted_dates = ["2022-01-02", "2022-01-01"]
        grouping.fit(unsorted_dates)
        with pytest.raises(ValueError, match="Provided data are not sorted and can not be used for Time Series Split."):
            grouping.transform(unsorted_dates)


def test_tanimoto_grouping_pickle_clone(test_cases):
    """Test that TanimotoGroupingFromMols can be pickled and unpickled with all parameters preserved"""

    original = cv_module.TanimotoGroupingFromMols(
        similarity_threshold=0.75,
        radius=3,
        fp_size=2048,
        include_chirality=False,
    )

    ## check if pickle/unpickle works

    pickled = pickle.dumps(original)
    unpickled = pickle.loads(pickled)

    assert unpickled.similarity_threshold == original.similarity_threshold
    assert unpickled.radius == original.radius
    assert unpickled.fp_size == original.fp_size
    assert unpickled.include_chirality == original.include_chirality
    assert unpickled.name == original.name

    original.fit(test_cases)
    unpickled.fit(test_cases)

    result_original = original.transform(test_cases)
    result_unpickled = unpickled.transform(test_cases)

    # Check that both instances produce the same clustering results
    assert (result_original == result_unpickled).all()
    cloned_model = skl_base.clone(original)

    assert original.similarity_threshold == cloned_model.similarity_threshold
    assert original.radius == cloned_model.radius
    assert original.fp_size == cloned_model.fp_size


def test_hdbscan_grouping_pickle_clone(test_cases):
    """Test that HdbscanGroupingFromMols can be pickled and unpickled with all parameters preserved"""

    hdbscan_engine = core.HdbscanGroupingFromMols(
        radius=3,
        fp_size=2048,
        include_chirality=False,
        min_cluster_size=3,
    )

    ## check if pickle/unpickle works

    pickled = pickle.dumps(hdbscan_engine)
    unpickled_engine = pickle.loads(pickled)

    assert hdbscan_engine.fp_size == unpickled_engine.fp_size
    assert hdbscan_engine.radius == unpickled_engine.radius
    assert hdbscan_engine.include_chirality == unpickled_engine.include_chirality
    assert hdbscan_engine.scaffold == unpickled_engine.scaffold
    assert hdbscan_engine.min_cluster_size == unpickled_engine.min_cluster_size

    hdbscan_engine.fit(test_cases)
    unpickled_engine.fit(test_cases)

    result_original = hdbscan_engine.transform(test_cases)
    result_unpickled = unpickled_engine.transform(test_cases)

    # Check that both instances produce the same clustering results
    assert (result_original == result_unpickled).all()
    cloned_model = skl_base.clone(hdbscan_engine)

    assert hdbscan_engine.fp_size == cloned_model.fp_size
    assert hdbscan_engine.radius == cloned_model.radius
    assert hdbscan_engine.include_chirality == cloned_model.include_chirality
    assert hdbscan_engine.scaffold == cloned_model.scaffold
    assert hdbscan_engine.min_cluster_size == cloned_model.min_cluster_size

    cloned_model.fit(test_cases)
    result_cloned = cloned_model.transform(test_cases)
    assert (result_original == result_cloned).all()


def test_kmedoids_grouping_pickle_clone(test_cases):
    """Test that KmedoidsGroupingFromMols can be pickled and unpickled with all parameters preserved"""

    kmedoids_engine = core.KMedoidsGroupingFromMols(
        scaffold="GenericMurcko",
        clusters_number=10,
        radius=1,
        fp_size=1024,
        include_chirality=True,
        iteration_method="random",
        max_iter=100,
        random_state=112,
    )

    ## check if pickle/unpickle works

    pickled = pickle.dumps(kmedoids_engine)
    unpickled_engine = pickle.loads(pickled)

    assert kmedoids_engine.fp_size == unpickled_engine.fp_size
    assert kmedoids_engine.radius == unpickled_engine.radius
    assert kmedoids_engine.include_chirality == unpickled_engine.include_chirality
    assert kmedoids_engine.scaffold == unpickled_engine.scaffold
    assert kmedoids_engine.clusters_number == unpickled_engine.clusters_number
    assert kmedoids_engine.iteration_method == unpickled_engine.iteration_method

    kmedoids_engine.fit(test_cases)
    unpickled_engine.fit(test_cases)

    result_original = kmedoids_engine.transform(test_cases)
    result_unpickled = unpickled_engine.transform(test_cases)

    # Check that both instances produce the same clustering results
    assert (result_original == result_unpickled).all()

    # check if cloning works
    cloned_model = skl_base.clone(kmedoids_engine)

    assert kmedoids_engine.fp_size == cloned_model.fp_size
    assert kmedoids_engine.radius == cloned_model.radius
    assert kmedoids_engine.include_chirality == cloned_model.include_chirality
    assert kmedoids_engine.clusters_number == cloned_model.clusters_number
    assert kmedoids_engine.iteration_method == cloned_model.iteration_method

    cloned_model.fit(test_cases)
    result_cloned = cloned_model.transform(test_cases)
    assert (result_original == result_cloned).all()


# %%
# TODO join with test_pipeline groups and move to conftest.py
# @pytest.fixture(params=[{"tanimoto_grouping": {"parameters": {"smiliarity_threshold": 0.3}}}])
# def groups(mols: Iterable, request) -> pd.DataFrame:
#     cv_conf: cv_module.CVSettings = cv_module.CVSettings(**request.param)
#     cv_settings: cv_module.GenericCVModel = cv_conf.get_cv_settings()
#     groups: pd.DataFrame
#     if cv_conf.cv_type == cv_module.CVtype.TANIMOTO_GROUPING:
#         groups_engine: BaseEstimator = cv_module.TanimotoGroupingFromMols(**cv_settings.model_dump())
#         groups = groups_engine.set_output(transform="pandas").fit_transform(mols)  # type: ignore
#     elif cv_conf.cv_type == cv_module.CVtype.TIME_SERIES:
#         groups_engine: BaseEstimator = cv_module.TimeSeriesGrouping(**cv_settings.model_dump())
#         groups = groups_engine.set_output(transform="pandas").fit_transform(mols)  # type: ignore
#     elif cv_conf.cv_type == cv_module.CVtype.GROUPS:
#         groups_engine: BaseEstimator = cv_module.TimeSeriesGrouping(**cv_settings.model_dump())
#         groups = groups_engine.set_output(transform="pandas").fit_transform(mols)  # type: ignore
#     assert len(groups) == len(list(mols))
#     assert isinstance(groups, pd.DataFrame)
#     return groups
