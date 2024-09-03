import pytest
from dms_3d_features.process_motifs import trim

import pytest
import pandas as pd
import numpy as np


class TestTrim:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "sequence": ["ATCGATCG", "GCTAGCTA"],
                "structure": ["((..)).)", "(.(.)).)"],
                "data": [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
            }
        )

    def test_trim_start_only(self, sample_df):
        result = trim(sample_df, 2, 0)
        assert result["sequence"].tolist() == ["CGATCG", "TAGCTA"]
        assert result["structure"].tolist() == ["..)).)", "(.)).)"]
        assert result["data"].tolist() == [[3, 4, 5, 6, 7, 8], [6, 5, 4, 3, 2, 1]]

    def test_trim_end_only(self, sample_df):
        result = trim(sample_df, 0, 2)
        assert result["sequence"].tolist() == ["ATCGAT", "GCTAGC"]
        assert result["structure"].tolist() == ["((..))", "(.(.))"]
        assert result["data"].tolist() == [[1, 2, 3, 4, 5, 6], [8, 7, 6, 5, 4, 3]]

    def test_trim_start_and_end(self, sample_df):
        result = trim(sample_df, 2, 2)
        assert result["sequence"].tolist() == ["CGAT", "TAGC"]
        assert result["structure"].tolist() == ["..))", "(.))"]
        assert result["data"].tolist() == [[3, 4, 5, 6], [6, 5, 4, 3]]

    def test_trim_no_change(self, sample_df):
        result = trim(sample_df, 0, 0)
        assert result.equals(sample_df)

    def test_trim_empty_df(self):
        empty_df = pd.DataFrame(columns=["sequence", "structure", "data"])
        result = trim(empty_df, 2, 2)
        assert result.empty

    def test_trim_missing_columns(self):
        df = pd.DataFrame({"sequence": ["ATCG"], "other": [1]})
        result = trim(df, 1, 1)
        assert result["sequence"].tolist() == ["TC"]
        assert "other" in result.columns
        assert result["other"].tolist() == [1]

    def test_trim_data_as_numpy_array(self):
        df = pd.DataFrame(
            {"sequence": ["ATCGATCG"], "data": [np.array([1, 2, 3, 4, 5, 6, 7, 8])]}
        )
        result = trim(df, 2, 2)
        assert result["sequence"].tolist() == ["CGAT"]
        assert np.array_equal(result["data"].iloc[0], np.array([3, 4, 5, 6]))
