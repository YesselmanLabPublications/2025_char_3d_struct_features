import pytest

from dms_3d_features.hbond import HbondCalculator

RESOURCE_PATH = "test/resources/"


def test_hbond_calculator():
    pdb_path = f"{RESOURCE_PATH}/pdbs/ACG_CU/TWOWAY.6N7R.0-1.CU-ACG.0.pdb"
    hbond_calculator = HbondCalculator()
    df = hbond_calculator.calculate_hbond_strength(pdb_path)
    df.to_csv("hbonds.csv", index=False)
    assert df is not None
    assert len(df) == 10
