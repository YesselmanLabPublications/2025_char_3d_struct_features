import pandas as pd
import pytest

from dms_3d_features.pdb_features import (
    compute_solvent_accessibility,
    HbondCalculator,
)

RESOURCE_PATH = "test/resources/"


def _test_compute_solvent_accessibility():
    pdb_path = f"{RESOURCE_PATH}/pdbs/TWOWAY.3WBM.2-2.GACU-ACCC.0.pdb"
    df = compute_solvent_accessibility(pdb_path)
    df_org = pd.read_csv(f"{RESOURCE_PATH}/csvs/org_sasa.csv")
    df_merge = pd.merge(df, df_org, on=["r_nuc", "r_loc_pos"])
    for i, row in df_merge.iterrows():
        assert pytest.approx(row["sasa_x"]) == row["sasa_y"]


def test_hbond_calculator():
    pdb_path = f"{RESOURCE_PATH}/pdbs/ACG_CU/TWOWAY.6N7R.0-1.CU-ACG.0.pdb"
    hbond_calculator = HbondCalculator()
    df = hbond_calculator.calculate_hbond_strength(pdb_path)
    print(df)
