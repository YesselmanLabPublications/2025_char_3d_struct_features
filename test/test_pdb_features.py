import pandas as pd
import pytest

from dms_3d_features.pdb_features import (
    compute_solvent_accessibility,
    generate_hbond_output_file_from_dssr,
)

RESOURCE_PATH = "test/resources/"


def test_compute_solvent_accessibility():
    pdb_path = f"{RESOURCE_PATH}/pdbs/TWOWAY.3WBM.2-2.GACU-ACCC.0.pdb"
    df = compute_solvent_accessibility(pdb_path)
    df_org = pd.read_csv(f"{RESOURCE_PATH}/csvs/org_sasa.csv")
    df_merge = pd.merge(df, df_org, on=["r_nuc", "r_loc_pos"])
    for i, row in df_merge.iterrows():
        assert pytest.approx(row["sasa_x"]) == row["sasa_y"]


def test_generate_hbond_output_file_from_dssr():
    pdb_path = f"{RESOURCE_PATH}/pdbs/TWOWAY.3WBM.2-2.GACU-ACCC.0.pdb"
    generate_hbond_output_file_from_dssr(pdb_path)
