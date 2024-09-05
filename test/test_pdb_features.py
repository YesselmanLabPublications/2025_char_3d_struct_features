import pandas as pd
import pytest
import os

from dms_3d_features.pdb_features import (
    compute_solvent_accessibility,
    HbondCalculator,
    generate_basepair_details_from_3dna,
    extract_basepair_details_into_a_table,
)
from dms_3d_features.paths import DATA_PATH

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
    assert df is not None
    assert len(df) == 10


def test_generate_basepair_details_from_3dna():
    pdb_path = f"{RESOURCE_PATH}/pdbs/ACG_CU/TWOWAY.6N7R.0-1.CU-ACG.0.pdb"
    generate_basepair_details_from_3dna(pdb_path)
    os.remove(f"{RESOURCE_PATH}/pdbs/ACG_CU/TWOWAY.6N7R.0-1.CU-ACG.0_x3dna.out")


def test_extract_bp_type_and_res_num_into_a_table():
    out_file = f"{DATA_PATH}/dssr-output/TWOWAY.6N7R.0-1.CU-ACG.0_x3dna.out"
    r = extract_basepair_details_into_a_table(out_file)
    print(r)
