import pandas as pd
import pytest
import os

from dms_3d_features.pdb_features import (
    generate_basepair_details_from_3dna,
    extract_basepair_details_into_a_table,
)
from dms_3d_features.paths import DATA_PATH

RESOURCE_PATH = "test/resources/"


def test_generate_basepair_details_from_3dna():
    pdb_path = f"{RESOURCE_PATH}/pdbs/ACG_CU/TWOWAY.6N7R.0-1.CU-ACG.0.pdb"
    generate_basepair_details_from_3dna(pdb_path)
    os.remove(f"{RESOURCE_PATH}/pdbs/ACG_CU/TWOWAY.6N7R.0-1.CU-ACG.0_x3dna.out")


def test_extract_bp_type_and_res_num_into_a_table():
    out_file = f"{DATA_PATH}/dssr-output/TWOWAY.6N7R.0-1.CU-ACG.0_x3dna.out"
    r = extract_basepair_details_into_a_table(out_file)
    print(r)
