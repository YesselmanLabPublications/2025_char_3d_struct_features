import glob
import freesasa
import biopandas.pdb as PandasPdb
import pandas as pd
import os
from typing import List


def compute_solvent_accessability(pdb_path: str, probe_radius: 2.0) -> pd.DataFrame:
    """
    Computes the solvent accessibility of atoms in a protein structure.

    Args:
        pdb_path: The path to the PDB file.

    Returns:
        A pandas DataFrame containing the solvent accessibility information for each atom.

    Raises:
        FileNotFoundError: If the PDB file specified by pdb_path does not exist.
    """
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_path)
    ATOM = ppdb.df["ATOM"]
    resname = ATOM["residue_name"]
    atom = ATOM["atom_name"]
    resi_number = ATOM["residue_number"]
    length = len(ATOM.index)
    params = freesasa.Parameters(
        {"algorithm": freesasa.LeeRichards, "probe-radius": probe_radius}
    )
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure, params)
    m_sequence = pdb_path.split("/")[1].replace("_", "&")
    all_data = []
    for i in range(length):
        sasa = 0.0
        if atom[i] != "N1":
            continue
        selection = freesasa.selectArea(
            (
                f"n1,(name N1) and (resn A) and (resi {resi_number[i]}) ",
                f"n3, (name N3) and (resn C) and (resi {resi_number[i]})",
            ),
            structure,
            result,
        )
        if resname[i] == "A":
            sasa = selection["n1"]
        elif resname[i] == "C":
            sasa = selection["n3"]
        data = {
            "pdb": pdb_path,
            "m_sequence": m_sequence,
            "r_nuc": resname[i],
            "r_loc_pos": resi_number[i],
            "sasa": sasa,
        }
        all_data.append(data)
    df = pd.DataFrame(all_data)
    return df
