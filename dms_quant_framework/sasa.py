import pandas as pd
import freesasa
from biopandas.pdb import PandasPdb
import os
import glob

from dms_quant_framework.logger import get_logger

log = get_logger("sasa")


def compute_solvent_accessibility(
    pdb_path: str, probe_radius: float = 2.0
) -> pd.DataFrame:
    """
    Computes the solvent accessibility of specific atoms in a nucleic acid structure.

    Args:
        pdb_path (str): The path to the PDB file.
        probe_radius (float): The probe radius for SASA calculation. Defaults to 2.0.

    Returns:
        pd.DataFrame: A DataFrame containing the solvent accessibility information for
          N1 atoms of A and N3 atoms of C.

    Raises:
        FileNotFoundError: If the PDB file specified by pdb_path does not exist.
    """
    try:
        ppdb = PandasPdb().read_pdb(pdb_path)
    except FileNotFoundError:
        log.error(f"PDB file not found: {pdb_path}")
        raise

    ATOM = ppdb.df["ATOM"]
    # Filter for N1 atoms of A and N3 atoms of C
    mask = ((ATOM["residue_name"] == "A") & (ATOM["atom_name"] == "N1")) | (
        (ATOM["residue_name"] == "C") & (ATOM["atom_name"] == "N3")
    )
    filtered_ATOM = ATOM[mask]

    params = freesasa.Parameters(
        {"algorithm": freesasa.LeeRichards, "probe-radius": probe_radius}
    )
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure, params)

    m_sequence = os.path.basename(os.path.dirname(pdb_path)).replace("_", "&")

    all_data = []
    for _, row in filtered_ATOM.iterrows():
        selection = freesasa.selectArea(
            (
                f"n, (name {row['atom_name']}) and (resn {row['residue_name']}) and (resi {row['residue_number']})",
            ),
            structure,
            result,
        )
        data = {
            "pdb_path": pdb_path,
            "m_sequence": m_sequence,
            "r_nuc": row["residue_name"],
            "pdb_r_pos": row["residue_number"],
            "sasa": selection["n"],
        }
        all_data.append(data)

    return pd.DataFrame(all_data)


def compute_solvent_accessibility_all(
    pdb_dir: str, probe_radius: float = 2.0
) -> pd.DataFrame:
    """
    Computes the solvent accessibility for all PDB files in a directory.

    Args:
        pdb_dir (str): The directory containing the PDB files.
        probe_radius (float, optional): The probe radius for computing solvent accessibility.
            Defaults to 2.0.

    Returns:
        pd.DataFrame: A DataFrame containing the computed solvent accessibility values.

    """
    pdb_paths = glob.glob(f"{pdb_dir}/*/*.pdb")
    log.info(f"Processing {len(pdb_paths)} PDB files.")
    dfs = []
    for pdb_path in pdb_paths:
        try:
            df = compute_solvent_accessibility(pdb_path, probe_radius)
            dfs.append(df)
        except Exception as e:
            log.error(f"Error processing {pdb_path}: {str(e)}")

    if not dfs:
        log.warning("No PDB files were successfully processed.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Processed {len(dfs)} PDB files successfully.")
    return df


def generate_sasa_dataframe():
    dfs = []
    for probe_radius in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        # need to use pdbs with 2 extra base pairs built by farfar
        log.info(f"Processing probe radius: {probe_radius}")
        df = compute_solvent_accessibility_all("data/pdbs_w_2bp", probe_radius)
        probe_radius = str(probe_radius).replace(".", "_")
        df.rename({"sasa": f"sasa_{probe_radius}"}, axis=1, inplace=True)
        dfs.append(df)
    df_final = dfs[0]
    dfs.pop(0)
    for df in dfs:
        df_final = df_final.merge(
            df, on=["m_sequence", "pdb_r_pos", "pdb_path", "r_nuc"]
        )
    return df_final
