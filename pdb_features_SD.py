import glob
import numpy as np
import math
import freesasa
from biopandas.pdb import PandasPdb
import pandas as pd
import os
from typing import List
import subprocess

# solvent accessibility ##############################################################
DATA_PATH = "data/pdbs"


def compute_solvent_accessibility(
    pdb_path: str, probe_radius: float = 2.0
) -> pd.DataFrame:
    """
    Computes the solvent accessibility of atoms in a protein structure.

    Args:
        pdb_path: The path to the PDB file.

    Returns:
        A pandas DataFrame containing the solvent accessibility information for each atom.

    Raises:
        FileNotFoundError: If the PDB file specified by pdb_path does not exist.
    """
    ppdb = PandasPdb().read_pdb(pdb_path)
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
    m_sequence = pdb_path.split("/")[-2].replace("_", "&")
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
        else:
            continue
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


def compute_solvent_accessibility_all(pdb_dir):
    pdb_paths = glob.glob(f"{pdb_dir}/*/*.pdb")
    dfs = []
    for pdb_path in pdb_paths:
        df = compute_solvent_accessibility(pdb_path)
        dfs.append(df)
    df = pd.concat(dfs)
    return df


# hydrogen bonds #####################################################################


def generate_hbond_output_file_from_dssr(pdb_path: str) -> None:
    """
    Run x3dna-dssr to extract H-bonds from a model PDB file.

    Args:
        pdb_path (str): The path to the model PDB file.

    Returns:
        None

    Example:
        >>> generate_hbond_output_file_from_dssr('/path/to/model.pdb')
    """
    pdb_path_name = os.path.basename(pdb_path)
    output_file = f"data/dssr-output/{pdb_path_name[:-4]}_hbond.txt"
    command_dssr = f"x3dna-dssr -i={pdb_path} --get-hbonds -o={output_file}"
    subprocess.call(command_dssr, shell=True)
    try:
        os.remove("dssr-*")
    except:
        pass


def load_hbonds_file(file_path: str) -> pd.DataFrame:
    """
    Read the H-bonds file into a DataFrame.

    Args:
        file_path (str): The path to the H-bonds file.

    Returns:
        pd.DataFrame: The DataFrame containing H-bonds data.
    """
    return pd.read_csv(
        file_path,
        skiprows=2,
        delimiter="\s+",
        names=[
            "position_1",
            "position_2",
            "hbond_num",
            "type",
            "distance",
            "hbond_atoms",
            "atom_1",
            "atom_2",
        ],
    )


def extract_hbond_length(motif: str) -> pd.DataFrame:
    """
    Calculates the length of hydrogen bonds in a PDB file.

    Args:
        motif (str): Folder name where the pdb is (also the motif).

    Returns:
        pd.DataFrame: A DataFrame containing the lengths of hydrogen bonds.
                      The DataFrame has the following columns:
                      - Column 1: Atom 1
                      - Column 2: Atom 2
                      - Column 3: Hydrogen bond length

    Example:
        >>> pdb_path = "/path/to/pdb/file.pdb"
        >>> calculate_hbond_length(pdb_path)
        Returns a DataFrame with the hydrogen bond lengths.
    """

    filenames = sorted(glob.glob(f"{DATA_PATH}/{motif}/*.pdb"))
    all_hbonds = []

    for pdb in filenames:
        generate_hbond_output_file_from_dssr(pdb)
        txt_file = os.path.basename(pdb)
        read_file = load_hbonds_file(f"data/dssr-output/{txt_file[:-4]}_hbond.txt")
        all_hbonds.append(read_file)
        break

    return pd.concat(all_hbonds, ignore_index=True) if all_hbonds else pd.DataFrame()


def multiply_list(lst: list, char: str) -> list:
    """
    Generate a list of pairs from the list and character.

    Args:
        lst (list): The list of positions.
        char (str): The character to pair with each position.

    Returns:
        list: A list of pairs where each pair contains the character and a position.
    """
    return [char + str(elem) for elem in lst]


def find_positions_of_a_and_cs(motif: str) -> tuple:
    """
    Parse the motif and return positions for A and C nucleotides.

    Args:
        motif (str): The motif string in the format "motif1_motif2".

    Returns:
        tuple: Four lists containing positions of A and C nucleotides in both motifs.
    """
    motif1, motif2 = motif.split("_")
    a_pos1 = [(pos + 3) for pos, char in enumerate(motif1) if char == "A"]
    a_pos2 = [(pos + 7 + len(motif1)) for pos, char in enumerate(motif2) if char == "A"]
    c_pos1 = [(pos + 3) for pos, char in enumerate(motif1) if char == "C"]
    c_pos2 = [(pos + 7 + len(motif1)) for pos, char in enumerate(motif2) if char == "C"]
    return a_pos1, a_pos2, c_pos1, c_pos2


def calculate_hbond_strength(pdb_dir: str) -> pd.DataFrame:
    """
    Calculate the strength of H-bonds for the specified PDB file and motif.

    Args:
        pdb_dir (str): The path to the directory containing the PDB files.

    Returns:
        pd.DataFrame: A DataFrame containing H-bond strength data.
    """
    var_holder = {}
    all_data = []
    pdb_files = glob.glob(f"{pdb_dir}/*/*.pdb")

    for pdb in pdb_files:
        if pdb != "data/pdbs/ACG_CU/TWOWAY.6N7R.0-1.CU-ACG.0.pdb":
            continue
        motif = pdb.split("/")[-2]
        name = pdb.split("/")[-1]
        a_pos1, a_pos2, c_pos1, c_pos2 = find_positions_of_a_and_cs(motif)
        pos1 = a_pos1 + a_pos2 + c_pos1 + c_pos2
        a_pos = multiply_list(a_pos1, "A") + multiply_list(a_pos2, "A")
        c_pos = multiply_list(c_pos1, "C") + multiply_list(c_pos2, "C")
        pos = a_pos + c_pos

        for n, l in zip(pos1, pos):
            var_holder[f"n{n}"] = l
            var_holder[f"a{n}"] = "N1" if l[0] == "A" else "N3"

        n1 = var_holder[f"n{n}"]
        at1 = var_holder[f"a{n}"]

        pdb_name = os.path.basename(pdb)
        df_fn = extract_hbond_length(motif)

        for i, row in df_fn.iterrows():
            if not (row["distance"] < 3.3 and row["type"] == "p"):
                continue

            ppdb = PandasPdb().read_pdb(pdb)
            ATOM = ppdb.df["ATOM"]
            pos_1 = row["atom_1"].split("@")[1]
            pos_2 = row["atom_2"].split("@")[1]
            num_1 = int(pos_1[1:])
            num_2 = int(pos_2[1:])
            ps_1 = row["atom_1"].split("@")[0]
            ps_2 = row["atom_2"].split("@")[0]

            angle_atom_pair = {
                "N1": "C2",
                "N3": "C2",
                "N6": "C6",
                "N7": "C5",
                "N9": "C4",
                "O2": "C2",
                "N4": "C4",
                "N2": "C2",
                "O6": "C6",
                "O4": "C4",
                "O2'": "C2'",
                "O3'": "C3'",
                "O4'": "C4'",
                "O5'": "C5'",
                "OP1": "O5'",
            }

            if pos_1 == f"{n1}" or pos_2 == f"{n1}":
                if not (ps_1 in angle_atom_pair and ps_2 in angle_atom_pair):
                    continue
                coords_11 = ATOM[
                    (ATOM["atom_name"] == ps_1) & (ATOM["residue_number"] == num_1)
                ]
                coords_12 = ATOM[
                    (ATOM["atom_name"] == angle_atom_pair[ps_1])
                    & (ATOM["residue_number"] == num_1)
                ]
                coords_21 = ATOM[
                    (ATOM["atom_name"] == ps_2) & (ATOM["residue_number"] == num_2)
                ]
                coords_22 = ATOM[
                    (ATOM["atom_name"] == angle_atom_pair[ps_2])
                    & (ATOM["residue_number"] == num_2)
                ]
                if (
                    coords_11.empty
                    or coords_12.empty
                    or coords_21.empty
                    or coords_22.empty
                ):
                    continue
                a1 = coords_11[["x_coord", "y_coord", "z_coord"]].values[0]
                a2 = coords_12[["x_coord", "y_coord", "z_coord"]].values[0]
                b1 = coords_21[["x_coord", "y_coord", "z_coord"]].values[0]
                b2 = coords_22[["x_coord", "y_coord", "z_coord"]].values[0]
                a1a2 = a2 - a1
                a1b1 = b1 - a1
                angle_1_radian = np.arccos(
                    np.dot(a1a2, a1b1) / (np.linalg.norm(a1a2) * np.linalg.norm(a1b1))
                )
                angle_1_degrees = math.degrees(angle_1_radian)
                b1b2 = b2 - b1
                b1a1 = a1 - b1
                angle_2_radian = np.arccos(
                    np.dot(b1b2, b1a1) / (np.linalg.norm(b1b2) * np.linalg.norm(b1a1))
                )
                angle_2_degrees = math.degrees(angle_2_radian)

                if row["hbond_atoms"] == "O:O":
                    strength = (2.2 / row["distance"]) * 21
                elif row["hbond_atoms"] == "N:N":
                    strength = (2.2 / row["distance"]) * 13
                elif row["hbond_atoms"] == "N:O":
                    strength = (2.2 / row["distance"]) * 8
                else:
                    strength = (2.2 / row["distance"]) * 8

                data = {
                    "motif": motif,
                    "hbond_length": row["distance"],
                    "name": pdb_name,
                    "atom_1": row["atom_1"],
                    "atom_2": row["atom_2"],
                    "type": row["type"],
                    "angle_1": angle_1_degrees,
                    "angle_2": angle_2_degrees,
                    "nuc_1": pos_1,
                    "nuc_2": pos_2,
                    "n_or_other": (
                        "N-included"
                        if ps_1 == f"{at1}" or ps_2 == f"{at1}"
                        else "Other"
                    ),
                    "hbond_atoms": row["hbond_atoms"],
                    "hbond_strength": strength,
                }
                all_data.append(data)
        break

    return pd.DataFrame(all_data)


if __name__ == "__main__":
    df = calculate_hbond_strength(DATA_PATH)
    print(df)
