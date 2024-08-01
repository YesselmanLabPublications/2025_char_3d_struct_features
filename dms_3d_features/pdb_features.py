import glob
import numpy as np
import math
import freesasa
import biopandas.pdb as PandasPdb
import pandas as pd
import os
from typing import List
import subprocess

# solvent accessibility ##############################################################


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
    ppdb = PandasPdb.PandasPdb()
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
    output_file = f"data/dssr-output/{pdb_path_name}_hbond.txt"
    command_dssr = f"x3dna-dssr -i={pdb_path} --get-hbonds -o={output_file}"
    subprocess.call(command_dssr, shell=True)
    try:
        os.remove("dssr-*")
    except:
        pass


def read_hbonds_file(file_path: str) -> pd.DataFrame:
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


def calculate_hbond_length(pdb_path: str) -> pd.DataFrame:
    """
    Calculates the length of hydrogen bonds in a PDB file.

    Args:
        pdb_path (str): The path to the PDB file.

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

    filenames = sorted(glob.glob(f"{DATA_PATH}/{motif}/{name}/*.pdb"))
    all_hbonds = []

    for model_pdb in filenames:
        extract_hbonds(model_pdb)
        txt_path = os.path.basename(model_pdb)
        read_file = read_hbonds_file(
            f"{DATA_PATH}/{motif}/{name}/{txt_path[:-4]}_FARFAR-hbonds.txt"
        )
        all_hbonds.append(read_file)

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


def calculate_hbond_strength(pdb_dir) -> pd.DataFrame:
    """
    Calculate the strength of H-bonds for the specified PDB file and motif.

    Args:
        pdb_file (str): The name of the PDB file.
        path (str): The path to the directory containing the PDB files.
        motif (str): The motif string in the format "motif1_motif2".
    """

    var_holder = {}
    hbond_data = {
        "name": [],
        "motif": [],
        "atom_1": [],
        "nuc_1": [],
        "atom_2": [],
        "nuc_2": [],
        "h_bond_length": [],
        "type": [],
        "n_or_other": [],
        "hbond_atoms": [],
        "hbond_strength": [],
        "angle": [],
    }
    pdb_files = glob.glob(f"{pdb_dir}/*/*.pdb")
    for pdb in pdb_files:
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
        a1 = var_holder[f"a{n}"]

        pdb_name = os.path.basename(pdb)
        df_fn = calculate_hbond_length(motif, name)

        for i, row in df_fn.iterrows():
            if row["distance"] < 3.3 and row["type"] == "p":
                ppdb = PandasPdb().read_pdb(pdb)
                ATOM = ppdb.df["ATOM"]
                pos_1 = row["atom_1"].split("@")[1]
                pos_2 = row["atom_2"].split("@")[1]
                num_1 = int(pos_1[1:])
                num_2 = int(pos_2[1:])
                ps_1 = row["atom_1"].split("@")[0]
                ps_2 = row["atom_2"].split("@")[0]
                if pos_1 == f"{n1}" or pos_2 == f"{n1}":

                    coords_1 = ATOM[
                        (ATOM["atom_name"] == ps_1) & (ATOM["residue_number"] == num_1)
                    ]
                    coords_2 = ATOM[
                        (ATOM["atom_name"] == ps_2) & (ATOM["residue_number"] == num_2)
                    ]
                    if coords_1.empty or coords_2.empty:
                        continue
                    a = coords_1[["x_coord", "y_coord", "z_coord"]].values[0]
                    b = coords_2[["x_coord", "y_coord", "z_coord"]].values[0]
                    mod_a = np.linalg.norm(a)
                    mod_b = np.linalg.norm(b)
                    angle_radian = np.arccos(np.dot(a, b) / (mod_a * mod_b))
                    angle_degrees = math.degrees(angle_radian)
                    hbond_data["motif"].append(motif)
                    hbond_data["h_bond_length"].append(row["distance"])
                    hbond_data["name"].append(pdb_name)
                    hbond_data["atom_1"].append(row["atom_1"])
                    hbond_data["atom_2"].append(row["atom_2"])
                    hbond_data["type"].append(row["type"])
                    hbond_data["angle"].append(angle_degrees)
                    hbond_data["nuc_1"].append(pos_1)
                    hbond_data["nuc_2"].append(pos_2)
                    hbond_data["n_or_other"].append(
                        "N-included" if ps_1 == f"{a1}" or ps_2 == f"{a1}" else "Other"
                    )
                    hbond_data["hbond_atoms"].append(row["hbond_atoms"])
                    if row["hbond_atoms"] == "O:O":
                        strength = (2.2 / row["distance"]) * 21
                    elif row["hbond_atoms"] == "N:N":
                        strength = (2.2 / row["distance"]) * 13
                    elif row["hbond_atoms"] == "N:O":
                        strength = (2.2 / row["distance"]) * 8
                    else:
                        strength = (2.2 / row["distance"]) * 8
                    hbond_data["hbond_strength"].append(strength)
        break

    return pd.DataFrame(hbond_data)
