import subprocess
import glob
import pandas as pd
import os
import shutil
from biopandas.pdb import PandasPdb
import numpy as np

DATA_PATH = "data"
RESOURCE_PATH = "resources"

def generate_basepair_details_from_3dna(pdb: str) -> None:
    """
    Generate base-pair details from a given PDB file using 3DNA tools.

    This function uses the 3DNA tools 'find_pair' and 'analyze' to generate base-pair
    details for the given PDB file. The output is saved with the same name as the PDB
    file but with an '.out' extension.

    Args:
        pdb (str): Path to the PDB file.
    """
    pdbname = os.path.basename(pdb)
    cmd_find = f"find_pair {pdb} {pdb[:-4]}.inp"
    subprocess.call(cmd_find, shell=True)
    cmd_analyze = f"analyze {pdb[:-4]}.inp"
    subprocess.call(cmd_analyze, shell=True)
    shutil.move(f"{pdbname[:-4]}.out", f"{pdb[:-4]}_bp_data.out")


def extract_bp_type_and_res_num_into_a_table(filename: str) -> list:
    """
    Extract base-pair type and residue numbers from the 3DNA output file.

    This function parses the 3DNA output file to extract the base-pair type (WC or NON-WC)
    and residue numbers for the base pairs.

    Args:
        filename (str): Path to the 3DNA output file.

    Returns:
        list: A list containing three lists: base-pair types, residue numbers for the first base,
              and residue numbers for the second base.
    """
    start_marker = (
        "RMSD of the bases (----- for WC bp, + for isolated bp, x for helix change)"
    )
    end_marker_1 = "Note: This structure contains"
    end_marker_2 = (
        "****************************************************************************"
    )

    bp_types = []
    res_nums1 = []
    res_nums2 = []

    with open(filename, "r") as file:
        lines = file.readlines()

    start_index = end_index_1 = end_index_2 = None

    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i
        if end_marker_1 in line and start_index is not None:
            end_index_1 = i
            break
        elif end_marker_2 in line and start_index is not None:
            end_index_2 = i
            break

    if start_index is not None:
        if end_index_1 is not None:
            table_section = lines[start_index + 3 : end_index_1 - 1]
        elif end_index_2 is not None:
            table_section = lines[start_index + 3 : end_index_2]
        for j, line in enumerate(table_section):
            tokens = line.split()

            res_num2 = tokens[2].split("_")[1].split(".")[-1]
            res_nums2.append(res_num2)
            if tokens[2].split("_")[0].split(":")[1].split(".")[2] == "":
                res_num1 = tokens[2].split("_")[0].split(":")[1].split(".")[3]
                res_nums1.append(res_num1)
            else:
                res_num1 = tokens[2].split("_")[0].split(":")[1].split(".")[2]
                res_nums1.append(res_num1)

            bp = tokens[2].split("]")[1].split("[")[0][1:-1]

            if bp != "-----":
                bp_type = "NON-WC"
                bp_types.append(bp_type)
            else:
                bp_type = "WC"
                bp_types.append(bp_type)

    return bp_types, res_nums1, res_nums2


def extract_basepair_details_into_a_table(filename: str) -> pd.DataFrame:
    """
    Extract base-pair details from the 3DNA output file and save them into a DataFrame.

    This function reads the 3DNA output file, extracts the base-pair parameters, and
    saves them into a pandas DataFrame.

    Args:
        filename (str): Path to the 3DNA output file.

    Returns:
        pd.DataFrame: DataFrame containing the extracted base-pair parameters.
    """
    start_marker = "Local base-pair parameters"
    end_marker_1 = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    end_marker_2 = (
        "****************************************************************************"
    )

    with open(filename, "r") as file:
        lines = file.readlines()

    start_index = end_index_1 = end_index_2 = None

    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i
        if end_marker_1 in line and start_index is not None:
            end_index_1 = i
            break
        elif end_marker_2 in line and start_index is not None:
            end_index_2 = i
            break

    all_data = []

    if start_index is not None:
        if end_index_1 is not None:
            table_section = lines[start_index + 2 : end_index_1]
        elif end_index_2 is not None:
            table_section = lines[start_index + 2 : end_index_2]
        for j, line in enumerate(table_section):
            tokens = line[1:].split()
            res = extract_bp_type_and_res_num_into_a_table(filename)
            bp = tokens[1][0] + tokens[1][-1]
            motif = filename.split("/")[-2]
            data = {
                "name": os.path.basename(filename),
                "motif": motif,
                "r_type": res[0][j],
                "res_num1": res[1][j],
                "res_num2": res[2][j],
                "bp": bp,
                "shear": tokens[2],
                "stretch": tokens[3],
                "stagger": tokens[4],
                "buckle": tokens[5],
                "propeller": tokens[6],
                "opening": tokens[7],
            }
            all_data.append(data)
        return pd.DataFrame(all_data)
    else:
        return pd.DataFrame()


def kabsch_algorithm(P: list, Q: list) -> list:
    """
    Perform the Kabsch algorithm to find the optimal rotation matrix
    that aligns matrix P to matrix Q.

    Args:
    P (list): A list of coordinates representing the first structure.
    Q (list): A list of coordinates representing the second structure.

    Returns:
    list: The optimal rotation matrix that aligns P to Q.
    """
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)

    C = np.dot(np.transpose(P_centered), Q_centered)

    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, W)

    return U


def superimpose_structures(mobile_coords: list, target_coords: list) -> list:
    """
    Superimpose the mobile structure onto the target structure using the Kabsch algorithm.

    Args:
    mobile_coords (list): A list of coordinates for the mobile structure.
    target_coords (list): A list of coordinates for the target structure.

    Returns:
    list: The rotated mobile coordinates that best align with the target structure.
    """
    rotation_matrix = kabsch_algorithm(mobile_coords, target_coords)
    mobile_center = np.mean(mobile_coords, axis=0)
    target_center = np.mean(target_coords, axis=0)

    mobile_coords_aligned = (
        np.dot(mobile_coords - mobile_center, rotation_matrix) + target_center
    )

    return mobile_coords_aligned


def rmsd_calculation_for_bp(
    allowed_atoms: dict,
    bp: str,
    ideal_df: pd.DataFrame,
    selected_pdb_df: pd.DataFrame,
    resi_nums: list,
) -> float:
    """
    Calculate the RMSD (Root Mean Square Deviation) for a base pair.

    Args:
    allowed_atoms (dict): Dictionary of allowed atom names for each base type.
    bp (str): The base pair (e.g., "AU", "GC").
    ideal_df (pd.DataFrame): DataFrame containing the ideal structure coordinates.
    selected_pdb_df (pd.DataFrame): DataFrame containing the selected structure's coordinates.
    resi_nums (list): List of residue numbers for the base pair in the selected structure.

    Returns:
    float: The RMSD value between the ideal and selected structures. Returns None if calculation is not possible.
    """
    total_rmsd = 0
    count = 0

    ideal_coords_list = []
    pdb_code_list = []

    for atom in allowed_atoms[bp[0]]:
        atom_ideal_df = ideal_df[
            (ideal_df["residue_number"] == 1) & (ideal_df["atom_name"] == atom)
        ]
        atom_selected_pdb_df = selected_pdb_df[
            (selected_pdb_df["residue_number"] == resi_nums[0])
            & (selected_pdb_df["atom_name"] == atom)
        ]

        if not atom_ideal_df.empty and not atom_selected_pdb_df.empty:
            ideal_coords_list.append(
                atom_ideal_df[["x_coord", "y_coord", "z_coord"]].values[0]
            )
            pdb_code_list.append(
                atom_selected_pdb_df[["x_coord", "y_coord", "z_coord"]].values[0]
            )

    for atom in allowed_atoms[bp[1]]:
        atom_ideal_df = ideal_df[
            (ideal_df["residue_number"] == 2) & (ideal_df["atom_name"] == atom)
        ]
        atom_selected_pdb_df = selected_pdb_df[
            (selected_pdb_df["residue_number"] == resi_nums[1])
            & (selected_pdb_df["atom_name"] == atom)
        ]

        if not atom_ideal_df.empty and not atom_selected_pdb_df.empty:
            ideal_coords_list.append(
                atom_ideal_df[["x_coord", "y_coord", "z_coord"]].values[0]
            )
            pdb_code_list.append(
                atom_selected_pdb_df[["x_coord", "y_coord", "z_coord"]].values[0]
            )

    if ideal_coords_list and pdb_code_list:
        ideal_coords = np.array(ideal_coords_list)
        selected_coords = np.array(pdb_code_list)

        aligned_selected_coords = superimpose_structures(selected_coords, ideal_coords)

        total_rmsd = np.sum((ideal_coords - aligned_selected_coords) ** 2)
        count = len(ideal_coords)

        if count > 0:
            rmsd_result = np.sqrt(total_rmsd / count)
            return rmsd_result
        else:
            return None
    else:
        return None


def calculate_bp_rmsd(bp: str, filename: str, resi_nums: list) -> float:
    """
    Calculate the RMSD for a given base pair in a PDB structure.
    """
    allowed_atoms = {
        "A": ["N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"],
        "G": ["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"],
        "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
        "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    }

    try:
        ppdb_ideal = PandasPdb().read_pdb(f"{RESOURCE_PATH}/ideal_pdbs/{bp}.pdb")
        ideal_df = ppdb_ideal.df["ATOM"]
        ppdb_pdb = PandasPdb().read_pdb(filename)
        pdb_df = ppdb_pdb.df["ATOM"]
        selected_pdb_df = pdb_df[pdb_df["residue_number"].isin(resi_nums)]

        if selected_pdb_df.empty:
            print(
                f"Warning: Selected PDB DataFrame is empty for {bp} at residues {resi_nums}"
            )

        rmsd_value = rmsd_calculation_for_bp(
            allowed_atoms, bp, ideal_df, selected_pdb_df, resi_nums
        )

        if rmsd_value is None:
            print(f"RMSD could not be calculated for {bp} at residues {resi_nums}")
        return rmsd_value

    except Exception as e:
        print(f"Error calculating RMSD for {bp} in {filename}: {e}")
        return None


filenames = sorted(glob.glob(f"{DATA_PATH}/pdbs/*/*.pdb"))

all_tables = []
for out in filenames:
    generate_basepair_details_from_3dna(out)
    extracted_table = extract_basepair_details_into_a_table(f"{out[:-4]}_bp_data.out")
    if not extracted_table.empty:
        all_tables.append(extracted_table)

combined_df = pd.concat(all_tables, ignore_index=True)
filtered_df = combined_df[combined_df["r_type"] == "WC"].copy()

RMSD = []
allowed_bp = ["AU", "UA", "CG", "GC", "GU", "UG"]

for i, row in filtered_df.iterrows():
    res_num1 = int(row["res_num1"])
    res_num2 = int(row["res_num2"])

    pdb_path = f"{DATA_PATH}/pdbs/{row['motif']}/{row['name'][:-12]}.pdb"
    rmsd_val = calculate_bp_rmsd(row["bp"], pdb_path, [res_num1, res_num2])

    if rmsd_val is not None:
        print(f"RMSD for {row['bp']} at residues {res_num1}, {res_num2}: {rmsd_val}")
    RMSD.append(rmsd_val)

filtered_df["rmsd"] = RMSD
filtered_df.to_csv(f"{RESOURCE_PATH}/csvs/wc_with_rmsd.csv", index=False)
df_all = pd.read_json(f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues.json")

dms_dict = {}
for k, all_row in df_all.iterrows():
    key = (all_row["m_sequence"], all_row["r_nuc"], all_row["pdb_r_pos"])
    if key not in dms_dict:
        dms_dict[key] = all_row["r_data"]
    else:
        if isinstance(dms_dict[key], list):
            dms_dict[key].append(all_row["r_data"])
        else:
            dms_dict[key] = [dms_dict[key], all_row["r_data"]]

all_data = []
for j, row1 in filtered_df.iterrows():
    key1 = (row1["motif"].replace("_", "&"), row1["bp"][0], int(row1["res_num1"]))
    key2 = (row1["motif"].replace("_", "&"), row1["bp"][1], int(row1["res_num2"]))

    r_data_list = []
    if key1 in dms_dict:
        r_data_list = dms_dict.get(key1, ["None"])
        print(f"Key1 found: {key1}, r_data_list: {r_data_list}")
    elif key2 in dms_dict:
        r_data_list = dms_dict.get(key2, ["None"])
        print(f"Key2 found: {key2}, r_data_list: {r_data_list}")
    else:
        print(f"No match found for {key1} or {key2}, skipping...")
        continue
    strand_1_length = len(row1["motif"].split("_")[0])

    if (int(row1["res_num1"]) == 3 or int(row1["res_num2"]) == 3) and (
        int(row1["res_num1"]) == (len(row1["motif"]) + 5)
        or int(row1["res_num2"]) == (len(row1["motif"]) + 5)
    ):
        flanking_pair = "YES"
    elif (
        int(row1["res_num1"]) == (strand_1_length + 2)
        or int(row1["res_num2"]) == (strand_1_length + 2)
    ) and (
        int(row1["res_num1"]) == (strand_1_length + 7)
        or int(row1["res_num2"]) == (strand_1_length + 7)
    ):
        flanking_pair = "YES"
    else:
        flanking_pair = "NO"

    for r_data in r_data_list:
        data = {
            "m_sequence": row1["motif"].replace("_", "&"),
            "bp": row1["bp"],
            "pdb_r_pos1": int(row1["res_num1"]),
            "pdb_r_pos2": int(row1["res_num2"]),
            "rmsd_from_ideal": row1["rmsd"],
            "shear": float(row1["shear"]),
            "stretch": float(row1["stretch"]),
            "stagger": float(row1["stagger"]),
            "buckle": float(row1["buckle"]),
            "propeller": float(row1["propeller"]),
            "opening": float(row1["opening"]),
            "r_data": r_data,
            "flanking_pairs": flanking_pair,
        }
        all_data.append(data)

#cmd_delete = "rm -rf *.pdb *.par *.dat *.scr *.r3d"
#subprocess.call(cmd_delete, shell=True)

df_fin = pd.DataFrame(all_data)
df_fin.to_json(f"{RESOURCE_PATH}/jsons/wc_details.json", orient="records")
