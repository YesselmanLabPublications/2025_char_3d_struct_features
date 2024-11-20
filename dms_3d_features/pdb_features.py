import glob
import concurrent.futures
from itertools import product
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional
import subprocess
import shutil
import regex as re
from biopandas.pdb import PandasPdb

from dms_3d_features.logger import get_logger
from dms_3d_features.paths import DATA_PATH
from dms_3d_features.stats import r2

log = get_logger("pdb-features")


# base pair level features #######################################################


def check_command_accessibility(command: str) -> None:
    """
    Check if a command is accessible in the system PATH.

    Args:
        command (str): The command to check.

    Raises:
        RuntimeError: If the command is not accessible or not found.
    """
    try:
        result = subprocess.run(
            ["which", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if not result.stdout.strip():
            raise RuntimeError(f"{command} not found in PATH")
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"{command} is not accessible. Please ensure 3DNA tools are properly installed and in your PATH."
        )


def generate_basepair_details_from_3dna(pdb: str, output_dir: str = None) -> None:
    """
    Generate base-pair details from a given PDB file using 3DNA tools.

    This function uses the 3DNA tools 'find_pair' and 'analyze' to generate base-pair
    details for the given PDB file. The output is saved with the same name as the PDB
    file but with an '.out' extension.

    Args:
        pdb (str): Path to the PDB file.
        output_dir (str): Path to the output directory. Defaults to None.
    """
    if output_dir is None:
        output_dir = os.path.dirname(pdb)

    # Check if required 3DNA tools are accessible
    check_command_accessibility("find_pair")
    check_command_accessibility("analyze")

    pdbname = os.path.basename(pdb)
    subprocess.call(
        f"find_pair {pdb} {pdb[:-4]}.inp",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.call(
        f"analyze {pdb[:-4]}.inp",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.move(f"{pdbname[:-4]}.out", f"{output_dir}/{pdbname[:-4]}_x3dna.out")
    remove_files = [
        "auxiliary.par",
        "bestpairs.pdb",
        "bp_helical.par",
        "bp_order.dat",
        "bp_step.par",
        "cf_7methods.par",
        "col_chains.scr",
        "col_helices.scr",
        "hel_regions.pdb",
        "ref_frames.dat",
        "stacking.pdb",
        "hstacking.pdb",
    ]
    for file in remove_files:
        os.remove(file)


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
    pattern = re.compile(
        r"\s*(\d*)\s*\((.*?)\)\s+.*?(\d+)_:\[\.\.(.)\](.)[-A-Z\*\+\-]+[A-Z]+\[\.\.(.)\]:\.*(\d+)_:-<.*?\((.*?)\)"
    )

    bp_types = []
    res_nums1 = []
    res_nums2 = []

    with open(filename, "r") as file:
        lines = file.readlines()

        for line in lines:
            match = pattern.search(line)
            if match:
                res_num1 = match.group(3)
                res_num2 = match.group(7)
                bp_type = "WC" if "-----" in line else "NON-WC"

                res_nums1.append(res_num1)
                res_nums2.append(res_num2)
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
    pattern = re.compile(
        r"\s*(\d*)\s*([A-Z]\+[A-Z]|[A-Z]-[A-Z])\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+"
        r"([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)"
    )

    # Extract base-pair types and residue numbers from a separate function
    bp_types, res_nums1, res_nums2 = extract_bp_type_and_res_num_into_a_table(filename)

    # Read file content
    with open(filename, "r") as file:
        lines = file.readlines()

    # Initialize variables
    is_shear_table = False
    motif_index = None
    all_data = []
    j = 0

    # Process each line
    for i, line in enumerate(lines):
        if "Shear    Stretch   Stagger    Buckle  Propeller  Opening" in line:
            is_shear_table = True
            continue

        if line.startswith("File name:"):
            motif_index = i

        if not is_shear_table:
            continue

        match = pattern.search(line)
        if not match:
            break

        motif = lines[motif_index].split("/")[2]
        bp = re.split(r"[-+]", match.group(2))[0] + re.split(r"[-+]", match.group(2))[1]

        data = {
            "name": os.path.basename(filename),
            "motif": motif,
            "r_type": bp_types[j],
            "res_num1": res_nums1[j],
            "res_num2": res_nums2[j],
            "bp": bp,
            "shear": float(match.group(3)),
            "stretch": float(match.group(4)),
            "stagger": float(match.group(5)),
            "buckle": float(match.group(6)),
            "propeller": float(match.group(7)),
            "opening": float(match.group(8)),
        }
        all_data.append(data)
        j += 1

    return pd.DataFrame(all_data)


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


def calculate_rmsd_bp(bp: str, filename: str, resi_nums: list) -> float:
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
        ppdb_ideal = PandasPdb().read_pdb(f"{DATA_PATH}/ideal_pdbs/{bp}.pdb")
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


def process_basepair_details():
    pdb_paths = sorted(glob.glob(f"{DATA_PATH}/pdbs/*/*.pdb"))
    output_dir = f"{DATA_PATH}/dssr-output/"

    all_tables = []
    for pdb_path in pdb_paths:
        pdb_name = os.path.basename(pdb_path)[:-4]
        x3dna_out_path = f"{output_dir}/{pdb_name}_x3dna.out"
        if not os.path.exists(x3dna_out_path):
            log.info(f"Generating basepair details for {pdb_name}")
            generate_basepair_details_from_3dna(pdb_path, output_dir)
        extracted_table = extract_basepair_details_into_a_table(x3dna_out_path)
        if not extracted_table.empty:
            all_tables.append(extracted_table)

    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_df.to_csv(f"{DATA_PATH}/csvs/all_bp_details.csv", index=False)
    filtered_df = combined_df[combined_df["r_type"] == "WC"].copy()

    rmsd = []
    for i, row in filtered_df.iterrows():
        res_num1 = int(row["res_num1"])
        res_num2 = int(row["res_num2"])

        pdb_path = f"{DATA_PATH}/pdbs/{row['motif']}/{row['name'][:-10]}.pdb"
        rmsd_val = calculate_rmsd_bp(row["bp"], pdb_path, [res_num1, res_num2])

        if rmsd_val is not None:
            print(
                f"RMSD for {row['bp']} at residues {res_num1}, {res_num2}: {rmsd_val}"
            )
        rmsd.append(rmsd_val)

    filtered_df["rmsd"] = rmsd
    filtered_df.to_csv(f"{DATA_PATH}/csvs/wc_with_rmsd.csv", index=False)
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
        elif key2 in dms_dict:
            r_data_list = dms_dict.get(key2, ["None"])
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
                "pdb_r_pos1": row1["res_num1"],
                "pdb_r_pos2": row1["res_num2"],
                "rmsd_from_ideal": row1["rmsd"],
                "shear": row1["shear"],
                "stretch": row1["stretch"],
                "stagger": row1["stagger"],
                "buckle": row1["buckle"],
                "propeller": row1["propeller"],
                "opening": row1["opening"],
                "r_data": r_data,
                "flanking_pairs": flanking_pair,
            }
            all_data.append(data)

    df_fin = pd.DataFrame(all_data)
    df_fin.to_csv(f"{DATA_PATH}/csvs/wc_details.csv", index=False)


## distance #######################################################################


def get_coordinates_for_atoms_in_res(df_atom: pd.DataFrame, resi_number: int):
    atoms_in_residue = df_atom[df_atom["residue_number"] == resi_number]
    if atoms_in_residue.empty:
        raise ValueError(f"No atoms found for residue {resi_number}")
    x_coords = atoms_in_residue["x_coord"].values
    y_coords = atoms_in_residue["y_coord"].values
    z_coords = atoms_in_residue["z_coord"].values
    atom_names = atoms_in_residue["atom_name"].values
    resi_names = atoms_in_residue["residue_name"].values
    return x_coords, y_coords, z_coords, atom_names, resi_names


def cal_distances_between_all_atom_pairs(
    x1: list, x2: list, y1: list, y2: list, z1: list, z2: list
):
    dist_pairs = []
    for k1 in range(len(x1)):
        for k2 in range(len(x2)):
            distance = np.sqrt(
                (x2[k2] - x1[k1]) ** 2 + (y2[k2] - y1[k1]) ** 2 + (z2[k2] - z1[k1]) ** 2
            )

            dist_pairs.append((k1, k2, distance))
    return dist_pairs


def get_distance_between_all_atom_pairs_dataframe(
    pdb_path: str, max_distance: float = 10
):
    try:
        ppdb = PandasPdb().read_pdb(pdb_path)
    except FileNotFoundError:
        log.error(f"PDB file not found: {pdb_path}")
        raise

    df_atom = ppdb.df["ATOM"]
    resi_num = df_atom["residue_number"].unique()

    motif = pdb_path.split("/")[-2].split("_")

    all_data = []

    if resi_num[0] != 3:
        strand_len = len(motif[1])
    else:
        strand_len = len(motif[0])

    for i, res1 in enumerate(resi_num[:strand_len]):
        for j, res2 in enumerate(resi_num[strand_len:]):
            if res1 == res2:
                continue

            x1, y1, z1, atom_names1, resi_names1 = get_coordinates_for_atoms_in_res(
                df_atom, res1
            )
            x2, y2, z2, atom_names2, resi_names2 = get_coordinates_for_atoms_in_res(
                df_atom, res2
            )

            dist_pairs = cal_distances_between_all_atom_pairs(x1, x2, y1, y2, z1, z2)

            for k1, k2, distance in dist_pairs:
                if distance > max_distance:
                    continue
                if res1 < res2:
                    data = {
                        "pdb_name": pdb_path.split("/")[-1],
                        "res_num1": res1,
                        "res_name1": resi_names1[k1],
                        "atom_name1": atom_names1[k1],
                        "res_num2": res2,
                        "res_name2": resi_names2[k2],
                        "atom_name2": atom_names2[k2],
                        "distance": round(distance, 2),
                    }
                else:
                    data = {
                        "pdb_name": pdb_path.split("/")[-1],
                        "res_num1": res2,
                        "res_name1": resi_names2[k2],
                        "atom_name1": atom_names2[k2],
                        "res_num2": res1,
                        "res_name2": resi_names1[k1],
                        "atom_name2": atom_names1[k1],
                        "distance": round(distance, 2),
                    }
                all_data.append(data)
    return pd.DataFrame(all_data)


def generate_distance_dataframe(max_distance: float = 10):
    folders = glob.glob(f"{DATA_PATH}/pdbs/*")
    all_dfs = []
    for folder in folders:
        filenames = glob.glob(f"{folder}/*.pdb")
        for file in filenames:
            df = get_distance_between_all_atom_pairs_dataframe(file, max_distance)
            all_dfs.append(df)
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df


## reactivity correlation with distance ##########################################


def calculate_atom_distances(df_pdb, df_dist, r_atom, pair_atom):
    data = []
    for i, g in df_pdb.groupby(["pdb_name", "pdb_r_pos"]):
        row = g.iloc[0]
        if row["pair_pdb_r_pos"] == -1:
            continue
        if row.pdb_r_pos < row.pair_pdb_r_pos:
            df_sub = df_dist.query(
                f'pdb_name == "{row.pdb_name}" and '
                f"res_num1 == {row.pdb_r_pos} and res_num2 == {row.pair_pdb_r_pos} and "
                f'atom_name1 == "{r_atom}" and atom_name2 == "{pair_atom}"'
            )
        else:
            df_sub = df_dist.query(
                f'pdb_name == "{row.pdb_name}" and '
                f"res_num1 == {row.pair_pdb_r_pos} and res_num2 == {row.pdb_r_pos} and "
                f'atom_name1 == "{pair_atom}" and atom_name2 == "{r_atom}"'
            )
        if len(df_sub) == 0:
            # print("couldnt find distance")
            continue
        data.append(
            {
                "pdb_name": row.pdb_path,
                "pdb_r_pos": row.pdb_r_pos,
                "pair_pdb_r_pos": row.pair_pdb_r_pos,
                "pdb_r_bp_type": row.pdb_r_bp_type,
                "distance": df_sub.iloc[0]["distance"],
                "average_b_factor": row.average_b_factor,
                "normalized_b_factor": row.normalized_b_factor,
                "pdb_res": row.pdb_res,
                "ln_r_data_mean": g["ln_r_data"].mean(),
                "ln_r_data_std": g["ln_r_data"].std(),
            }
        )
    return pd.DataFrame(data)


def process_pair_and_atoms(df_pdb, df_dist, args):
    p, atom1, atom2 = args
    df_pdb_pairs = df_pdb.query(
        f"r_nuc == '{p[0]}' and r_type == 'NON-WC' and pdb_r_pair == '{p}' and no_of_interactions == 1"
    ).copy()
    df_pdb_pairs.dropna(
        subset=["pdb_path", "pdb_r_bp_type", "ln_r_data", "average_b_factor"],
        inplace=True,
    )
    df_dist_pairs = calculate_atom_distances(df_pdb_pairs, df_dist, atom1, atom2)
    df_dist_pairs["pair"] = p
    df_dist_pairs["atom1"] = atom1
    df_dist_pairs["atom2"] = atom2
    df_dist_pairs["r2"] = r2(df_dist_pairs["distance"], df_dist_pairs["ln_r_data_mean"])
    return df_dist_pairs


def get_all_atom_distances():
    df_pdb = pd.read_json(
        f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues_pdb.json"
    )
    df_dist = pd.read_csv(f"{DATA_PATH}/pdb-features/distances_all.csv")
    df_bfact = pd.read_csv(f"{DATA_PATH}/pdb-features/b_factor.csv")
    df_bfact = df_bfact[
        ["pdb_name", "pdb_r_pos", "average_b_factor", "normalized_b_factor"]
    ]
    df_pdb = df_pdb.merge(df_bfact, on=["pdb_name", "pdb_r_pos"], how="left")

    pairs = ["A-G", "A-A", "C-A", "C-C", "C-U"]
    import multiprocessing
    from itertools import product

    all_combinations = []
    for pair in pairs:
        pair_atoms1 = list(
            df_dist.query(f"res_name1 == '{pair[0]}'")["atom_name1"].unique()
        )
        pair_atoms2 = list(
            df_dist.query(f"res_name1 == '{pair[2]}'")["atom_name1"].unique()
        )
        all_combinations.extend(list(product([pair], pair_atoms1, pair_atoms2)))

    with multiprocessing.Pool(processes=10) as pool:
        results = pool.starmap(
            process_pair_and_atoms,
            [(df_pdb, df_dist, combo) for combo in all_combinations],
        )

    df_all_results = pd.concat(results, ignore_index=True)
    df_all_results.to_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances.csv", index=False
    )


def get_non_canonical_atom_distances_reactivity_correlation():
    df = pd.read_csv(f"{DATA_PATH}/pdb-features/non_canonical_atom_distances.csv")
    data = []
    for (pair, atom1, atom2), g in df.groupby(["pair", "atom1", "atom2"]):
        data.append(
            {
                "pair": pair,
                "atom1": atom1,
                "atom2": atom2,
                "count": len(g),
                "r2": g["r2"].iloc[0],
            }
        )
    pd.DataFrame(data).to_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_reactivity_correlation.csv",
        index=False,
    )


## reactivity ratio with distance #################################################


def calculate_atom_distances_with_ratio(df, df_dist, r_atom, pair_atom, df_pdb):
    data_ratio = []
    seen = []
    for i, g in df.groupby(["pdb_name", "pdb_r_pos"]):
        row = g.iloc[0]
        if row["pair_pdb_r_pos"] == -1:
            continue
        if row.pdb_r_pos < row.pair_pdb_r_pos:
            df_sub = df_dist.query(
                f'pdb_name == "{row.pdb_name}" and '
                f"res_num1 == {row.pdb_r_pos} and res_num2 == {row.pair_pdb_r_pos} and "
                f'atom_name1 == "{r_atom}" and atom_name2 == "{pair_atom}"'
            )
        else:
            df_sub = df_dist.query(
                f'pdb_name == "{row.pdb_name}" and '
                f"res_num1 == {row.pair_pdb_r_pos} and res_num2 == {row.pdb_r_pos} and "
                f'atom_name1 == "{pair_atom}" and atom_name2 == "{r_atom}"'
            )
        if len(df_sub) == 0:
            continue

        key = (row.pdb_name, row.pdb_r_pos, row.pair_pdb_r_pos, r_atom)
        partner_key = (row.pdb_name, row.pair_pdb_r_pos, row.pdb_r_pos, pair_atom)
        if key in seen or partner_key in seen:
            continue
        seen.append(key)
        seen.append(partner_key)
        partner_g = df_pdb.query(
            f'pdb_name == "{row.pdb_name}" and pdb_r_pos == {row.pair_pdb_r_pos}'
        )
        if len(partner_g) == 0:
            continue
        ratio = g["ln_r_data"].mean() / partner_g["ln_r_data"].mean()
        data_ratio.append(
            {
                "pdb_name": row.pdb_path,
                "pdb_r_pos": row.pdb_r_pos,
                "pair_pdb_r_pos": row.pair_pdb_r_pos,
                "distance": df_sub.iloc[0]["distance"],
                "pdb_res": row.pdb_res,
                "ratio": ratio,
            }
        )

    return pd.DataFrame(data_ratio)


def process_pair_and_atoms_with_ratio(df_pdb, df_dist, args):
    p, atom1, atom2 = args
    df_pdb_pairs = df_pdb.query(
        f"r_nuc == '{p[0]}' and r_type == 'NON-WC' and pdb_r_pair == '{p}' and no_of_interactions == 1"
    ).copy()
    df_pdb_pairs.dropna(
        subset=["pdb_path", "pdb_r_bp_type", "ln_r_data", "average_b_factor"],
        inplace=True,
    )
    df_dist_pairs = calculate_atom_distances_with_ratio(
        df_pdb_pairs, df_dist, atom1, atom2, df_pdb
    )
    if len(df_dist_pairs) == 0:
        return pd.DataFrame()
    df_dist_pairs["pair"] = p
    df_dist_pairs["atom1"] = atom1
    df_dist_pairs["atom2"] = atom2
    df_dist_pairs["r2"] = r2(df_dist_pairs["distance"], df_dist_pairs["ratio"])
    return df_dist_pairs


def get_all_atom_distances_with_ratio():
    df_pdb = pd.read_json(
        f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues_pdb.json"
    )
    df_dist = pd.read_csv(f"{DATA_PATH}/pdb-features/distances_all.csv")
    df_bfact = pd.read_csv(f"{DATA_PATH}/pdb-features/b_factor.csv")
    df_bfact = df_bfact[
        ["pdb_name", "pdb_r_pos", "average_b_factor", "normalized_b_factor"]
    ]
    df_pdb = df_pdb.merge(df_bfact, on=["pdb_name", "pdb_r_pos"], how="left")

    pairs = ["A-A", "C-A", "C-C"]
    import multiprocessing
    from itertools import product

    all_combinations = []
    for pair in pairs:
        pair_atoms1 = list(
            df_dist.query(f"res_name1 == '{pair[0]}'")["atom_name1"].unique()
        )
        pair_atoms2 = list(
            df_dist.query(f"res_name1 == '{pair[2]}'")["atom_name1"].unique()
        )
        all_combinations.extend(list(product([pair], pair_atoms1, pair_atoms2)))

    with multiprocessing.Pool(processes=10) as pool:
        results = pool.starmap(
            process_pair_and_atoms_with_ratio,
            [(df_pdb, df_dist, combo) for combo in all_combinations],
        )

    df_all_results = pd.concat(results, ignore_index=True)
    df_all_results.to_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_with_ratio.csv",
        index=False,
    )


def get_non_canonical_atom_distances_reactivity_ratio_correlation():
    df = pd.read_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_with_ratio.csv"
    )
    data = []
    for (pair, atom1, atom2), g in df.groupby(["pair", "atom1", "atom2"]):
        data.append(
            {
                "pair": pair,
                "atom1": atom1,
                "atom2": atom2,
                "count": len(g),
                "r2": g["r2"].iloc[0],
            }
        )
    pd.DataFrame(data).to_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_reactivity_ratio_correlation.csv",
        index=False,
    )


if __name__ == "__main__":
    """df = pd.read_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_reactivity_correlation.csv"
    )
    for i, g in df.groupby("pair"):
        print(i)
        g = g.sort_values(by="r2", ascending=False)
        pair_count = g["count"].max()
        top_5 = g.head(20)
        for _, row in top_5.iterrows():
            print(
                row["pair"],
                row["atom1"],
                row["atom2"],
                row["r2"],
                row["count"],
                pair_count,
            )
    """
    get_all_atom_distances_with_ratio()
    get_non_canonical_atom_distances_reactivity_ratio_correlation()
    exit()
    df = pd.read_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_reactivity_ratio_correlation.csv"
    )
    for i, g in df.groupby("pair"):
        print(i)
        g = g.sort_values(by="r2", ascending=False)
        pair_count = g["count"].max()
        top_5 = g.head(20)
        for _, row in top_5.iterrows():
            print(
                row["pair"],
                row["atom1"],
                row["atom2"],
                row["r2"],
                row["count"],
                pair_count,
            )
