import glob
import numpy as np
import math
import freesasa
import biopandas.pdb as PandasPdb
import pandas as pd
import os
from typing import List, Dict, Tuple, Union, Optional
import subprocess
import shutil


from dms_3d_features.logger import get_logger

log = get_logger("pdb-features")

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
            "pdb_r_pos": resi_number[i],
            "sasa": sasa,
        }
        all_data.append(data)
    df = pd.DataFrame(all_data)
    return df


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

    Example:
        >>> pdb_dir = '/path/to/pdb/files'
        >>> probe_radius = 2.0
        >>> df = compute_solvent_accessibility_all(pdb_dir, probe_radius)
    """
    pdb_paths = glob.glob(f"{pdb_dir}/*/*.pdb")
    dfs = []
    for pdb_path in pdb_paths:
        df = compute_solvent_accessibility(pdb_path, probe_radius)
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


def extract_hbond_length(pdb_path: str) -> pd.DataFrame:
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
    generate_hbond_output_file_from_dssr(pdb_path)
    txt_file = os.path.basename(pdb_path)
    return load_hbonds_file(f"data/dssr-output/{txt_file[:-4]}_hbonds.txt")


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
    pdb_files = glob.glob(f"{pdb_dir}/*/*.pdb")

    for pdb in pdb_files:
        motif = pdb.split("/")[-2]
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
        ppdb = PandasPdb().read_pdb(pdb)
        ATOM = ppdb.df["ATOM"]

        for _, row in df_fn.iterrows():
            if not (row["distance"] < 3.3 and row["type"] == "p"):
                continue

            pos_1 = row["atom_1"].split("@")[1]
            pos_2 = row["atom_2"].split("@")[1]
            num_1 = int(pos_1[1:])
            num_2 = int(pos_2[1:])
            ps_1 = row["atom_1"].split("@")[0]
            ps_2 = row["atom_2"].split("@")[0]

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
                print(data)
                all_data.append(data)
                break

    return pd.DataFrame(all_data)


# dssr features #####################################################################


def calculate_structural_parameters_with_dssr(pdb_dir: str) -> None:
    """
    Calculate structural parameters for PDB structures using DSSR.

    This function processes all PDB files in the specified directory and
    subdirectories. For each PDB file, it creates a corresponding directory,
    moves the PDB file into this directory, and runs DSSR to calculate
    structural parameters. The results are stored in output files named after
    the original PDB file, with an additional extension '.out'. All output
    files are moved to 'data/dssr-output'.

    Args:
        pdb_dir (str): The directory containing the PDB files.

    Returns:
        None
    """
    output_base_dir = os.path.join("data", "dssr-output")
    os.makedirs(output_base_dir, exist_ok=True)
    pdb_paths = glob.glob(f"{pdb_dir}/*/*.pdb")
    log.info(f"Found {len(pdb_paths)} PDB files in {pdb_dir}")
    for pdb_path in pdb_paths:
        log.info(f"running dssr on: {pdb_path}")
        pdb_filename = os.path.basename(pdb_path)
        subprocess.call(
            [
                "x3dna-dssr",
                f"-i={pdb_path}",
                f"-o={output_base_dir}/{pdb_filename[:-4]}.out",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        torsion_file_src = "dssr-torsions.txt"
        torsion_file_dst = os.path.join(
            output_base_dir, f"{pdb_filename[:-4]}_torsions.txt"
        )
        shutil.move(torsion_file_src, torsion_file_dst)
        for temp_file in glob.glob(f"dssr-*"):
            os.remove(temp_file)


class DSSRTorsionFileProcessor:
    """
    A class to process DSSR torsion files and extract structural data.

    Attributes:
        output_dir (str): The directory to store the output files. Defaults to
                          'dssr-output' in the given pdb_dir if not provided.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the DSSRTorsionFileProcessor with specified directories.
        """
        self.output_dir = (
            output_dir if output_dir else os.path.join("data", "dssr-output")
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def process_file(
        self, dssr_torsion_file: str, name: str, motif: str
    ) -> pd.DataFrame:
        """
        Process a DSSR torsion file to extract structural data.

        Args:
            dssr_torsion_file (str): The path to the DSSR torsion file.
            name (str): The name identifier for the data.
            motif (str): The motif identifier for the data.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted structural data.
        """
        lines = self.__read_file_lines(dssr_torsion_file)
        start_indices = self.__find_start_indices(lines)

        column_headers = self.__get_column_headers()
        col_ranges = self.__get_column_ranges()

        all_data = []
        for i in range(len(motif) - 1):
            row_data = self.__extract_row_data(lines, col_ranges, start_indices, i)
            data = [name, motif] + row_data
            all_data.append(data)

        return pd.DataFrame(all_data, columns=column_headers)

    # Helper functions ##############################################################
    def __extract_substrings(
        self, lines: List[str], indices: Dict[str, Tuple[int, int]], offset: int
    ) -> List[str]:
        """
        Extract substrings from lines based on given indices.

        Args:
            lines (List[str]): The list of lines from the file.
            indices (Dict[str, Tuple[int, int]]): A dictionary mapping keys to
                tuples indicating the start and end indices for substring extraction.
            offset (int): The offset line index for data extraction.

        Returns:
            List[str]: Extracted substrings from the specified line.
        """
        data = []
        for _, (col_start, col_end) in indices.items():
            if offset < len(lines):
                line = lines[offset].strip()
                data.append(line[col_start:col_end].strip())
            else:
                data.append("")  # Append an empty string if offset is out of range
        return data

    def __read_file_lines(self, file_path: str) -> List[str]:
        """
        Read all lines from a file.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            List[str]: A list of lines read from the file.
        """
        try:
            with open(file_path, "r") as f:
                return f.readlines()
        except FileNotFoundError as e:
            log.error(f"File not found: {file_path}")
            return []

    def __find_start_indices(self, lines: List[str]) -> Dict[str, int]:
        """
        Find the starting indices for various data sections in the lines.

        Args:
            lines (List[str]): The list of lines from the file.

        Returns:
            Dict[str, int]: A dictionary mapping section names to their start indices.
        """
        start_indices = {"alpha": -1, "eta": -1, "puckering": -1, "bin": -1}
        for idx, line in enumerate(lines):
            if "nt" in line:
                if "alpha" in line:
                    start_indices["alpha"] = idx + 1
                elif "eta" in line:
                    start_indices["eta"] = idx + 1
                elif "v0" in line:
                    start_indices["puckering"] = idx + 1
                elif "bin" in line:
                    start_indices["bin"] = idx + 1
        if all(v == -1 for v in start_indices.values()):
            log.warning("Not all data sections found in the file.")
        return start_indices

    def __get_column_headers(self) -> List[str]:
        """
        Get the column headers for the DataFrame.

        Returns:
            List[str]: A list of column headers.
        """
        return [
            "pdb_name",
            "m_sequence",
            "r_loc_pos",
            "r_nuc",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "e-z",
            "chi",
            "phase-angle",
            "sugar-type",
            "sszp",
            "dp",
            "splay",
            "eta",
            "theta",
            "eta_1",
            "theta_1",
            "eta_2",
            "theta_2",
            "v0",
            "v1",
            "v2",
            "v3",
            "v4",
            "tm",
            "p",
            "puckering",
            "bin",
            "cluster",
            "suitness",
        ]

    def __get_column_ranges(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """
        Get the column ranges for data extraction.

        Returns:
            Dict[str, Dict[str, Tuple[int, int]]]: A dictionary mapping section
            names to their corresponding column range mappings.
        """
        return {
            "alpha": {
                "nt": (8, 9),
                "alpha": (26, 32),
                "beta": (33, 40),
                "gamma": (42, 48),
                "delta": (51, 56),
                "epsilon": (57, 65),
                "zeta": (66, 73),
                "e_z": (75, 83),
                "chi": (85, 98),
                "phase": (101, 116),
                "sugar": (117, 127),
                "ssZp": (130, 135),
                "Dp": (138, 143),
                "splay": (145, 151),
            },
            "eta": {
                "eta": (25, 33),
                "theta": (33, 41),
                "eta_1": (41, 48),
                "theta_1": (49, 56),
                "eta_2": (57, 64),
                "theta_2": (65, 73),
            },
            "puckering": {
                "v0": (27, 33),
                "v1": (34, 41),
                "v2": (43, 49),
                "v3": (50, 57),
                "v4": (59, 64),
                "tm": (67, 72),
                "P": (75, 80),
                "Puck": (81, 90),
            },
            "bin": {"bin": (24, 31), "clus": (33, 38), "suit": (42, 53)},
        }

    def __extract_row_data(
        self,
        lines: List[str],
        col_ranges: Dict[str, Dict[str, Tuple[int, int]]],
        start_indices: Dict[str, int],
        offset: int,
    ) -> List[str]:
        """
        Extract row data for a specific offset.

        Args:
            lines (List[str]): The list of lines from the file.
            col_ranges (Dict[str, Dict[str, Tuple[int, int]]]): The column range mappings.
            start_indices (Dict[str, int]): The starting indices for each section.
            offset (int): The row offset for data extraction.

        Returns:
            List[str]: Extracted data for the specified row.
        """
        data_alpha = self.__extract_substrings(
            lines, col_ranges["alpha"], start_indices["alpha"] + offset
        )
        data_eta = self.__extract_substrings(
            lines, col_ranges["eta"], start_indices["eta"] + offset
        )
        data_puckering = self.__extract_substrings(
            lines, col_ranges["puckering"], start_indices["puckering"] + offset
        )
        data_bin = self.__extract_substrings(
            lines, col_ranges["bin"], start_indices["bin"] + offset
        )
        nt_num = self.__extract_substrings(
            lines, {"nt_num": (9, 11)}, start_indices["alpha"] + offset
        )
        return nt_num + data_alpha + data_eta + data_puckering + data_bin


def get_all_torsional_parameters_from_dssr(pdb_dir: str) -> pd.DataFrame:
    """
    Extract all torsional parameters from DSSR files in the given directory.

    This function iterates through all PDB files in the specified directory and
    processes the corresponding DSSR torsion files to extract structural parameters.

    Args:
        pdb_dir (str): The directory containing the PDB files.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing torsional parameters
                      for all processed PDB files.
    """
    pdb_paths = sorted(glob.glob(f"{pdb_dir}/*/*.pdb"))
    dfs = []
    for pdb_path in pdb_paths:
        base_name = os.path.basename(pdb_path)[:-4]
        motif = os.path.basename(os.path.dirname(pdb_path)).replace("_", "&")
        log.info(f"Processing file: data/dssr-output/{base_name}_torsions.txt")

        torsion_file_path = f"data/dssr-output/{base_name}_torsions.txt"
        if os.path.exists(torsion_file_path):
            processor = DSSRTorsionFileProcessor()
            df = processor.process_file(torsion_file_path, base_name, motif)
            dfs.append(df)
        else:
            log.warning(f"Torsion file not found: {torsion_file_path}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        log.error("No data was processed. Check for missing files or errors.")
        return pd.DataFrame()  # Return an empty DataFrame if no data was processed
