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
        ppdb = PandasPdb.PandasPdb().read_pdb(pdb_path)
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
                f"atom, (name {row['atom_name']}) and (resn {row['residue_name']}) and (resi {row['residue_number']})"
            ),
            structure,
            result,
        )

        data = {
            "pdb_path": pdb_path,
            "m_sequence": m_sequence,
            "r_nuc": row["residue_name"],
            "pdb_r_pos": row["residue_number"],
            "sasa": selection["atom"],
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


# hydrogen bonds #####################################################################


def calculate_hbond_strength_all(pdb_dir: str) -> pd.DataFrame:
    """
    Main function to calculate hydrogen bond strength for all PDB files in a directory.

    Args:
        pdb_dir (str): Directory containing PDB files.

    Returns:
        pd.DataFrame: DataFrame containing hydrogen bond strength data.
    """
    pdb_files = glob.glob(f"{pdb_dir}/*/*.pdb")
    all_data = []
    hbond_calculator = HbondCalculator()
    for pdb in pdb_files:
        df = hbond_calculator.calculate_hbond_strength(pdb)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


class HbondCalculator:
    def __init__(self):
        self.angle_pairs = {
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
        self.strength_factors = {"O:O": 21, "N:N": 13, "N:O": 8}

    def calculate_hbond_strength(self, pdb_path: str) -> pd.DataFrame:
        """
        Calculate hydrogen bond strength for a single PDB file.

        Args:
            pdb_path (str): Path to the PDB file.

        Returns:
            pd.DataFrame: DataFrame containing hydrogen bond strength data.
        """
        motif = os.path.basename(os.path.dirname(pdb_path))
        pdb_name = os.path.basename(pdb_path)
        df_hbonds = self.__extract_hbond_length(pdb_path)
        atom_df = PandasPdb.PandasPdb().read_pdb(pdb_path).df["ATOM"]

        pos_data = self.__get_position_data(motif)
        hbond_data = self.__process_hbonds(df_hbonds, atom_df, pos_data, pdb_name)

        return pd.DataFrame(hbond_data)

    def __get_position_data(self, motif):
        a_pos1, a_pos2, c_pos1, c_pos2 = self.__find_positions_of_a_and_cs(motif)
        pos1 = a_pos1 + a_pos2 + c_pos1 + c_pos2
        a_pos = self.__prepend_char_to_list_elements(
            a_pos1, "A"
        ) + self.__prepend_char_to_list_elements(a_pos2, "A")
        c_pos = self.__prepend_char_to_list_elements(
            c_pos1, "C"
        ) + self.__prepend_char_to_list_elements(c_pos2, "C")
        pos = a_pos + c_pos
        var_holder = {f"n{n}": l for n, l in zip(pos1, pos)}
        var_holder.update(
            {f"a{n}": "N1" if l[0] == "A" else "N3" for n, l in zip(pos1, pos)}
        )

        return {
            "n1": var_holder[f"n{pos1[-1]}"],
            "at1": var_holder[f"a{pos1[-1]}"],
            "var_holder": var_holder,
        }

    def __process_hbonds(self, hbond_df, atom_df, pos_data, pdb_filename):
        def parse_atom_info(atom_string):
            atom_name, residue_number = atom_string.split("@")
            position = residue_number
            residue_index = int(residue_number[1:])
            return atom_name, position, residue_index

        processed_hbonds = []
        for _, hbond in hbond_df.iterrows():
            if hbond["distance"] >= 3.3 or hbond["type"] != "p":
                continue

            atom1_name, atom1_pos, atom1_index = parse_atom_info(hbond["atom_1"])
            atom2_name, atom2_pos, atom2_index = parse_atom_info(hbond["atom_2"])

            if not (atom1_name in self.angle_pairs and atom2_name in self.angle_pairs):
                log.warning(
                    f"Atom pair not found in angle_atom_pair: {hbond['atom_1']} - {hbond['atom_2']}"
                )
                continue

            atom_coordinates = self.__get_atom_coordinates(
                atom_df, atom1_name, atom2_name, atom1_index, atom2_index
            )
            if any(coord.empty for coord in atom_coordinates.values()):
                continue

            angle1, angle2 = self.__calculate_angles(atom_coordinates)
            hbond_strength = self.__calculate_strength(
                hbond["distance"], hbond["hbond_atoms"]
            )

            # Check if either atom_1 or atom_2 matches the target atom (at1)
            is_target_atom_involved = (
                hbond["atom_1"].split("@")[0] == pos_data["at1"]
                or hbond["atom_2"].split("@")[0] == pos_data["at1"]
            )
            # Classify the bond based on target atom involvement
            n_or_other = "N-included" if is_target_atom_involved else "Other"

            processed_hbonds.append(
                {
                    "hbond_length": hbond["distance"],
                    "pdb": pdb_filename,
                    "angle_1": angle1,
                    "angle_2": angle2,
                    "r_pos_1": atom1_index,
                    "r_pos_2": atom2_index,
                    "n_or_other": n_or_other,
                    "hbond_atoms": hbond["hbond_atoms"],
                    "hbond_strength": hbond_strength,
                    "atom_1": atom1_name,
                    "atom_2": atom2_name,
                }
            )
        return processed_hbonds

    def __generate_hbond_output_file_from_dssr(
        self, pdb_path: str, output_file: str
    ) -> str:
        if os.path.exists(output_file):
            log.info(f"Output file already exists: {output_file}. Skipping command.")
            return output_file

        try:
            subprocess.run(
                ["x3dna-dssr", f"-i={pdb_path}", "--get-hbonds", f"-o={output_file}"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running x3dna-dssr: {e.stderr}")

        try:
            for file in glob.glob("dssr-*"):
                os.remove(file)
        except OSError as e:
            log.error(f"Warning: Failed to remove temporary files: {e}")

        if not os.path.exists(output_file):
            raise FileNotFoundError(
                f"Expected output file not generated: {output_file}"
            )

        return output_file

    @staticmethod
    def __load_hbonds_file(file_path: str) -> pd.DataFrame:
        column_names = [
            "position_1",
            "position_2",
            "hbond_num",
            "type",
            "distance",
            "hbond_atoms",
            "atom_1",
            "atom_2",
        ]
        return pd.read_csv(
            file_path, skiprows=2, delim_whitespace=True, names=column_names
        )

    def __extract_hbond_length(self, pdb_path: str) -> pd.DataFrame:
        pdb_filename = os.path.basename(pdb_path)
        output_file = f"data/dssr-output/{pdb_filename[:-4]}_hbonds.txt"
        self.__generate_hbond_output_file_from_dssr(pdb_path, output_file)
        return self.__load_hbonds_file(output_file)

    @staticmethod
    def __prepend_char_to_list_elements(lst: list, char: str) -> list:
        return [f"{char}{elem}" for elem in lst]

    @staticmethod
    def __find_positions_of_a_and_cs(motif: str) -> tuple:
        motif1, motif2 = motif.split("_")
        a_pos1 = [pos + 3 for pos, char in enumerate(motif1) if char == "A"]
        a_pos2 = [
            pos + 7 + len(motif1) for pos, char in enumerate(motif2) if char == "A"
        ]
        c_pos1 = [pos + 3 for pos, char in enumerate(motif1) if char == "C"]
        c_pos2 = [
            pos + 7 + len(motif1) for pos, char in enumerate(motif2) if char == "C"
        ]
        return a_pos1, a_pos2, c_pos1, c_pos2

    def __get_atom_coordinates(self, atom_df, ps_1, ps_2, num_1, num_2):
        return {
            "coords_11": atom_df[
                (atom_df["atom_name"] == ps_1) & (atom_df["residue_number"] == num_1)
            ],
            "coords_12": atom_df[
                (atom_df["atom_name"] == self.angle_pairs[ps_1])
                & (atom_df["residue_number"] == num_1)
            ],
            "coords_21": atom_df[
                (atom_df["atom_name"] == ps_2) & (atom_df["residue_number"] == num_2)
            ],
            "coords_22": atom_df[
                (atom_df["atom_name"] == self.angle_pairs[ps_2])
                & (atom_df["residue_number"] == num_2)
            ],
        }

    @staticmethod
    def __calculate_angles(coords):
        a1, a2, b1, b2 = [
            coord[["x_coord", "y_coord", "z_coord"]].values[0]
            for coord in coords.values()
        ]

        def calculate_angle(v1, v2, v3):
            return np.degrees(
                np.arccos(
                    np.dot(v2 - v1, v3 - v1)
                    / (np.linalg.norm(v2 - v1) * np.linalg.norm(v3 - v1))
                )
            )

        angle_1_degrees = calculate_angle(a1, a2, b1)
        angle_2_degrees = calculate_angle(b1, b2, a1)

        return angle_1_degrees, angle_2_degrees

    def __calculate_strength(self, distance, hbond_atoms):
        return (2.2 / distance) * self.strength_factors.get(hbond_atoms, 8)


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
