import glob
import os
import itertools
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp, pearsonr, linregress, zscore


from seq_tools import SequenceStructure, fold, to_rna, has_5p_sequence
from seq_tools.structure import find as seq_ss_find
from rna_secstruct import SecStruct

from rna_map.mutation_histogram import (
    get_mut_histos_from_pickle_file,
    get_dataframe,
    convert_dreem_mut_histos_to_mutation_histogram,
)

from dms_3d_features.logger import get_logger, setup_logging
from dms_3d_features.plotting import plot_pop_avg_from_row, colors_for_sequence

# assume data/ is location of data
# assume data/mutation-histograms/ is location of mutation histograms

RESOURCES_PATH = "dms_3d_features/resources"
DATA_PATH = "data"

log = get_logger("process-motifs")

# helper functions ##################################################################


def r2(x, y):
    return pearsonr(x, y)[0] ** 2


def normalize(column):
    return (column - column.min()) / (column.max() - column.min())


def trim(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Trims the 'sequence', 'structure', and 'data' columns of the DataFrame to the given start and end indices.

    Args:
        df (pd.DataFrame): A DataFrame with 'sequence', 'structure', and 'data' columns, where 'data' contains lists of numbers.
        start (int): The start index for trimming.
        end (int): The end index for trimming.

    Returns:
        pd.DataFrame: A trimmed DataFrame with the 'sequence', 'structure', and 'data' columns adjusted to the specified indices.
    """

    def trim_column(column, start, end):
        if start == 0 and end == 0:
            return column
        if end == 0:
            return column.str[start:]
        elif start == 0:
            return column.str[:-end]
        else:
            return column.str[start:-end]

    df = df.copy()
    for col in ["sequence", "structure", "data"]:
        if col in df.columns:
            if col == "data":
                df[col] = df[col].apply(
                    lambda x: x[start:-end] if end != 0 else x[start:]
                )
            else:
                df[col] = trim_column(df[col], start, end)

    return df


def trim_p5_and_p3(df: pd.DataFrame, is_rna=True) -> pd.DataFrame:
    """
    Trims the 5' and 3' ends of the data in the DataFrame.

    This function reads a CSV file containing p5 sequences, converts these sequences to RNA,
    checks for a common p5 sequence in the given DataFrame, and trims the DataFrame based on
    the length of this common p5 sequence and a fixed 3' end length.

    Args:
        df (pd.DataFrame): A DataFrame with a 'data' column containing sequences as strings.

    Returns:
        pd.DataFrame: A trimmed DataFrame with the 5' and 3' ends trimmed.

    Raises:
        ValueError: If no common p5 sequence is found or the sequence is not registered in the CSV file.

    """
    df_p5 = pd.read_csv(f"{RESOURCES_PATH}/csvs/p5_sequences.csv")
    if is_rna:
        df_p5 = to_rna(df_p5)
    common_p5_seq = ""
    for p5_seq in df_p5["sequence"]:
        if has_5p_sequence(df, p5_seq):
            common_p5_seq = p5_seq
    if len(common_p5_seq) == 0:
        raise ValueError("No common p5 sequence found")
    log.debug(f"common p5 sequence: {common_p5_seq}")
    return trim(df, len(common_p5_seq), 20)


def split_dataframe(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    """
    Split a DataFrame into multiple chunks.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        n (int): The number of chunks to split the DataFrame into.

    Returns:
        List[pd.DataFrame]: A list of DataFrame chunks.

    Raises:
        None.

    """
    chunk_size = len(df) // n
    chunks = [df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]
    if len(df) % n != 0:  # Handle any remaining rows
        chunks.append(df.iloc[n * chunk_size :])
    return chunks


def flip_structure(structure: str) -> str:
    """
    Flips the structure of a sequence and inverts ( to ) and vice versa.

    Args:
        structure (str): The input structure to be flipped.

    Returns:
        str: The flipped structure.

    """
    new_structure = ""
    for e in structure[::-1]:
        if e == "(":
            new_structure += ")"
        elif e == ")":
            new_structure += "("
        else:
            new_structure += e
    return new_structure


def flip_pair(bp_name: str) -> str:
    """
    Flips a base pair name.

    Args:
        bp_name (str): The base pair name to be flipped.

    Returns:
        str: The flipped base pair name.

    """
    return bp_name[::-1]


def get_resi_pos(seq: str, pos: int) -> int:
    """
    Calculate the residue position starting from 0.

    Args:
        seq (str): The sequence string.
        pos (int): The original position (starting from 3).

    Returns:
        int: The adjusted position.
    """
    strand_len = seq.split("_")
    if pos > (len(strand_len[0]) + 2):
        new_pos = pos - 6
        return new_pos
    else:
        new_pos = pos - 3
        return new_pos


# random funcs ######################################################################


def find_max_nomod_mutations():
    df = pd.read_json("data/raw-jsons/pdb_library_nomod.json")
    absolute_m = 0
    for i, row in df.iterrows():
        m = max(row["data"])
        if m > absolute_m:
            absolute_m = m
    print(absolute_m)


# processing steps ##################################################################


# step 1: convert raw pickled mutation histograms to dataframe json files ##########
def process_mutation_histograms_to_json():
    """
    Processes mutation histograms from pickle files, converts them to DataFrames,
    and saves the results as JSON files.

    This function performs the following steps:
    1. Reads mutation histograms from pickle files
    2. Converts them to a standardized format
    3. Creates a DataFrame with relevant columns
    4. Processes the data (RNA conversion, folding, trimming)
    5. Saves the results as JSON files

    Returns:
        None
    """
    pickle_files = glob.glob("data/mutation-histograms/*.p")
    cols = [
        "name",
        "sequence",
        "structure",
        "pop_avg",
        "sn",
        "num_reads",
        "num_aligned",
        "no_mut",
        "1_mut",
        "2_mut",
        "3_mut",
        "3plus_mut",
    ]
    n_workers = 10

    for pfile in pickle_files:
        name = os.path.splitext(os.path.basename(pfile))[0]
        output_file = f"data/raw-jsons/constructs/{name}.json"

        if os.path.isfile(output_file):
            log.info(f"Skipping {name}: Output file already exists")
            continue

        log.info(f"Processing {name}")

        mut_histos = get_mut_histos_from_pickle_file(pfile)
        mut_histos = convert_dreem_mut_histos_to_mutation_histogram(mut_histos)
        df_results = get_dataframe(mut_histos, cols)
        df_results = df_results.rename(columns={"pop_avg": "data"})
        df_results = to_rna(df_results)

        df_chunks = split_dataframe(df_results, n_workers)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(fold, df_chunks))

        final_result = pd.concat(results)
        final_result = trim_p5_and_p3(final_result)
        final_result.to_json(output_file, orient="records")

    log.info("Mutation histogram processing and JSON conversion completed successfully")


# step 2: generate motif dataframes ################################################
class GenerateMotifDataFrame:
    """
    A class used to generate raw data from constructs and group it into motifs
    """

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the entire pipeline to filter, create, standardize, and compute average motif data.

        Args:
            df (pd.DataFrame): The input dataframe containing sequence and structure data.

        Returns:
            pd.DataFrame: The processed dataframe with average motif data.
        """
        # initial filtering
        df = df.query("num_aligned > 2000 and sn > 4.0")
        # step 2A: create initial motif dataframe each motif is its own row
        df_motif = self.__create_initial_motif_dataframe(df)
        # step 2B: standarize motifs so they are all in the same orientation
        df_motif = self.__standardize_motif_dataframe(df_motif)
        # step 2C: get the average motif data for each motif
        df_motif_avg = self.__get_avg_motif_dataframe(df_motif)
        return df_motif_avg

    # step 2A ##################################################################
    def __create_initial_motif_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the initial motif dataframe from the input dataframe.

        Args:
            df (pd.DataFrame): The input dataframe containing sequence and structure data.

        Returns:
            pd.DataFrame: The initial motif dataframe.
        """
        all_data = []
        for _, row in df.iterrows():
            ss = SecStruct(row["sequence"], row["structure"])
            for j, m in enumerate(ss.get_junctions()):
                all_data.append(self.__get_motif_data(m, row, j))
        df_motif = pd.DataFrame(all_data)
        df_motif.to_json(
            f"{DATA_PATH}/raw-jsons/motifs/pdb_library_1_motifs.json", orient="records"
        )
        return df_motif

    def __get_motif_data(self, m, row, m_pos) -> Dict[str, Any]:
        """
        Get the motif data for a given motif and construct row.

        Args:
            m (Motif): The motif object.
            row (pd.Series): The row containing the sequence, data, and name.
            m_pos (int): The position of the motif.

        Returns:
            Dict[str, Any]: A dictionary containing the motif data.
        """
        strands = m.strands
        # TODO rename these to make sense
        first_bp = [strands[0][0] - 1, strands[1][-1] + 1]
        second_bp = [strands[0][-1] + 1, strands[1][0] - 1]
        first_bp_id = row["sequence"][first_bp[0]] + row["sequence"][first_bp[1]]
        second_bp_id = row["sequence"][second_bp[0]] + row["sequence"][second_bp[1]]
        flank_bp_5p = row["sequence"][strands[0][0]] + row["sequence"][strands[1][-1]]
        flank_bp_3p = row["sequence"][strands[0][-1]] + row["sequence"][strands[1][0]]
        m_data = []
        for i, strand in enumerate(strands):
            for pos in strand:
                m_data.append(round(row["data"][pos], 6))
            if i == 0 and len(strands) == 2:
                m_data.append(0)
        m_strands = m.strands[0] + [-1] + m.strands[1]
        seqs = m.sequence.split("&")
        token = str(len(seqs[0]) - 2) + "x" + str(len(seqs[1]) - 2)
        data = {
            "construct": row["name"],  # name of the construct
            "m_data": m_data,  # reactivity for the motif
            "m_pos": m_pos,  # position of the motif 0 to 7?
            "m_sequence": m.sequence,  # sequence of the motif
            "m_structure": m.structure,  # structure of the motif
            "m_strands": m_strands,  # positions of each nuclotide of the motif
            "m_token": token,  # token for the motif such as 1x1, 2x2, etc
            "m_flank_bp_5p": flank_bp_5p,  # base pair at the 5' end of the motif
            "m_flank_bp_3p": flank_bp_3p,  # base pair at the 3' end of the motif
            "m_second_flank_bp_5p": first_bp_id,  # first base pair of the motif before flanking
            "m_second_flank_bp_3p": second_bp_id,  # second base pair of the motif after flanking
            "num_aligned": row["num_aligned"],  # number of aligned reads
            "sn": row["sn"],  # signal to noise ratio
        }
        return data

    # step 2B ##################################################################
    def __standardize_motif_dataframe(self, df_motif):
        df_motif["strand1"] = df_motif["m_sequence"].apply(lambda x: x.split("&")[0])
        df_motif["strand2"] = df_motif["m_sequence"].apply(lambda x: x.split("&")[1])
        df_motif["m_orientation"] = "non-flipped"
        for i, row in df_motif.iterrows():
            if len(row["strand2"]) < len(row["strand1"]):
                continue
            if len(row["strand1"]) == len(row["strand2"]):
                if row["strand1"] < row["strand2"]:
                    continue
            len_s1 = len(row["strand1"])
            flipped_strands = (
                row["m_strands"][len_s1 + 1 :] + [-1] + row["m_strands"][:len_s1]
            )
            flipped_data = row["m_data"][len_s1 + 1 :] + [0] + row["m_data"][:len_s1]
            df_motif.at[i, "m_sequence"] = row["strand2"] + "&" + row["strand1"]
            df_motif.at[i, "m_structure"] = flip_structure(row["m_structure"])
            df_motif.at[i, "m_strands"] = flipped_strands
            df_motif.at[i, "m_token"] = row["m_token"][::-1]
            df_motif.at[i, "m_data"] = flipped_data
            p3_bp = flip_pair(row["m_flank_bp_3p"])
            df_motif.at[i, "m_flank_bp_3p"] = flip_pair(row["m_flank_bp_5p"])
            df_motif.at[i, "m_flank_bp_5p"] = p3_bp
            p3_bp = flip_pair(row["m_second_flank_bp_3p"])
            df_motif.at[i, "m_second_flank_bp_3p"] = flip_pair(
                row["m_second_flank_bp_5p"]
            )
            df_motif.at[i, "m_second_flank_bp_5p"] = p3_bp
            df_motif.at[i, "m_orientation"] = "flipped"
        df_motif.drop(["strand1", "strand2"], axis=1, inplace=True)
        df_motif.to_json(
            f"{DATA_PATH}/raw-jsons/motifs/pdb_library_1_motifs_standard.json",
            orient="records",
        )
        return df_motif

    # step 2C ##################################################################
    def __get_pdb_path(self, m_sequence: str) -> List[str]:
        """
        Get the paths of PDB files for a given motif sequence.

        Args:
            m_sequence (str): The motif sequence.

        Returns:
            List[str]: A list of paths to PDB files.

        Raises:
            ValueError: If no PDB files are found for the given motif sequence.

        """
        motif_seq_path = m_sequence.replace("&", "_")
        path = f"{DATA_PATH}/pdbs"
        all_pdbs = []
        if os.path.exists(f"{path}/{motif_seq_path}"):
            pdbs = glob.glob(f"{path}/{motif_seq_path}/*.pdb")
            if len(pdbs) == 0:
                raise ValueError(f"No pdbs found for {motif_seq_path}")
            all_pdbs.extend(pdbs)
        rev_motif_seq_path = motif_seq_path[1] + "_" + motif_seq_path[0]
        if os.path.exists(f"{path}/{rev_motif_seq_path}"):
            pdbs = glob.glob(f"{path}/{rev_motif_seq_path}/*.pdb")
            if len(pdbs) == 0:
                raise ValueError(f"No pdbs found for {rev_motif_seq_path}")
            all_pdbs.extend(pdbs)
        return all_pdbs

    def __get_likely_pairs_for_symmetric_junction(self, m_sequence: str) -> List[str]:
        """
        Get the likely base pairs for a symmetric junction in the motif sequence.

        Args:
            m_sequence (str): The motif sequence.

        Returns:
            List[str]: A list of likely base pairs for the symmetric junction.
        """
        seqs = m_sequence.split("&")
        pairs, pairs_1, pairs_2 = [], [], []
        if len(seqs[0]) == len(seqs[1]):
            for n1, n2 in zip(seqs[0], seqs[1][::-1]):
                pairs_1.append(n1 + n2)
            pairs_1.append("")
            for n1, n2 in zip(seqs[0], seqs[1][::-1]):
                pairs_2.append(n2 + n1)
            pairs = pairs_1 + pairs_2[::-1]
        else:
            pairs = [""] * len(m_sequence)
        return pairs

    def __get_avg_motif_dataframe(self, df_motif: pd.DataFrame) -> pd.DataFrame:
        """
        Get the average motif data for each motif in the dataframe.

        Args:
            df_motif (pd.DataFrame): The standardized motif dataframe.

        Returns:
            pd.DataFrame: The dataframe with average motif data.
        """
        all_data = []
        for motif_seq, g in df_motif.groupby("m_sequence"):
            pdb_paths = self.__get_pdb_path(motif_seq)
            m_data_array = np.array(g["m_data"].tolist())
            m_data_avg = np.mean(m_data_array, axis=0)
            m_data_std = np.std(m_data_array, axis=0)
            m_data_cv = m_data_std / m_data_avg
            m_data_cv[np.isnan(m_data_cv)] = 0
            pairs = self.__get_likely_pairs_for_symmetric_junction(motif_seq)
            data = {
                "constructs": g["construct"].tolist(),
                "has_pdbs": len(pdb_paths) > 0,
                "m_data_array": m_data_array,
                "m_data_avg": m_data_avg,
                "m_data_cv": m_data_cv,
                "m_data_std": m_data_std,
                "m_flank_bp_5p": g["m_flank_bp_5p"].tolist(),
                "m_flank_bp_3p": g["m_flank_bp_3p"].tolist(),
                "m_orientation": g["m_orientation"].tolist(),
                "m_pos": g["m_pos"].tolist(),
                "m_second_flank_bp_5p": g["m_second_flank_bp_5p"].tolist(),
                "m_second_flank_bp_3p": g["m_second_flank_bp_3p"].tolist(),
                "m_sequence": motif_seq,
                "m_strands": g["m_strands"].tolist(),
                "m_structure": g["m_structure"].iloc[0],
                "m_token": g["m_token"].iloc[0],
                "pairs": pairs,
                "pdbs": pdb_paths,
            }
            all_data.append(data)
        df = pd.DataFrame(all_data)
        df.to_json(
            f"{DATA_PATH}/raw-jsons/motifs/pdb_library_1_motifs_avg.json",
            orient="records",
        )


# step 3: generate residue dataframes ##############################################
class GenerateResidueDataFrame:
    def run(self, df_motif):
        df_residues_avg = self.__generate_avg_residue_dataframe(df_motif)
        all_data = []
        for _, row in df_residues_avg.iterrows():
            for i in range(len(row["r_data"])):
                data = {
                    "both_purine": row["both_purine"],
                    "both_pyrimidine": row["both_pyrimidine"],
                    "constructs": row["constructs"][i],
                    "has_pdbs": row["has_pdbs"],
                    "likely_pair": row["likely_pair"],
                    "m_flank_bp_5p": row["m_flank_bp_5p"][i],
                    "m_flank_bp_3p": row["m_flank_bp_3p"][i],
                    "m_orientation": row["m_orientation"][i],
                    "m_pos": row["m_pos"][i],
                    "m_second_flank_bp_5p": row["m_second_flank_bp_5p"][i],
                    "m_second_flank_bp_3p": row["m_second_flank_bp_3p"][i],
                    "m_sequence": row["m_sequence"],
                    "m_structure": row["m_structure"],
                    "m_token": row["m_token"],
                    "n_pdbs": row["n_pdbs"],
                    "pair_type": None,
                    "p5_res": row["p5_res"],
                    "p5_type": row["p5_type"],
                    "p3_res": row["p3_res"],
                    "p3_type": row["p3_type"],
                    "r_data": row["r_data"][i],
                    "r_nuc": row["r_nuc"],
                    "r_loc_pos": row["r_loc_pos"],
                    "r_pos": row["r_pos"][i],
                    "r_type": row["r_type"],
                    "pdb_path": row["pdb_path"],
                    "pdb_r_pos": row["pdb_r_pos"],
                }
                all_data.append(data)
        df_residues = pd.DataFrame(all_data)
        df_residues["ln_r_data"] = np.log(df_residues["r_data"])
        df_residues["ln_r_data"].replace(-np.inf, -9.8, inplace=True)
        df_residues.to_json(
            f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues.json",
            orient="records",
        )

    def __generate_avg_residue_dataframe(self, df_motif):
        all_data = []
        for _, row in df_motif.iterrows():
            m_sequence = row["m_sequence"]
            m_structure = row["m_structure"]
            for i, (e, s) in enumerate(zip(m_sequence, m_structure)):
                key = (m_sequence, e, i)
                if e != "A" and e != "C":
                    continue
                r_type = "NON-WC"
                if s == "(" or s == ")":
                    r_type = "WC"
                p5_res = None
                p3_res = None
                p5_type = None
                p3_type = None
                both_purine = None
                both_pyrimidine = None
                if i != 0 and m_sequence[i - 1] != "_":
                    p5_res = m_sequence[i - 1]
                if i != len(m_sequence) - 1 and m_sequence[i + 1] != "_":
                    p3_res = m_sequence[i + 1]
                if p5_res == "A" or p5_res == "G":
                    p5_type = "PURINE"
                else:
                    p5_type = "PYRIMIDINE"
                if p3_res == "A" or p3_res == "G":
                    p3_type = "PURINE"
                else:
                    p3_type = "PYRIMIDINE"
                if p5_type == "PURINE" and p3_type == "PURINE":
                    both_purine = True
                else:
                    both_purine = False
                if p5_type == "PYRIMIDINE" and p3_type == "PYRIMIDINE":
                    both_pyrimidine = True
                else:
                    both_pyrimidine = False
                pdb_r_pos = i + 3
                break_pos = m_sequence.find("&")
                if break_pos < i:
                    pdb_r_pos += 3  # 2 for each of the 2 residues of each strand for the extra 2 basepairs minus 1 for "&"
                data = {
                    "both_purine": both_purine,
                    "both_pyrimidine": both_pyrimidine,
                    "constructs": row["constructs"],
                    "has_pdbs": len(row["pdbs"]) > 0,
                    "likely_pair": row["pairs"][i],
                    "m_flank_bp_5p": row["m_flank_bp_5p"],
                    "m_flank_bp_3p": row["m_flank_bp_3p"],
                    "m_orientation": row["m_orientation"],
                    "m_pos": row["m_pos"],
                    "m_second_flank_bp_5p": row["m_second_flank_bp_5p"],
                    "m_second_flank_bp_3p": row["m_second_flank_bp_3p"],
                    "m_sequence": m_sequence,
                    "m_structure": m_structure,
                    "m_token": row["m_token"],
                    "n_pdbs": len(row["pdbs"]),
                    "pair_type": None,
                    "p5_res": p5_res,
                    "p5_type": p5_type,
                    "p3_res": p3_res,
                    "p3_type": p3_type,
                    "r_avg": row["m_data_avg"][i],
                    "r_cv": row["m_data_cv"][i],
                    "r_data": np.array(row["m_data_array"])[:, i].tolist(),
                    "r_nuc": e,
                    "r_loc_pos": i,
                    "r_pos": np.array(row["m_strands"])[:, i].tolist(),
                    "r_std": row["m_data_std"][i],
                    "r_type": r_type,
                    "pdb_path": row["pdbs"],
                    "pdb_r_pos": pdb_r_pos,
                }
                all_data.append(data)
        df_residues = pd.DataFrame(all_data)
        df_residues.to_json(
            f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues_avg.json",
            orient="records",
        )
        return df_residues


# step 4: merge pdb info into motif and residue dataframes ##########################
def generate_pdb_residue_dataframe(df_residue):
    df_pairs = pd.read_csv(
        f"dms_3d_features/resources/csvs/basepair_data_for_motifs.csv"
    )
    df_res = pd.read_csv(f"dms_3d_features/resources/csvs/pdb_res.csv")
    df_res.drop(["m_sequence"], axis=1, inplace=True)
    df_res["pdb_name"] = [x + ".pdb" for x in df_res["pdb_name"]]
    df_paths = []
    for i, row in df_pairs.iterrows():
        try:
            path = glob.glob(f"data/pdbs/*/{row['pdb_name']}")[0]
        except:
            log.info(f"no pdb found for {row['pdb_name']}")
            path = ""
        df_paths.append(path)
    df_pairs["pdb_path"] = df_paths
    df_pairs = df_pairs.merge(df_res, on="pdb_name")
    df_pairs.to_csv("data/pdb-features/pairs.csv", index=False)
    df_residue = df_residue.query("has_pdbs == True").copy()
    df_residue["m_sequence"] = df_residue["m_sequence"].apply(
        lambda x: x.replace("&", "_")
    )
    df_residue.drop(["has_pdbs", "pdb_path", "r_nuc", "r_type"], axis=1, inplace=True)
    df_final = df_pairs.merge(df_residue, on=["m_sequence", "pdb_r_pos"], how="left")
    return df_final


def mark_outliers(df_residues):
    pass


def generate_stats(df):
    all_data = []
    for [m_sequence, r_loc_pos], g in df.groupby(["m_sequence", "r_loc_pos"]):
        g_sort = g.sort_values("r_pos", ascending=True)
        r = linregress(g["r_pos"], normalize(g["r_data"]))
        if len(g) > 20:
            min_vals = g_sort["r_data"].values[0:10]
            max_vals = g_sort["r_data"].values[-10:]
            ks_2samp_stat, ks_2samp_p_value = ks_2samp(min_vals, max_vals)
        else:
            ks_2samp_stat, ks_2samp_p_value = -1, -1
        data = {
            "r_nuc": g["r_nuc"].iloc[0],
            "r_type": g["r_type"].iloc[0],
            "pairs": g["likely_pair"].iloc[0],
            "m_sequence": m_sequence,
            "m_token": g["m_token"].iloc[0],
            "r_loc_pos": r_loc_pos,
            "slope": r.slope,
            "intercept": r.intercept,
            "r2": r.rvalue**2,
            "p_val ": r.pvalue,
            "ks_stat": ks_2samp_stat,
            "ks_p_val": ks_2samp_p_value,
            "count": len(g),
            "r_pos_min": g["r_pos"].min(),
            "r_pos_max": g["r_pos"].max(),
            "r_data_min": g["r_data"].min(),
            "r_data_max": g["r_data"].max(),
            "r_data": g["r_data"].tolist(),
            "r_pos": g["r_pos"].tolist(),
        }
        all_data.append(data)
    df_stats = pd.DataFrame(all_data)
    df_stats.to_json("stats.json", orient="records")


def regen_data():
    df = pd.read_json(f"{DATA_PATH}/raw-jsons/constructs/pdb_library_1_combined.json")
    gen = GenerateMotifDataFrame()
    gen.run(df)
    df = pd.read_json(f"{DATA_PATH}/raw-jsons/motifs/pdb_library_1_motifs_avg.json")
    gen = GenerateResidueDataFrame()
    gen.run(df)
    df = pd.read_json(f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues.json")
    df = generate_pdb_residue_dataframe(df)
    df.to_json(
        f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues_pdb.json",
        orient="records",
    )


def main():
    """
    main function for script
    """
    setup_logging()
    process_mutation_histograms_to_json()
    exit()
    df = pd.read_json(f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues.json")
    df["z_score"] = 0
    df["r_data_outlier"] = False
    data = []
    for i, g in df.groupby(["m_sequence", "r_loc_pos"]):
        g["z_score"] = zscore(g["r_data"])
        g["r_data_outlier"] = g["z_score"].abs() > 3
        data.append(g)
    df = pd.concat(data)
    df_outlier = df.query("r_data_outlier == True")
    print(len(df_outlier))
    exit()
    regen_data()
    exit()


if __name__ == "__main__":
    main()
