# Standard library imports
from concurrent.futures import ThreadPoolExecutor
import glob
import os
from typing import Any, Dict, List, Tuple, Optional

# Third party imports
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, linregress, pearsonr, zscore
from matplotlib import pyplot as plt

# Yesselman lab imports
from rna_map.mutation_histogram import (
    convert_dreem_mut_histos_to_mutation_histogram,
    get_dataframe,
    get_mut_histos_from_pickle_file,
)
from rna_secstruct import SecStruct
from seq_tools import SequenceStructure, fold, has_5p_sequence, to_rna
from seq_tools.structure import find as seq_ss_find

# Local imports
from dms_3d_features.logger import get_logger, setup_logging
from dms_3d_features.paths import DATA_PATH, REVISION_PATH


log = get_logger("process-motifs")

# helper functions ##################################################################


def r2(x, y):
    return pearsonr(x, y)[0] ** 2


def normalize(column):
    return (column - column.min()) / (column.max() - column.min())


def trim(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Trims the 'sequence', 'structure', and 'data' columns of the DataFrame to the
    given start and end indices.

    Args:
        df (pd.DataFrame): A DataFrame with 'sequence', 'structure', and 'data'
            columns, where 'data' contains lists of numbers.
        start (int): The start index for trimming.
        end (int): The end index for trimming.

    Returns:
        pd.DataFrame: A trimmed DataFrame with the 'sequence', 'structure', and
        'data' columns adjusted to the specified indices.
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

    This function reads a CSV file containing p5 sequences, converts these
    sequences to RNA, checks for a common p5 sequence in the given DataFrame,
    and trims the DataFrame based on the length of this common p5 sequence and
    a fixed 3' end length.

    Args:
        df (pd.DataFrame): A DataFrame with a 'data' column containing sequences
            as strings.
        is_rna (bool): Flag indicating if the sequences are RNA. Default is True.

    Returns:
        pd.DataFrame: A trimmed DataFrame with the 5' and 3' ends trimmed.

    Raises:
        ValueError: If no common p5 sequence is found or the sequence is not
            registered in the CSV file.
    """
    df_p5 = pd.read_csv(f"{DATA_PATH}/csvs/p5_sequences.csv")
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
    log.info("Processing mutation histograms")
    if not os.path.isdir(f"{DATA_PATH}/mutation-histograms"):
        raise ValueError(
            f"{DATA_PATH}/mutation-histograms directory does not exist, please download our data from FigShare"
        )
    log.info(
        f"Found {len(glob.glob(DATA_PATH + '/mutation-histograms/*.p'))} pickle files"
    )
    pickle_files = glob.glob(f"{DATA_PATH}/mutation-histograms/*.p")
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
        output_file = f"{DATA_PATH}/raw-jsons/constructs/{name}.json"

        if os.path.isfile(output_file):
            log.info(f"Skipping {name}: Output file already exists")
            continue

        log.info(f"Processing {name}")

        mut_histos = get_mut_histos_from_pickle_file(pfile)
        mut_histos = convert_dreem_mut_histos_to_mutation_histogram(mut_histos)
        df_results = get_dataframe(mut_histos, cols)
        # not sure why this is needed, but it is
        if df_results.iloc[0]["name"].startswith("seq_"):
            df_results["name"] = df_results["name"].str.replace(
                "seq_", "construct", regex=False
            )
        df_results = df_results.rename(columns={"pop_avg": "data"})
        df_results = to_rna(df_results)

        df_chunks = split_dataframe(df_results, n_workers)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(fold, df_chunks))

        final_result = pd.concat(results)
        final_result = trim_p5_and_p3(final_result)
        final_result.to_json(output_file, orient="records")

    log.info("Mutation histogram processing and JSON conversion completed successfully")


def divide_lists(
    row: pd.Series, data_col: str, trim_5prime: int, trim_3prime: int
) -> list[float]:
    """
    Element-wise division of construct data by normalization data in `row[data_col]`.
    If lengths differ, trims normalization array (e.g., extra 5′/3′ regions).
    """
    if "data_construct" not in row or data_col not in row:
        missing = [k for k in ("data_construct", data_col) if k not in row]
        raise KeyError(f"divide_lists: missing columns in row: {missing}")

    a = np.array(row["data_construct"], dtype=float)
    b = np.array(row[data_col], dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            "Expected 1D arrays for data_construct and normalization column."
        )

    if a.shape != b.shape:
        if (trim_5prime + trim_3prime) >= len(b):
            raise ValueError(
                f"Trimming ({trim_5prime}+{trim_3prime}) removes entire normalization array of length {len(b)}."
            )
        b = b[trim_5prime : len(b) - trim_3prime]
        if a.shape != b.shape:
            raise ValueError(
                f"After trimming, shapes still differ: construct={a.shape}, norm={b.shape}. "
                f"Adjust trim_5prime/trim_3prime."
            )

    out = np.full_like(a, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(a, b, out=out, where=(b != 0))
    out[~np.isfinite(out)] = np.nan  # set inf/-inf to NaN
    return out.tolist()


def merge_and_normalize(
    df_construct: pd.DataFrame,
    df_norm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge on 'name' and compute element-wise ratios: data_construct / data{suffix}.
    Example suffixes: '_nomod', '_denature'.
    """
    merged = pd.merge(df_construct, df_norm, on="name", suffixes=("_org", "_norm"))
    for i, row in merged.iterrows():
        if row["sequence_org"] != row["sequence_norm"]:
            print("fail")
    # Divide data_org by data_norm, replace inf/-inf with 0.0, and store in new column data_ratio
    data_org_arr = merged["data_org"].apply(np.array)
    data_norm_arr = merged["data_norm"].apply(np.array)

    def safe_divide(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.true_divide(a, b)
            out[~np.isfinite(out)] = 0.0  # replace inf, -inf, nan with 0.0
        return out.tolist()

    merged["data_ratio"] = [
        safe_divide(a, b) for a, b in zip(data_org_arr, data_norm_arr)
    ]
    return merged


# step 2: generate motif dataframes ################################################
class GenerateMotifDataFrame:
    """
    A class used to generate and process motif data from constructs.
    """

    def run(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Process the input dataframe to generate motif data.

        Args:
            df (pd.DataFrame): Input dataframe with sequence and structure data.

        Returns:
            pd.DataFrame: Processed dataframe with average motif data.
        """
        self.name = name
        log.info(f"Processing {name} with {len(df)} rows")
        df_filtered = df.query("num_aligned > 2000 and sn > 4.0")
        log.info(
            f"removed {len(df) - len(df_filtered)} rows with num_aligned <= 2000 or sn <= 4.0"
        )
        df_motif = self._create_motif_dataframe(df_filtered)
        df_motif.to_json(
            f"{DATA_PATH}/raw-jsons/motifs/{self.name}_motifs.json",
            orient="records",
        )
        df_motif_helix = self.__create_helix_motif_dataframe(df_filtered)
        df_motif.to_json(
            f"{DATA_PATH}/raw-jsons/motifs/{self.name}_helix.json", orient="records"
        )
        dfs = [df_motif, df_motif_helix]
        df_motif_concat = pd.concat(dfs).reset_index(drop=True)
        df_motif_concat.to_json(
            f"{DATA_PATH}/raw-jsons/motifs/{self.name}_motifs_concat.json",
            orient="records",
        )
        df_motif_concat_standardized = self._standardize_motifs(df_motif_concat)
        df_motif_concat_standardized.to_json(
            f"{DATA_PATH}/raw-jsons/motifs/{self.name}_motifs_standard.json",
            orient="records",
        )
        df_motif_avg = self._calculate_average_motif_data(df_motif_concat_standardized)
        df_motif_avg.to_json(
            f"{DATA_PATH}/raw-jsons/motifs/{self.name}_motifs_avg.json",
            orient="records",
        )
        return df_motif_avg

    def run_normalized_df(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Process the input normalized dataframe to generate motif data.

        Args:
            df (pd.DataFrame): Input dataframe with sequence and structure data.

        Returns:
            pd.DataFrame: Processed dataframe with average motif data.
        """
        self.name = name
        log.info(f"Processing {name} with {len(df)} rows")
        df_filtered = df.query(f"num_aligned > 2000 and sn > 0.0")
        log.info(
            f"removed {len(df) - len(df_filtered)} rows with num_aligned <= 2000 or sn <= 0.0"
        )
        df_motif = self._create_motif_dataframe(df_filtered)
        df_motif.to_json(
            f"{REVISION_PATH}/normalized/motifs/{self.name}_motifs.json",
            orient="records",
        )
        df_motif_helix = self.__create_helix_motif_dataframe(df_filtered)
        dfs = [df_motif, df_motif_helix]
        df_motif_concat = pd.concat(dfs).reset_index(drop=True)
        df_motif_concat_standardized = self._standardize_motifs(df_motif_concat)
        df_motif_avg = self._calculate_average_motif_data(df_motif_concat_standardized)
        df_motif_avg.to_json(
            f"{REVISION_PATH}/normalized/motifs/{self.name}_motifs_avg.json",
            orient="records",
        )
        return df_motif_avg

    def run_different_threshold_df(
        self, df: pd.DataFrame, name: str, threshold: int, sn_val: int
    ) -> pd.DataFrame:
        """
        Process the input dataframe to generate motif data.

        Args:
            df (pd.DataFrame): Input dataframe with sequence and structure data.
            threshold : the threshold value for the num_aligned.
            sn_val : threshold value for the signal-to-noise ratio.

        Returns:
            pd.DataFrame: Processed dataframe with average motif data.
        """
        self.name = name
        log.info(f"Processing {name} with {len(df)} rows")
        df_filtered = df.query(f"num_aligned > {threshold} and sn > {sn_val}")
        log.info(
            f"removed {len(df) - len(df_filtered)} rows with num_aligned <= {threshold} or sn <= {sn_val}"
        )
        df_motif = self._create_motif_dataframe(df_filtered)
        df_motif.to_json(
            f"{REVISION_PATH}/dif_threshold/motifs/{self.name}_motifs_{threshold}.json",
            orient="records",
        )
        df_motif_helix = self.__create_helix_motif_dataframe(df_filtered)
        dfs = [df_motif, df_motif_helix]
        df_motif_concat = pd.concat(dfs).reset_index(drop=True)
        df_motif_concat_standardized = self._standardize_motifs(df_motif_concat)
        df_motif_avg = self._calculate_average_motif_data(df_motif_concat_standardized)
        df_motif_avg.to_json(
            f"{REVISION_PATH}/dif_threshold/motifs/{self.name}_motifs_avg_{threshold}.json",
            orient="records",
        )
        return df_motif_avg

    def _create_motif_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the initial motif dataframe from the filtered data."""
        motif_data = []
        for _, row in df.iterrows():
            junctions = SecStruct(row["sequence"], row["structure"]).get_junctions()
            for j, m in enumerate(junctions):
                motif_data.append(self._extract_motif_data(row, j, m))
        df_motif = pd.DataFrame(motif_data)

        return df_motif

    def _extract_motif_data(
        self, row: pd.Series, m_pos: int, m: SecStruct
    ) -> Dict[str, Any]:
        """Extract motif data for a single motif."""
        strands = m.strands
        flank_bps = self._get_flanking_base_pairs(row, strands)
        m_data = self._get_motif_reactivity_data(row, strands)
        m_strands = strands[0] + [-1] + strands[1]
        token = self._generate_motif_token(m.sequence)

        return {
            "construct": row["name"],
            "m_data": m_data,
            "m_pos": m_pos,
            "m_sequence": m.sequence,
            "m_structure": m.structure,
            "m_strands": m_strands,
            "m_token": token,
            # This line is unpacking the dictionary 'flank_bps' and adding its key-value pairs to the returned dictionary.
            **flank_bps,
            "num_aligned": row["num_aligned"],
            "sn": row["sn"],
        }

    def __create_helix_motif_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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
            for j, m in enumerate(ss.get_helices()):
                all_data.append(self.__get_helix_data(m, row, j))
        df_motif = pd.DataFrame(all_data)

        return df_motif

    def __get_helix_data(self, m, row, m_pos) -> Dict[str, Any]:
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
        second_bp = [strands[0][-1], strands[1][0]]
        second_bp_id = row["sequence"][second_bp[0]] + row["sequence"][second_bp[1]]
        flank_bp_5p = row["sequence"][strands[0][0]] + row["sequence"][strands[1][-1]]
        flank_bp_3p = row["sequence"][strands[0][-2]] + row["sequence"][strands[1][1]]
        m_data = []
        m_strands = strands[0][:-1] + [-1] + strands[1][1:]
        for pos in m_strands:
            if pos == -1:
                m_data.append(0)
            else:
                m_data.append(round(row["data"][pos], 6))
        seqs = m.sequence.split("&")
        ss = m.structure.split("&")
        token = "HELIX." + str(len(seqs[0]))
        data = {
            "constructs": row["name"],  # name of the construct
            "m_data": m_data,  # reactivity for the motif
            "m_pos": m_pos,  # position of the motif 0 to 7?
            "m_sequence": seqs[0][:-1] + "&" + seqs[1][1:],  # sequence of the motif
            "m_structure": ss[0][:-1] + "&" + ss[1][1:],  # structure of the motif
            "m_strands": m_strands,  # positions of each nuclotide of the motif
            "m_token": token,
            "m_flank_bp_5p": flank_bp_5p,  # base pair at the 5' end of the motif
            "m_flank_bp_3p": flank_bp_3p,  # base pair at the 3' end of the motif
            "m_second_flank_bp_5p": "N/A",  # first base pair of the motif before flanking
            "m_second_flank_bp_3p": second_bp_id,  # second base pair of the motif after flanking
            "num_aligned": row["num_aligned"],  # number of aligned reads
            "sn": row["sn"],  # signal to noise ratio
        }
        return data

    def _get_flanking_base_pairs(
        self, row: pd.Series, strands: List[List[int]]
    ) -> Dict[str, str]:
        """Get the flanking base pairs for a motif."""
        seq = row["sequence"]
        return {
            "m_flank_bp_5p": seq[strands[0][0]] + seq[strands[1][-1]],
            "m_flank_bp_3p": seq[strands[0][-1]] + seq[strands[1][0]],
            "m_second_flank_bp_5p": seq[strands[0][0] - 1] + seq[strands[1][-1] + 1],
            "m_second_flank_bp_3p": seq[strands[0][-1] + 1] + seq[strands[1][0] - 1],
        }

    def _get_motif_reactivity_data(
        self, row: pd.Series, strands: List[List[int]]
    ) -> List[float]:
        """Get the reactivity data for a motif."""
        m_data = [round(row["data"][pos], 6) for strand in strands for pos in strand]
        if len(strands) == 2:
            m_data.insert(len(strands[0]), 0)
        return m_data

    def _generate_motif_token(self, sequence: str) -> str:
        """Generate a token for the motif."""
        seqs = sequence.split("&")
        return f"{len(seqs[0]) - 2}x{len(seqs[1]) - 2}"

    def _standardize_motifs(self, df_motif: pd.DataFrame) -> pd.DataFrame:
        """Standardize motifs to ensure consistent orientation."""
        df_motif = df_motif.copy()
        df_motif[["strand1", "strand2"]] = df_motif["m_sequence"].str.split(
            "&", expand=True
        )
        df_motif["m_orientation"] = "non-flipped"

        flip_mask = [False] * len(df_motif)
        for i, row in df_motif.iterrows():
            if len(row["strand2"]) > len(row["strand1"]):
                flip_mask[i] = True
            elif (
                len(row["strand2"]) == len(row["strand1"])
                and row["strand1"] > row["strand2"]
            ):
                flip_mask[i] = True

        df_motif.iloc[flip_mask] = df_motif.iloc[flip_mask].apply(
            self._flip_motif, axis=1
        )

        df_motif = df_motif.drop(columns=["strand1", "strand2"])
        return df_motif

    def _flip_motif(self, row: pd.Series) -> pd.Series:
        """Flip a single motif."""
        len_s1 = len(row["strand1"])
        row["m_sequence"] = f"{row['strand2']}&{row['strand1']}"
        row["m_structure"] = self._flip_structure(row["m_structure"])
        row["m_strands"] = (
            row["m_strands"][len_s1 + 1 :] + [-1] + row["m_strands"][:len_s1]
        )
        row["m_data"] = row["m_data"][len_s1 + 1 :] + [0] + row["m_data"][:len_s1]
        if not row["m_token"].startswith("HELIX"):
            row["m_token"] = row["m_token"][::-1]
        row["m_flank_bp_5p"], row["m_flank_bp_3p"] = self._flip_pair(
            row["m_flank_bp_3p"]
        ), self._flip_pair(row["m_flank_bp_5p"])
        row["m_second_flank_bp_5p"], row["m_second_flank_bp_3p"] = self._flip_pair(
            row["m_second_flank_bp_3p"]
        ), self._flip_pair(row["m_second_flank_bp_5p"])
        row["m_orientation"] = "flipped"
        return row

    @staticmethod
    def _flip_structure(structure: str) -> str:
        """Flip the structure string."""
        return structure[::-1].translate(str.maketrans("().", ")(.", ""))

    @staticmethod
    def _flip_pair(pair: str) -> str:
        """Flip a base pair."""
        return pair[::-1]

    def _calculate_average_motif_data(self, df_motif: pd.DataFrame) -> pd.DataFrame:
        """Calculate average motif data for each unique motif sequence."""
        grouped = df_motif.groupby("m_sequence")
        avg_data = []

        for motif_seq, group in grouped:
            pdb_paths = self._get_pdb_paths(motif_seq)
            m_data_array = np.array(group["m_data"].tolist())
            m_data_avg, m_data_std, m_data_cv = self._calculate_statistics(m_data_array)
            pairs = self._get_likely_pairs(motif_seq)

            avg_data.append(
                {
                    "constructs": group["construct"].tolist(),
                    "has_pdbs": bool(pdb_paths),
                    "m_data_array": m_data_array,
                    "m_data_avg": m_data_avg,
                    "m_data_cv": m_data_cv,
                    "m_data_std": m_data_std,
                    "m_flank_bp_5p": group["m_flank_bp_5p"].tolist(),
                    "m_flank_bp_3p": group["m_flank_bp_3p"].tolist(),
                    "m_orientation": group["m_orientation"].tolist(),
                    "m_pos": group["m_pos"].tolist(),
                    "m_second_flank_bp_5p": group["m_second_flank_bp_5p"].tolist(),
                    "m_second_flank_bp_3p": group["m_second_flank_bp_3p"].tolist(),
                    "m_sequence": motif_seq,
                    "m_strands": group["m_strands"].tolist(),
                    "m_structure": group["m_structure"].iloc[0],
                    "m_token": group["m_token"].iloc[0],
                    "pairs": pairs,
                    "pdbs": pdb_paths,
                }
            )

        df_avg = pd.DataFrame(avg_data)
        return df_avg

    def _get_pdb_paths(self, motif_seq: str) -> List[str]:
        """Get PDB file paths for a given motif sequence."""
        motif_seq_path = motif_seq.replace("&", "_")
        rev_motif_seq_path = "_".join(reversed(motif_seq_path.split("_")))
        pdb_paths = []

        for seq_path in [motif_seq_path, rev_motif_seq_path]:
            # be consistent and use pdbs with 2 extra base pairs built by farfar
            path = f"{DATA_PATH}/pdbs_w_2bp/{seq_path}"
            if os.path.exists(path):
                pdbs = glob.glob(f"{path}/*.pdb")
                if pdbs:
                    pdb_paths.extend(pdbs)
                else:
                    log.warning(f"No PDB files found for {seq_path}")

        return pdb_paths

    @staticmethod
    def _calculate_statistics(
        data_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate average, standard deviation, and coefficient of variation."""
        avg = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        cv = np.divide(std, avg, out=np.zeros_like(std), where=avg != 0)
        return avg, std, cv

    @staticmethod
    def _get_likely_pairs(motif_seq: str) -> List[str]:
        """Get likely base pairs for a symmetric junction in the motif sequence."""
        seqs = motif_seq.split("&")
        if len(seqs[0]) == len(seqs[1]):
            pairs_1 = [n1 + n2 for n1, n2 in zip(seqs[0], reversed(seqs[1]))] + [""]
            pairs_2 = [n2 + n1 for n1, n2 in zip(seqs[0], reversed(seqs[1]))]
            return pairs_1 + list(reversed(pairs_2))
        return [""] * len(motif_seq)


# step 3: generate residue dataframes ##############################################
class GenerateResidueDataFrame:
    def run(self, df_motif, name, inf_sub):
        self.name = name
        df_residues_avg = self.__generate_avg_residue_dataframe(df_motif)
        self.__save_avg_residues_to_json(df_residues_avg)
        df_residues = self.__expand_residue_dataframe(df_residues_avg)
        df_residues = self.__add_log_data(df_residues, inf_sub)
        df_residues = self.__mark_outliers(df_residues)
        self.__save_residues_to_json(df_residues)

    def run_normalized_df(self, df_motif, name, inf_sub: int):
        """
        Execute the full residue analysis pipeline on a normalized motif dataframes.
        """
        self.name = name
        df_residues_avg = self.__generate_avg_residue_dataframe(df_motif)
        df_residues_avg.to_json(
            f"{REVISION_PATH}/normalized/residues/{self.name}_residues_avg.json",
            orient="records",
        )
        df_residues = self.__expand_residue_dataframe(df_residues_avg)
        df_residues = self.__add_log_data(df_residues, inf_sub)
        df_residues = self.__mark_outliers(df_residues)
        df_residues.to_json(
            f"{REVISION_PATH}/normalized/residues/{self.name}_residues.json",
            orient="records",
        )

    def run_different_threshold_df(self, df_motif, name, threshold: int, inf_sub: int):
        """
        Execute the full residue analysis pipeline on a motif dataframes at different thresholds.

        This method processes the input motif dataframe through a series of steps:
        1. Generates an averaged residue dataframe based on the specified threshold.
        2. Expands the averaged dataframe into a full residue-level dataframe.
        3. Adds logarithmic-transformed data columns, with substitution handling.
        4. Marks statistical outliers within the dataset.
        5. Saves the processed residue dataframe to a JSON file, labeled by threshold.

        Parameters
        ----------
        df_motif : pandas.DataFrame
            Input dataframe containing motif-related residue data.
        name : str
            Name identifier for this run, stored as an instance attribute.
        threshold : int
            Cutoff value used to filter motifs when generating the averaged residue dataframe.
        inf_sub : int
            Value used to substitute infinite or undefined values during log-data computation.

        Returns
        -------
        None
            Results are saved to a JSON file; no dataframe is returned directly.
        """
        self.name = name
        df_residues_avg = self.__generate_avg_residue_dataframe(df_motif)
        df_residues_avg.to_json(
            f"{REVISION_PATH}/dif_threshold/residues/{self.name}_residues_avg_{threshold}.json",
            orient="records",
        )
        df_residues = self.__expand_residue_dataframe(df_residues_avg)
        df_residues = self.__add_log_data(df_residues, inf_sub)
        df_residues = self.__mark_outliers(df_residues)
        df_residues.to_json(
            f"{REVISION_PATH}/dif_threshold/residues/{self.name}_residues_{threshold}.json",
            orient="records",
        )

    def __generate_avg_residue_dataframe(self, df_motif):
        all_data = []
        for _, row in df_motif.iterrows():
            residue_data = self.__process_motif_row(row)
            all_data.extend(residue_data)
        df_residues = pd.DataFrame(all_data)
        return df_residues

    def __process_motif_row(self, row):
        residue_data = []
        m_sequence = row["m_sequence"]
        m_structure = row["m_structure"]
        for i, (e, s) in enumerate(zip(m_sequence, m_structure)):
            if e not in ["A", "C"]:
                continue
            residue_info = self.__get_residue_info(
                row, m_sequence, m_structure, i, e, s
            )
            residue_data.append(residue_info)
        return residue_data

    def __get_residue_info(self, row, m_sequence, m_structure, i, e, s):
        r_type = "Flank-WC" if s in "()" else "NON-WC"
        # helix motifs aren't associated with a pdb
        if row["m_token"].startswith("HELIX"):
            pdb_r_pos = -1
            r_type = "WC"
        else:
            pdb_r_pos = self.__calculate_pdb_r_pos(i, m_sequence)

        return {
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
            "r_avg": row["m_data_avg"][i],
            "r_cv": row["m_data_cv"][i],
            "r_data": np.array(row["m_data_array"])[:, i].tolist(),
            "r_nuc": e,
            "r_loc_pos": i,
            "r_pos": np.array(row["m_strands"])[:, i].tolist(),
            "r_stack" "r_std": row["m_data_std"][i],
            "r_type": r_type,
            "pdb_path": row["pdbs"],
            "pdb_r_pos": pdb_r_pos,
        }

    def __get_residue_types(self, p5_res, p3_res):
        p5_type = "PURINE" if p5_res in ["A", "G"] else "PYRIMIDINE" if p5_res else None
        p3_type = "PURINE" if p3_res in ["A", "G"] else "PYRIMIDINE" if p3_res else None
        return p5_type, p3_type

    def __check_neighboring_types(self, p5_type, p3_type):
        both_purine = p5_type == "PURINE" and p3_type == "PURINE"
        both_pyrimidine = p5_type == "PYRIMIDINE" and p3_type == "PYRIMIDINE"
        return both_purine, both_pyrimidine

    def __calculate_pdb_r_pos(self, i, m_sequence):
        pdb_r_pos = i + 3
        break_pos = m_sequence.find("&")
        if break_pos < i:
            pdb_r_pos += 3
        return pdb_r_pos

    def __expand_residue_dataframe(self, df_residues_avg):
        all_data = []
        for _, row in df_residues_avg.iterrows():
            for i in range(len(row["r_data"])):
                data = self.__create_expanded_row(row, i)
                all_data.append(data)
        return pd.DataFrame(all_data)

    def __get_neighboring_residues(
        self, m_sequence, i, m_second_flank_bp_5p, m_second_flank_bp_3p
    ):
        def get_p5_res():
            if i == 0:
                return m_second_flank_bp_5p[0]
            elif m_sequence[i - 1] == "&":
                return m_second_flank_bp_3p[1]
            else:
                return m_sequence[i - 1]

        def get_p3_res():
            if i == len(m_sequence) - 1:
                return m_second_flank_bp_5p[1]
            elif m_sequence[i + 1] == "&":
                return m_second_flank_bp_3p[0]
            else:
                return m_sequence[i + 1]

        p5_res = get_p5_res()
        p3_res = get_p3_res()
        return p5_res, p3_res

    def __create_expanded_row(self, row, i):
        m_sequence = row["m_sequence"]
        p5_res, p3_res = self.__get_neighboring_residues(
            m_sequence,
            row["r_loc_pos"],
            row["m_second_flank_bp_5p"][i],
            row["m_second_flank_bp_3p"][i],
        )
        p5_type, p3_type = self.__get_residue_types(p5_res, p3_res)
        both_purine, both_pyrimidine = self.__check_neighboring_types(p5_type, p3_type)
        return {
            "both_purine": both_purine,
            "both_pyrimidine": both_pyrimidine,
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
            "p5_res": p5_res,
            "p5_type": p5_type,
            "p3_res": p3_res,
            "p3_type": p3_type,
            "r_data": row["r_data"][i],
            "r_nuc": row["r_nuc"],
            "r_loc_pos": row["r_loc_pos"],
            "r_pos": row["r_pos"][i],
            "r_type": row["r_type"],
            "r_stack": p5_res + p3_res,
            "pdb_path": row["pdb_path"],
            "pdb_r_pos": row["pdb_r_pos"],
        }

    def __add_log_data(self, df_residues, inf_sub):
        df_residues["ln_r_data"] = np.log(df_residues["r_data"])
        df_residues["ln_r_data"].replace(-np.inf, inf_sub, inplace=True)
        return df_residues

    def __mark_outliers(self, df_residues):
        df_residues["z_score"] = 0
        df_residues["r_data_outlier"] = False
        data = []
        for i, g in df_residues.groupby(["m_sequence", "r_loc_pos"]):
            g["z_score"] = zscore(g["r_data"])
            g["r_data_outlier"] = g["z_score"].abs() > 3
            data.append(g)
        df_residues = pd.concat(data)
        return df_residues

    def __save_residues_to_json(self, df_residues):
        df_residues.to_json(
            f"{DATA_PATH}/raw-jsons/residues/{self.name}_residues.json",
            orient="records",
        )

    def __save_avg_residues_to_json(self, df_residues):
        df_residues.to_json(
            f"{DATA_PATH}/raw-jsons/residues/{self.name}_residues_avg.json",
            orient="records",
        )


# step 4: merge pdb info into motif and residue dataframes ##########################
def generate_pdb_residue_dataframe(df_residue):
    # this stores what type of non-wc bair each residue is part of
    df_pairs = pd.read_csv(f"{DATA_PATH}/csvs/basepair_data_for_motifs.csv")
    # describes which residues are in a pair
    df_pair_info = pd.read_csv(f"{DATA_PATH}/csvs/all_bp_details.csv")
    df_pair_info = df_pair_info[["name", "motif", "res_num1", "res_num2", "bp"]]
    df_pair_info["name"] = df_pair_info["name"].str.replace("_x3dna.out", "")
    df_pair_info.rename(columns={"name": "pdb_name"}, inplace=True)
    df_pair_info["pdb_name"] = df_pair_info["pdb_name"] + ".pdb"
    # gives the resolution of each pdb
    df_res = pd.read_csv(f"{DATA_PATH}/csvs/pdb_res.csv")
    df_res.drop(["m_sequence"], axis=1, inplace=True)
    df_res["pdb_name"] = [x + ".pdb" for x in df_res["pdb_name"]]
    # gives the b-factor of each residue
    df_bfact = pd.read_csv(f"data/pdb-features/b_factor.csv")
    df_bfact = df_bfact[
        ["pdb_name", "pdb_r_pos", "average_b_factor", "normalized_b_factor"]
    ]
    # get the paths for each pdb
    df_paths = []
    for i, row in df_pairs.iterrows():
        try:
            path = glob.glob(f"{DATA_PATH}/pdbs_w_2bp/*/{row['pdb_name']}")[0]
        except:
            log.info(f"no pdb found for {row['pdb_name']}")
            path = ""
        df_paths.append(path)
    df_pairs["pdb_path"] = df_paths
    df_pairs = df_pairs.merge(df_res, on="pdb_name")
    # df_pairs = df_pairs.merge(df_bfact, on=["pdb_name", "pdb_r_pos"])
    df_pairs.to_csv(f"{DATA_PATH}/pdb-features/pairs.csv", index=False)
    df_residue = df_residue.query("has_pdbs == True").copy()
    df_residue["m_sequence"] = df_residue["m_sequence"].apply(
        lambda x: x.replace("&", "_")
    )
    df_residue.drop(["has_pdbs", "pdb_path", "r_nuc", "r_type"], axis=1, inplace=True)
    df_final = df_pairs.merge(df_residue, on=["m_sequence", "pdb_r_pos"], how="left")
    df_final["pair_pdb_r_pos"] = -1
    # TODO need to understand why some are not being found is it just because they are not in the 3dna output? or like the multiple iteraction thing?
    for i, row in df_final.iterrows():
        if "lone" in row["pdb_r_bp_type"]:
            continue
        df_sub = df_pair_info.query(
            f"pdb_name == '{row['pdb_name']}' and res_num1 == {row['pdb_r_pos']}"
        )
        if len(df_sub) == 1:
            df_final.loc[i, "pair_pdb_r_pos"] = df_sub["res_num2"].values[0]
        elif len(df_sub) == 0:
            df_sub = df_pair_info.query(
                f"pdb_name == '{row['pdb_name']}' and res_num2 == {row['pdb_r_pos']}"
            )
            if len(df_sub) == 1:
                df_final.loc[i, "pair_pdb_r_pos"] = df_sub["res_num1"].values[0]
            else:
                continue
                # print(row["pdb_r_bp_type"], row["pdb_path"], row["pdb_r_pos"])
        else:
            continue
            # print(row["pdb_r_bp_type"], row["pdb_path"], row["pdb_r_pos"]
    return df_final


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

    # inf_sub values for denatured_norm = -7.6, nomod_norm = -5.3, pdb_library_1 = -9.8
    df_denature = pd.read_json(
        f"{DATA_PATH}/raw-jsons/constructs/pdb_library_denature.json"
    )
    df = pd.read_json(f"{DATA_PATH}/raw-jsons/constructs/pdb_library_1.json")
    df_nomod = pd.read_json(f"{DATA_PATH}/raw-jsons/constructs/pdb_library_nomod.json")

    ## NORMALIZING USING DENATURE DATA ##
    df_denature_norm = merge_and_normalize(df, df_denature)
    df_denature_norm.to_json(
        f"{REVISION_PATH}/normalized/constructs/pdb_library_denature_normalized.json",
        orient="records",
    )

    ## NORMALIZING USING NOMOD DATA ##
    df_nomod_norm = merge_and_normalize(df, df_nomod)
    df_nomod_norm.to_json(
        f"{REVISION_PATH}/normalized/constructs/pdb_library_nomod_normalized.json",
        orient="records",
    )
    exit()

    ## GENERATING DATAFRAMES WITH DIFFERENT THRESHOLDS ##
    thresholds = [1000, 2000, 3000, 4000, 5000, 10000, 20000, 40000, 80000]
    sn_val_diff = 4.0
    inf_sub = -9.8

    for t in thresholds:
        gen = GenerateMotifDataFrame()
        log.info("Generating motif dataframe for denature data")
        gen.run_different_threshold_df(df, "pdb_library_1", t, sn_val_diff)
        df1 = pd.read_json(
            f"{REVISION_PATH}/dif_threshold/motifs/pdb_library_1_motifs_avg_{t}.json"
        )
        log.info("Generating residue dataframe for denature data")
        gen = GenerateResidueDataFrame()
        gen.run_different_threshold_df(df1, "pdb_library_1", t, inf_sub)

    ## GENERATING DATAFRAMES FOR PDB_LIBRARY_1 ##
    gen = GenerateMotifDataFrame()
    log.info("Generating motif dataframe")
    gen.run(df, "pdb_library_1")
    df1 = pd.read_json(f"{DATA_PATH}/raw-jsons/motifs/pdb_library_1_motifs_avg.json")
    log.info("Generating residue dataframe")
    gen = GenerateResidueDataFrame()
    gen.run(df1, "pdb_library_1", inf_sub)
    df1 = pd.read_json(f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues.json")
    log.info("Generating pdb residue dataframe")
    df1 = generate_pdb_residue_dataframe(df1)
    df1.to_json(
        f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues_pdb.json",
        orient="records",
    )

    ## GENERATING DATAFRAMES FOR DENATURE DATA ##
    sn_val = 0
    threshold = 2000
    gen = GenerateMotifDataFrame()
    log.info("Generating motif dataframe for denature data")
    gen.run_different_threshold_df(
        df_denature, "pdb_library_denature", threshold, sn_val
    )
    df_denature = pd.read_json(
        f"{REVISION_PATH}/dif_threshold/motifs/pdb_library_denature_motifs_avg_2000.json"
    )
    log.info("Generating residue dataframe for denature data")
    gen = GenerateResidueDataFrame()
    gen.run_different_threshold_df(
        df_denature, "pdb_library_denature", threshold, inf_sub
    )

    ## GENERATING DATAFRAMES FOR NORMALIZED DATA WHICH WAS NORMALIZED USING DENATURE DATA ##
    inf_sub_den_norm = -7.6
    gen = GenerateMotifDataFrame()
    log.info("Generating normalized motif dataframe using denature data")
    gen.run_normalized_df(df_denature_norm, "pdb_library_denature_normalized")
    df_denature_norm = pd.read_json(
        f"{REVISION_PATH}/normalized/motifs/pdb_library_denature_normalized_motifs_avg.json"
    )
    log.info("Generating normalized residue dataframe using denature data")
    gen = GenerateResidueDataFrame()
    gen.run_normalized_df(
        df_denature_norm, "pdb_library_denature_normalized", inf_sub_den_norm
    )

    ## GENERATING DATAFRAMES FOR NORMALIZED DATA WHICH WAS NORMALIZED USING NOMOD DATA ##
    inf_sub_nomod_norm = -5.3
    gen = GenerateMotifDataFrame()
    log.info("Generating normalized motif dataframe using nomod data")
    gen.run_normalized_df(df_nomod_norm, "pdb_library_nomod_normalized")
    df_nomod_norm = pd.read_json(
        f"{REVISION_PATH}/normalized/motifs/pdb_library_nomod_normalized_motifs_avg.json"
    )
    log.info("Generating normalized residue dataframe using nomod data")
    gen = GenerateResidueDataFrame()
    gen.run_normalized_df(
        df_nomod_norm, "pdb_library_nomod_normalized", inf_sub_nomod_norm
    )


def main():
    """
    main function for script
    """
    setup_logging()
    regen_data()


if __name__ == "__main__":
    main()
