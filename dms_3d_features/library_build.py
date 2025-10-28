# Standard library imports
import re
import os
import random
from typing import List, Tuple, Dict
from collections import defaultdict

# Third party imports
import pandas as pd
import editdistance
from vienna import fold
import numpy as np

# Local imports
from dms_3d_features.logger import get_logger, setup_logging
from dms_3d_features.paths import DATA_PATH

log = get_logger("library-build")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the motif sequences data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the motif sequences.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def initialize_variables() -> dict:
    """
    Initialize global variables used in the sequence generation process.

    Returns:
        dict: Dictionary containing initialized variables for the process.
    """
    return {
        "pool": [],
        "pool_motifs": [],
        "pool_m_ss": [],
        "usable_seq": [],
        "usable_ss": [],
        "usable_motifs": [],
        "usable_m_ss": [],
        "seq_len": [],
        "ens_def": [],
        "edit_dis": [],
    }


def generate_complementary_pairs(rna_bases: dict) -> Tuple[List[str], List[str]]:
    """
    Generate complementary RNA base pairs for hairpin sets.

    Args:
        RNA_bases (dict): Dictionary mapping RNA bases to their complementary pairs.

    Returns:
        tuple[list[str], list[str]]: Two lists representing complementary hairpin sets.
    """
    items = list(rna_bases.items())
    hairpin_set1, hairpin_set2 = [], []
    for _ in range(2):
        random.shuffle(items)
        key, value = random.choice(items)
        hairpin_set1.append(key)
        hairpin_set2.append(value)
    return hairpin_set1, hairpin_set2[::-1]


def select_rows(df: pd.DataFrame, num_rows_to_select: int) -> List[int]:
    """
    Randomly select rows from the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing motif sequences.
        num_rows_to_select (int): Number of rows to select.

    Returns:
        list[int]: List of indices for selected rows.
    """
    selected_rows = []
    available_rows = [row for row in df.index if row not in selected_rows]
    while len(selected_rows) < num_rows_to_select:
        if not available_rows:
            break
        random_row = random.choice(available_rows)
        selected_rows.append(random_row)
    return selected_rows


def get_rows_with_min_std_dev(
    df: pd.DataFrame,
    selected_count: Dict[str, int],
    attempts=10,
    min_rows=5,
    max_rows=7,
) -> list:
    """
    Find the set of rows that minimize the standard deviation of motif counts.

    Args:
        df (DataFrame): The data frame containing motif sequences.
        selected_count (dict): Dictionary tracking the count of each motif.
        attempts (int): Number of attempts to generate row combinations.
        min_rows (int): Minimum number of rows to select.
        max_rows (int): Maximum number of rows to select.

    Returns:
        list: The rows with the minimum standard deviation of motif counts.
    """
    min_std_dev = float("inf")
    best_selected_rows = None

    for _ in range(attempts):
        num_rows_to_select = random.randint(min_rows, max_rows)
        candidate_rows = select_rows(df, num_rows_to_select)

        motif_counts = [
            selected_count.get(df.loc[row, "motif_seq"], 0) + 1
            for row in candidate_rows
        ]

        std_dev = np.std(motif_counts)

        if std_dev < min_std_dev:
            min_std_dev = std_dev
            best_selected_rows = candidate_rows

    return best_selected_rows


def construct_sequences(
    selected_rows: List[int],
    df: pd.DataFrame,
    hairpin_set1: List[str],
    hairpin_set2: List[str],
    hairpin: List[str],
    hairpin_ss: List[str],
    five_prime: List[str],
    five_prime_ss: List[str],
    three_prime: List[str],
    three_prime_ss: List[str],
    rna_bases: dict,
) -> Tuple[str, str, List[str], List[str]]:
    """
    Construct RNA sequences and their secondary structures.

    Args:
        selected_rows (list[int]): Indices of selected rows.
        df (pd.DataFrame): DataFrame containing the motif data.
        hairpin_set1 (list[str]): First complementary hairpin set.
        hairpin_set2 (list[str]): Second complementary hairpin set.
        hairpin (list[str]): Hairpin sequence.
        hairpin_ss (list[str]): Hairpin secondary structure.
        five_prime (list[str]): 5' sequence.
        five_prime_ss (list[str]): 5' secondary structure.
        three_prime (list[str]): 3' sequence.
        three_prime_ss (list[str]): 3' secondary structure.
        rna_bases (dict): RNA base-pair mapping.

    Returns:
        tuple[str, str, list[str], list[str]]: Full RNA sequence, its secondary structure,
                                               motifs used and their secondary structures
    """
    items = list(rna_bases.items())
    full_seq_right, full_seq_left = [], []
    full_seq_right_ss, full_seq_left_ss = [], []
    selected_motifs, selected_ss = [], []

    for row in selected_rows:
        seq_value = df.loc[row, "motif_seq"]
        selected_motifs.append(seq_value)
        ss_value = df.loc[row, "motif_ss"]
        selected_ss.append(ss_value)
        set1, set2 = [], []
        for _ in range(3):
            random.shuffle(items)
            position1 = random.randint(0, len(items) - 1)
            key, value = items[position1]
            set1.append(key)
            set2.append(value)
            st1, st2_rev = "".join(set1), "".join(set2[::-1])
        st1_ss, st2_ss = "(((", ")))"
        seq_value1, seq_value2 = seq_value.split("&")
        ss_value1, ss_value2 = ss_value.split("&")

        full_seq_left.append(st1 + seq_value1)
        full_seq_left_ss.append(st1_ss + ss_value1)
        full_seq_right.insert(0, seq_value2 + st2_rev)
        full_seq_right_ss.insert(0, ss_value2 + st2_ss)

    seq = (
        five_prime
        + full_seq_left
        + hairpin_set1
        + hairpin
        + hairpin_set2
        + full_seq_right
        + three_prime
    )
    ss = (
        five_prime_ss
        + full_seq_left_ss
        + ["(", "("]
        + hairpin_ss
        + [")", ")"]
        + full_seq_right_ss
        + three_prime_ss
    )
    return "".join(seq), "".join(ss), selected_motifs, selected_ss


def validate_sequence(seq_length: int, usable_seq: List[str]) -> bool:
    """
    Validate the generated sequence based on length.

    Args:
        seq_length (int): the length of the sequence
        usable_seq (list[str]): List of already usable sequences.

    Returns:
        bool: True if the sequence is valid, otherwise False.
    """
    if seq_length <= 140:
        return False
    if usable_seq:
        max_allowed_length = max(len(seq) for seq in usable_seq) * 1.05
        min_allowed_length = min(len(seq) for seq in usable_seq) * 0.95
        if not (min_allowed_length <= seq_length <= max_allowed_length):
            return False

    return True


def add_to_pool(
    full_seq: str,
    full_ss: str,
    pool: List[str],
    pool_motifs: List[str],
    pool_m_ss: List[str],
    selected_motif: List[str],
    selected_ss: List[str],
) -> None:
    """
    Add a valid sequence to the pool.

    Args:
        full_seq (str): Generated RNA sequence.
        full_ss (str): Secondary structure of the sequence.
        pool (list[str]): List of sequences in the pool.
        pool_motifs (list[str]): Motifs corresponding to pool sequences.
        pool_m_ss (list[str]): Secondary structures of the motifs in the pool.
        selected_motif (list[str]): List of selected motifs.
        selected_ss (list[str]): List of secondary structures for the motifs.
    """
    full_ss_RNAfold = fold(full_seq).dot_bracket
    if full_ss == full_ss_RNAfold:
        pool.append(full_seq)
        pool_motifs.append(selected_motif)
        pool_m_ss.append(selected_ss)


def no_of_seqs_less_than_50(
    selected_rows: List[int],
    df: pd.DataFrame,
    hairpin_set1: List[str],
    hairpin_set2: List[str],
    hairpin: List[str],
    hairpin_ss: List[str],
    five_prime: List[str],
    five_prime_ss: List[str],
    three_prime: List[str],
    three_prime_ss: List[str],
    rna_bases: Dict[str, str],
    variables: Dict,
    selected_count: Dict[str, int],
) -> bool:
    """
    Process the iteration when usable sequences are less than 50.

    Args:
        selected_rows (list[int]): Indices of selected rows.
        df (pd.DataFrame): DataFrame containing the motif data.
        hairpin_set1 (list[str]): First complementary hairpin set.
        hairpin_set2 (list[str]): Second complementary hairpin set.
        hairpin (list[str]): Hairpin sequence.
        hairpin_ss (list[str]): Hairpin secondary structure.
        five_prime (list[str]): 5' sequence.
        five_prime_ss (list[str]): 5' secondary structure.
        three_prime (list[str]): 3' sequence.
        three_prime_ss (list[str]): 3' secondary structure.
        rna_bases (dict): RNA base-pair mapping.
        variables (dict): Dictionary containing global variables for the process.
        selected_count (dict[str, int]): Dictionary that would give the number of times
                                         motif was used in the final pool.

    Returns:
        bool: True if a sequence is added to the pool, otherwise False.
    """
    for row in selected_rows:
        seq_value = df.loc[row, "motif_seq"]
        selected_count[seq_value] = selected_count.get(seq_value, 0) + 1

    full_seq, full_ss, selected_motifs, selected_ss = construct_sequences(
        selected_rows,
        df,
        hairpin_set1,
        hairpin_set2,
        hairpin,
        hairpin_ss,
        five_prime,
        five_prime_ss,
        three_prime,
        three_prime_ss,
        rna_bases,
    )

    add_to_pool(
        full_seq,
        full_ss,
        variables["pool"],
        variables["pool_motifs"],
        variables["pool_m_ss"],
        selected_motifs,
        selected_ss,
    )
    return True


def no_of_seqs_greater_than_50(
    df: pd.DataFrame,
    hairpin_set1: List[str],
    hairpin_set2: List[str],
    hairpin: List[str],
    hairpin_ss: List[str],
    five_prime: List[str],
    five_prime_ss: List[str],
    three_prime: List[str],
    three_prime_ss: List[str],
    rna_bases: Dict[str, str],
    variables: Dict,
    selected_count: Dict[str, int],
    length_w_no_motifs: int,
) -> bool:
    """
    Process the iteration when usable sequences are greater than or equal to 50.

    Args:
        df (pd.DataFrame): DataFrame containing the motif data.
        hairpin_set1 (list[str]): First complementary hairpin set.
        hairpin_set2 (list[str]): Second complementary hairpin set.
        hairpin (list[str]): Hairpin sequence.
        hairpin_ss (list[str]): Hairpin secondary structure.
        five_prime (list[str]): 5' sequence.
        five_prime_ss (list[str]): 5' secondary structure.
        three_prime (list[str]): 3' sequence.
        three_prime_ss (list[str]): 3' secondary structure.
        rna_bases (dict): RNA base-pair mapping.
        variables (dict): Dictionary containing global variables for the process.
        selected_count (dict[str, int]): Dictionary that would give the number of times
                                         motif was used in the final pool.
        length_w_no_motifs (int): length of the sequence without adding the motifs

    Returns:
        bool: True if a sequence is added to the pool, otherwise False.
    """
    best_selected_rows = get_rows_with_min_std_dev(df, selected_count)

    if not best_selected_rows:
        return False

    motifs_length = sum(
        (len(df.loc[row, "motif_seq"]) - 1) for row in best_selected_rows
    )
    seq_length = length_w_no_motifs + motifs_length

    if not validate_sequence(seq_length, variables["usable_seq"]):
        return False

    for row in best_selected_rows:
        seq_value = df.loc[row, "motif_seq"]
        selected_count[seq_value] = selected_count.get(seq_value, 0) + 1

    full_seq, full_ss, selected_motifs, selected_ss = construct_sequences(
        best_selected_rows,
        df,
        hairpin_set1,
        hairpin_set2,
        hairpin,
        hairpin_ss,
        five_prime,
        five_prime_ss,
        three_prime,
        three_prime_ss,
        rna_bases,
    )

    add_to_pool(
        full_seq,
        full_ss,
        variables["pool"],
        variables["pool_motifs"],
        variables["pool_m_ss"],
        selected_motifs,
        selected_ss,
    )
    return True


def finalize_sequences(
    pool: List[str], variables: dict, desired_sequences: int
) -> None:
    """
    Filter and finalize usable sequences from the pool.

    Args:
        pool (list[str]): List of sequences in the pool.
        variables (dict): Dictionary containing global variables for the process.
        desired_sequences (int): Number of desired sequences to finalize.
    """
    usable_seq, usable_motifs, usable_ss, usable_m_ss, seq_len, ens_def, edit_dis = (
        variables["usable_seq"],
        variables["usable_motifs"],
        variables["usable_ss"],
        variables["usable_m_ss"],
        variables["seq_len"],
        variables["ens_def"],
        variables["edit_dis"],
    )

    for i, (p1, m1, s1) in enumerate(
        zip(pool, variables["pool_motifs"], variables["pool_m_ss"])
    ):
        folded_p1 = fold(p1)
        ens_defect_p1 = folded_p1.ens_defect

        for p2 in pool[i:]:
            y = editdistance.eval(p1, p2)
            if y > 20 and ens_defect_p1 <= 5:
                if p1 not in usable_seq:
                    log.info(p1)
                    usable_motifs.append(m1)
                    usable_m_ss.append(s1)
                    usable_seq.append(p1)
                    usable_ss.append(folded_p1.dot_bracket)
                    seq_len.append(len(p1))
                    ens_def.append(ens_defect_p1)
                    edit_dis.append(y)
                break

        if len(usable_seq) >= desired_sequences:
            break


def build_pdb_library_from_motif_df(
    df: pd.DataFrame, desired_sequences: int = 100
) -> pd.DataFrame:
    """
    Generates a DataFrame of RNA sequences with secondary structures, given a motifs dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing motif sequences (as from load_data()).
        desired_sequences (int): The number of sequences to generate (default 100).

    Returns:
        pd.DataFrame: DataFrame with usable sequences and associated features.
    """
    hairpin = list("GCGAGUAGC")
    hairpin_ss = list("((.....))")
    rna_bases = {"A": "U", "U": "A", "C": "G", "G": "C"}
    five_prime = list("GGGCUUCGGCCCA")
    five_prime_ss = list("((((....)))).")
    three_prime = list("AAAGAAACAACAACAACAAC")
    three_prime_ss = list("....................")

    variables = initialize_variables()
    selected_count = defaultdict(int)

    helices_length = 46
    length_w_no_motifs = (
        len(hairpin) + len(five_prime) + len(three_prime) + helices_length
    )

    while len(variables["usable_seq"]) < desired_sequences:
        hairpin_set1, hairpin_set2 = generate_complementary_pairs(rna_bases)

        num_rows_to_select = random.randint(5, 7)
        selected_rows = select_rows(df, num_rows_to_select)

        motifs_length = sum(
            (len(df.loc[row, "motif_seq"]) - 1) for row in selected_rows
        )
        seq_length = length_w_no_motifs + motifs_length

        if not validate_sequence(seq_length, variables["usable_seq"]):
            continue

        if len(variables["usable_seq"]) < 50:
            success = no_of_seqs_less_than_50(
                selected_rows,
                df,
                hairpin_set1,
                hairpin_set2,
                hairpin,
                hairpin_ss,
                five_prime,
                five_prime_ss,
                three_prime,
                three_prime_ss,
                rna_bases,
                variables,
                selected_count,
            )
        else:
            success = no_of_seqs_greater_than_50(
                df,
                hairpin_set1,
                hairpin_set2,
                hairpin,
                hairpin_ss,
                five_prime,
                five_prime_ss,
                three_prime,
                three_prime_ss,
                rna_bases,
                variables,
                selected_count,
                length_w_no_motifs,
            )

        if success:
            finalize_sequences(variables["pool"], variables, desired_sequences)

    # Return a DataFrame with the same structure as the original output
    df_final = pd.DataFrame(
        {
            "seq": variables["usable_seq"],
            "ss": variables["usable_ss"],
            "motifs": variables["usable_motifs"],
            "motifs_ss": variables["usable_m_ss"],
            "len": variables["seq_len"],
            "ens_defect": variables["ens_def"],
            "edit_distance": variables["edit_dis"],
        }
    )
    return df_final


def main() -> None:
    setup_logging()
    df = load_data(f"{DATA_PATH}/csvs/motif_sequences.csv")
    df_final = build_pdb_library_from_motif_df(df, desired_sequences=10)
    df_final.to_json(f"{DATA_PATH}/jsons/pdb_library.json", orient="records")


if __name__ == "__main__":
    main()
