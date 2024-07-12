import glob
import os
import itertools
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp


from seq_tools import SequenceStructure, fold, to_rna, has_5p_sequence
from seq_tools import trim as seq_ss_trim
from seq_tools.structure import find as seq_ss_find
from rna_secstruct import SecStruct

from rna_map.mutation_histogram import (
    get_mut_histos_from_pickle_file,
    get_dataframe,
    convert_dreem_mut_histos_to_mutation_histogram,
)

from dms_3d_features.plotting import plot_pop_avg_from_row

# assume data/ is location of data
# assume data/mutation-histograms/ is location of mutation histograms

RESOURCES_PATH = "dms_3d_features/resources"
DATA_PATH = "data"

# helper functions ##################################################################


def trim(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Trims the 'sequence', 'structure', and 'data' columns of the DataFrame to the given start and end indices.

    Args:
        df (pd.DataFrame): A DataFrame with 'sequence', 'structure', and 'data' columns, where 'data' contains lists of numbers.
        start (int): The start index for trimming.
        end (int): The end index for trimming.

    Returns:
        pd.DataFrame: A trimmed DataFrame with the 'sequence', 'structure', and 'data' columns adjusted to the specified indices.

    Example:
        >>> df = pd.DataFrame({
        ...     "sequence": ["ABCDEFG", "HIJKLMN", "OPQRSTU"],
        ...     "structure": ["1234567", "2345678", "3456789"],
        ...     "data": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        ... })
        >>> trimmed_df = trim(df, 1, 2)
        >>> print(trimmed_df)
          sequence structure         data
        0     BCDEF    23456     [2, 3, 4]
        1     IJKLM    34567     [7, 8, 9]
        2     PQRST    45678  [12, 13, 14]
    """
    df = seq_ss_trim(df, start, end)
    if start == 0 and end != 0:
        df["data"] = df["data"].apply(lambda x: x[:-end])
    elif end == 0 and start != 0:
        df["data"] = df["data"].apply(lambda x: x[start:])
    elif start == 0 and end == 0:
        df["data"] = df["data"].apply(lambda x: x)
    else:
        df["data"] = df["data"].apply(lambda x: x[start:-end])
    return df


def trim_p5_and_p3(df: pd.DataFrame) -> pd.DataFrame:
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

    Example:
        >>> df = pd.DataFrame({"data": ["GGAAGATCGAGTAGATCAAAGCATGC", "GGAAGATCGAGTAGATCAAAGCATGC", "GGAAGATCGAGTAGATCAAAGCATGC"]})
        >>> trimmed_df = trim_p5_and_p3(df)
        >>> print(trimmed_df)
           data
        0  GCATGCAT
        1  GCATGCAT
        2  GCATGCAT
    """
    df_p5 = pd.read_csv(f"{RESOURCES_PATH}/csvs/p5_sequences.csv")
    df_p5 = to_rna(df_p5)
    common_p5_seq = ""
    for p5_seq in df_p5["sequence"]:
        if has_5p_sequence(df, p5_seq):
            common_p5_seq = p5_seq
    if len(common_p5_seq) == 0:
        raise ValueError("No common p5 sequence found")
    # log.info(f"common p5 sequence: {common_p5_seq}")
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

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})
        >>> split_dataframe(df, 2)
        [   A  B
         0  1  6
         1  2  7
         2  3  8,
           A   B
         3  4   9
         4  5  10]
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


# step 1: convert raw pickled mutation histograms to dataframe json files
def convert_mut_histos_to_df():
    """
    Converts mutation histograms to a DataFrame and saves the results as JSON files.

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
    n = 10
    for pfile in pickle_files:
        name = pfile.split("/")[-1].split(".")[0]
        print(name)
        if os.path.isfile(f"data/raw-jsons/{name}.json"):
            continue
        mut_histos = get_mut_histos_from_pickle_file(pfile)
        mut_histos = convert_dreem_mut_histos_to_mutation_histogram(mut_histos)
        df_results = get_dataframe(mut_histos, cols)
        df_results.rename(columns={"pop_avg": "data"}, inplace=True)
        df_results = to_rna(df_results)
        df_chunks = split_dataframe(df_results, n)
        results = []
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = {executor.submit(fold, chunk): chunk for chunk in df_chunks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        # Combine the results into a single DataFrame
        final_result = pd.concat(results)
        final_result = trim_p5_and_p3(final_result)
        final_result.to_json(f"data/raw-jsons/{name}.json", orient="records")


class GenerateMotifDataFrame:
    def __init__(self):
        pass

    def run(self, df):
        # initial filtering
        df = df.query("num_aligned > 2000 and sn > 4.0")
        # create initial motif dataframe each motif is its own row
        df_motif = self.__create_initial_motif_dataframe(df)
        # standarize motifs so they are all in the same orientation
        df_motif = self.__standardize_motif_dataframe(df_motif)
        all_data = []

    # step 1 ##################################################################
    def __create_initial_motif_dataframe(self, df):
        all_data = []
        for _, row in df.iterrows():
            ss = SecStruct(row["sequence"], row["structure"])
            for j, m in enumerate(ss.get_junctions()):
                all_data.append(self.__get_motif_data(m, row, j))
        df_motif = pd.DataFrame(all_data)
        df_motif.to_json(
            "data/raw-jsons/pdb_library_1_combined_motifs.json", orient="records"
        )
        return df_motif

    def __get_motif_data(self, m, row, m_pos) -> Dict[str, Any]:
        """
        Get the motif data for a given motif and construct row.

        Args:
            m (Motif): The motif object.
            row (dict): The row containing the sequence, data, and name.

        Returns:
            dict: A dictionary containing the motif data.

        """
        strands = m.strands
        first_bp = [strands[0][0] - 1, strands[1][-1] + 1]
        second_bp = [strands[0][-1] + 1, strands[1][0] - 1]
        first_bp_id = row["sequence"][first_bp[0]] + row["sequence"][first_bp[1]]
        second_bp_id = row["sequence"][second_bp[0]] + row["sequence"][second_bp[1]]
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
            "m_first_bp": first_bp_id,  # first base pair of the motif before flanking
            "m_second_bp": second_bp_id,  # second base pair of the motif after flanking
            "num_aligned": row["num_aligned"],  # number of aligned reads
            "sn": row["sn"],  # signal to noise ratio
        }
        return data

    # step 2 ##################################################################
    def __standardize_motif_dataframe(self, df_motif):
        df_motif["strand1"] = df_motif["m_sequence"].apply(lambda x: x.split("&")[0])
        df_motif["strand2"] = df_motif["m_sequence"].apply(lambda x: x.split("&")[1])
        df_motif["orientation"] = "non-flipped"
        for i, row in df_motif.iterrows():
            if len(row["strand2"]) < len(row["strand1"]):
                continue
            if len(row["strand1"]) == len(row["strand2"]):
                if row["strand1"] < row["strand2"]:
                    continue
            df_motif.at[i, "m_sequence"] = row["strand2"] + "&" + row["strand1"]
            df_motif.at[i, "m_structure"] = flip_structure(row["m_structure"])
            df_motif.at[i, "m_strands"] = row["m_strands"][::-1]
            df_motif.at[i, "m_token"] = row["m_token"][::-1]
            df_motif.at[i, "m_data"] = row["m_data"][::-1]
            second_bp = row["m_second_bp"]
            df_motif.at[i, "m_second_bp"] = row["m_first_bp"]
            df_motif.at[i, "m_first_bp"] = second_bp
            df_motif.at[i, "orientation"] = "flipped"
        df_motif.drop(["strand1", "strand2"], axis=1, inplace=True)
        df_motif.to_json(
            "data/raw-jsons/pdb_library_1_combined_motifs_standardized.json",
            orient="records",
        )
        return df_motif


def generate_motif_and_residue_dataframe():
    df = pd.read_json("data/raw-jsons/pdb_library_1_combined_motifs.json")
    all_data = []
    res_data = []
    path = "data/pdbs"
    for motif_seq, g in df.groupby("m_sequence"):
        motif_seq_path = motif_seq.replace("&", "_")
        all_pdbs = []
        if os.path.exists(f"{path}/{motif_seq_path}"):
            pdbs = glob.glob(f"{path}/{motif_seq_path}/*.pdb")
            if len(pdbs) == 0:
                pdbs = glob.glob(f"{path}/{motif_seq_path}/*/*.pdb")
            if len(pdbs) == 0:
                raise ValueError(f"No pdbs found for {motif_seq_path}")
            all_pdbs.extend(pdbs)
        rev_motif_seq_path = motif_seq_path[::-1]
        if os.path.exists(f"{path}/{rev_motif_seq_path}"):
            pdbs = glob.glob(f"{path}/{rev_motif_seq_path}/*.pdb")
            if len(pdbs) == 0:
                pdbs = glob.glob(f"{path}/{rev_motif_seq_path}/*/*.pdb")
            if len(pdbs) == 0:
                raise ValueError(f"No pdbs found for {rev_motif_seq_path}")
            all_pdbs.extend(pdbs)
        m_data_array = np.array(g["m_data"].tolist())
        m_data_avg = np.mean(m_data_array, axis=0)
        m_data_std = np.std(m_data_array, axis=0)
        m_data_cv = m_data_std / m_data_avg
        m_data_cv[np.isnan(m_data_cv)] = 0
        pairs = []
        seqs = g["m_sequence"].iloc[0].split("&")
        m_strands = []
        for s in g["m_strands"].to_list():
            m_strands.append(s[0] + [-1] + s[1])
        if len(seqs[0]) == len(seqs[1]):
            for n1, n2 in zip(seqs[0], seqs[1][::-1]):
                pairs.append(n1 + n2)
            pairs.append("")
            for n1, n2 in zip(seqs[0], seqs[1][::-1]):
                pairs.append(n2 + n1)
        else:
            pairs = [""] * len(motif_seq_path)
        for i, (e, s) in enumerate(zip(motif_seq_path, g.iloc[0]["m_structure"])):
            if e == "_":
                continue
            if e != "A" and e != "C":
                continue
            break_char = motif_seq_path.find("_")
            r_type = "NON-WC"
            if s == "(" or s == ")":
                r_type = "WC"
            pdb_path = ""
            if len(all_pdbs) > 0:
                pdb_path = all_pdbs[0]
            data = {
                "m_sequence": motif_seq,
                "m_structure": g["m_structure"].iloc[0],
                "m_token": g["m_token"].iloc[0],
                "m_length": len(motif_seq) - 1,
                "m_strands": m_strands,
                "nuc": e,
                "r_pos": i,
                "r_data": m_data_array[:, i].tolist(),
                "r_avg": m_data_avg[i],
                "r_std": m_data_std[i],
                "r_cv": m_data_cv[i],
                "r_type": r_type,
                "pair_type": None,
                "has_pdbs": len(all_pdbs) > 0,
                "constructs": g["construct"].tolist(),
                "likely_pair": pairs[i],
                "n_pdbs": len(all_pdbs),
                "pdb_path": pdb_path,
                "pdb_pos": i + 3 if i < break_char else i + 3 + 3,
            }
            res_data.append(data)

        data = {
            "m_sequence": motif_seq,
            "m_structure": g["m_structure"].iloc[0],
            "m_token": g["m_token"].iloc[0],
            "m_data_avg": m_data_avg,
            "m_data_std": m_data_std,
            "m_data_cv": m_data_cv,
            "m_data_array": m_data_array,
            "constructs": g["construct"].tolist(),
            "m_strands": g["m_strands"].tolist(),
            "m_first_bp": g["m_first_bp"].tolist(),
            "m_second_bp": g["m_second_bp"].tolist(),
            "pdbs": all_pdbs,
            "has_pdbs": len(all_pdbs) > 0,
        }
        all_data.append(data)
    df = pd.DataFrame(all_data)
    df.to_json(
        "data/raw-jsons/pdb_library_1_combined_motifs_avg.json", orient="records"
    )
    df_res = pd.DataFrame(res_data)
    df_res.to_json(
        "data/raw-jsons/pdb_library_1_combined_motifs_res.json", orient="records"
    )


def generate_all_residue_dataframe():
    df = pd.read_json("data/raw-jsons/pdb_library_1_combined_motifs_res.json")
    all_data = []
    for i, row in df.iterrows():
        r_pos = row["r_pos"]
        for j in range(len(row["r_data"])):
            data = {
                "m_sequence": row["m_sequence"],
                "m_structure": row["m_structure"],
                "m_token": row["m_token"],
                "nuc": row["nuc"],
                "nuc_pos": row["m_strands"][j][r_pos],
                "r_pos": row["r_pos"],
                "r_data": row["r_data"][j],
                "r_pos": r_pos,
                "r_type": row["r_type"],
                "pair_type": row["likely_pair"],
                "pdb_pos": row["pdb_pos"],
                "construct": row["constructs"][j],
            }
            all_data.append(data)
    df = pd.DataFrame(all_data)
    df.to_json(
        "data/raw-jsons/pdb_library_1_combined_motifs_res_all.json", orient="records"
    )


def plot_dms_vs_nomod(df, df_nomod):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    pos = 1000
    plot_pop_avg_from_row(df.iloc[pos], ax=ax[0])
    plot_pop_avg_from_row(df_nomod.iloc[pos], ax=ax[1])
    ax[0].set_ylim(0, 0.08)
    ax[1].set_ylim(0, 0.08)
    plt.show()


def plot_1x1(df):
    df["ln_r_avg"] = np.log(df["r_avg"])
    df_non_wc = df.query("r_type == 'NON-WC'")
    df_wc = df.query("r_type == 'WC'")
    avg_non_wc = df.query("r_type == 'NON-WC'")["ln_r_avg"].mean()
    avg_wc = df.query("r_type == 'WC'")["ln_r_avg"].mean()
    df_ac = df.query(
        "m_token == '1x1' and (likely_pair == 'AC' or likely_pair == 'CA')"
    ).copy()
    df_ac["r_type"] = "AC"
    df_ga = df.query(
        "m_token == '1x1' and (likely_pair == 'GA' or likely_pair == 'AG')"
    ).copy()
    df_ga["r_type"] = "GA"
    df_aa = df.query(
        "m_token == '1x1' and (likely_pair == 'AA' or likely_pair == 'AA')"
    ).copy()
    df_aa["r_type"] = "AA"
    df_cc = df.query(
        "m_token == '1x1' and (likely_pair == 'CC' or likely_pair == 'CC')"
    ).copy()
    df_cc["r_type"] = "CC"
    df_cu = df.query(
        "m_token == '1x1' and (likely_pair == 'CU' or likely_pair == 'UC')"
    ).copy()
    df_cu["r_type"] = "CC"
    dfs = [df_wc, df_non_wc, df_ac, df_ga, df_aa, df_cc, df_cu]
    df = pd.concat(dfs)
    data = []
    seen = []
    for df1, df2 in itertools.product([df_wc, df_non_wc], dfs):
        key1 = df1.iloc[0]["r_type"] + "-" + df2.iloc[0]["r_type"]
        key2 = df2.iloc[0]["r_type"] + "-" + df1.iloc[0]["r_type"]
        if key1 in seen or key2 in seen:
            continue
        ks, p_value = ks_2samp(df1["ln_r_avg"], df2["ln_r_avg"])
        data.append([df1.iloc[0]["r_type"], df2.iloc[0]["r_type"], ks, p_value])
        seen.append(key1)
        seen.append(key2)
    df_stat = pd.DataFrame(data, columns=["r_type1", "r_type2", "ks", "p_value"])
    print(df_stat)
    exit()

    sns.violinplot(x="r_type", y="ln_r_avg", data=df)
    x1, x2 = 0, 1
    plt.plot([x1, len(dfs)], [avg_wc, avg_wc], color="red", linewidth=2, linestyle="--")
    plt.plot(
        [x2, len(dfs)],
        [avg_non_wc, avg_non_wc],
        color="red",
        linewidth=2,
        linestyle="--",
    )

    plt.show()


def plot_ac_vs_size(df):
    # plot_1x1(df)
    df_ac_1x1 = df.query(
        "m_token == '1x1' and (likely_pair == 'AC' or likely_pair == 'CA')"
    ).copy()
    df_ac_2x2 = df.query(
        "m_token == '2x2' and (likely_pair == 'AC' or likely_pair == 'CA')"
    ).copy()
    df_ac_3x3 = df.query(
        "m_token == '3x3' and (likely_pair == 'AC' or likely_pair == 'CA')"
    ).copy()
    df_ac_4x4 = df.query(
        "m_token == '4x4' and (likely_pair == 'AC' or likely_pair == 'CA')"
    ).copy()
    dfs = [df_ac_1x1, df_ac_2x2, df_ac_3x3, df_ac_4x4]
    df_ac_1x1 = df_ac_1x1[
        ["m_sequence", "m_structure", "nuc", "r_avg", "r_pos", "pdb_path"]
    ]
    df_ac_1x1.sort_values("r_avg", ascending=False, inplace=True)
    df_ac_1x1.to_csv("ac_1x1.csv", index=False)
    df_ac_4x4 = df_ac_4x4[
        ["m_sequence", "m_structure", "nuc", "r_avg", "r_pos", "pdb_path"]
    ]
    df_ac_4x4.to_csv("ac_4x4.csv", index=False)
    df = pd.concat(dfs)
    sns.violinplot(x="m_token", y="ln_r_avg", data=df)
    plt.show()


def main():
    """
    main function for script
    """
    df = df = pd.read_json("data/raw-jsons/pdb_library_1_combined.json")
    gen = GenerateMotifDataFrame()
    gen.run(df)
    exit()

    generate_all_residue_dataframe()
    exit()
    df = pd.read_json("data/raw-jsons/pdb_library_1_combined_motifs_res.json")
    df["ln_r_avg"] = np.log(df["r_avg"])
    df_a_0x1 = df.query(
        "(m_token == '0x1' or m_token == '1x0') and nuc == 'A' and r_type == 'NON-WC'"
    ).copy()
    df_a_0x1.sort_values("r_avg", ascending=False, inplace=True)
    sns.violinplot(x="m_sequence", y="ln_r_avg", data=df_a_0x1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    exit()
    df["ln_r_avg"] = np.log(df["r_avg"])
    df_wc = df.query("r_type == 'WC'")
    df_non_wc = df.query("r_type == 'NON-WC'")
    df = pd.read_json("data/raw-jsons/pdb_library_1_combined_motifs_res.json")
    df_sub = df.query("m_length == 6 and r_type == 'NON-WC'")
    plt.bar(df_sub["m_sequence"], df_sub["r_avg"])
    plt.xticks(rotation=90)
    plt.show()
    generate_motif_and_residue_dataframe()
    df = pd.read_json("data/raw-jsons/pdb_library_1_combined_motifs_res.json")
    df = df.query("has_pdbs == True and (nuc == 'A' or nuc == 'C')")
    df.to_csv("res_data.csv", index=False)


if __name__ == "__main__":
    main()
