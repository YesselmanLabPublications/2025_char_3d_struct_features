import pandas as pd
from collections import defaultdict
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from dms_3d_features.stats import r2

from rna_motif_library.util import (  # pyright: ignore[reportMissingImports]
    add_motif_indentifier_columns,
    parse_motif_indentifier,
)
from rna_motif_library.motif import (  # pyright: ignore[reportMissingImports]
    get_cached_motifs,
)

DATA_PATH = "data"


def save_dict_to_pickle(dictionary, filename):
    """
    Save a dictionary to a pickle file.

    Args:
        dictionary (dict): The dictionary to save.
        filename (str): The path to the pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump(dictionary, f)


def load_dict_from_pickle(filename):
    """
    Load a dictionary from a pickle file.

    Args:
        filename (str): The path to the pickle file.

    Returns:
        dict: The loaded dictionary.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def get_sequences_without_pdbs():
    df = pd.read_json(f"{DATA_PATH}/raw-jsons/motifs/pdb_library_1_motifs_avg.json")
    df["has_pdbs"] = df["pdbs"].apply(lambda x: 0 if len(x) == 0 else 1)
    df = df.query("has_pdbs == 0").copy().reset_index(drop=True)
    return df


def assign_new_pdbs_to_sequences():
    df = get_sequences_without_pdbs()
    df_motifs = pd.read_csv("scripts/all_motifs_issues.csv")
    df_motifs = df_motifs.query("motif_type == 'TWOWAY'")
    df_motifs = (
        df_motifs.query(
            "flanking_helices == 1 and contains_helix == 0 and has_missing_residues == 0"
        )
        .copy()
        .reset_index(drop=True)
    )
    df_motifs = add_motif_indentifier_columns(df_motifs, "motif_name")
    new_motif_sequences = defaultdict(list)
    new_motif_rev_sequences = defaultdict(list)
    for i, row in df_motifs.iterrows():
        seq = row["msequence"].replace("-", "&")
        rev_seq = "&".join(seq.split("&")[::-1])
        new_motif_sequences[seq].append(row["motif_name"])
        new_motif_rev_sequences[rev_seq].append(row["motif_name"])
    pdbs = []
    rev_pdbs = []
    has_pdbs = []
    for i, row in df.iterrows():
        seq = row["m_sequence"]
        has_pdb = 0
        if seq in new_motif_sequences:
            pdbs.append(new_motif_sequences[seq])
            has_pdb = 1
        else:
            pdbs.append([])
        if seq in new_motif_rev_sequences:
            rev_pdbs.append(new_motif_rev_sequences[seq])
            has_pdb = 1
        else:
            rev_pdbs.append([])
        has_pdbs.append(has_pdb)
    df["pdbs"] = pdbs
    df["rev_pdbs"] = rev_pdbs
    df["has_pdbs"] = has_pdbs
    df = df.query("has_pdbs == 1").copy().reset_index(drop=True)
    return df


def get_new_motifs_with_pdbs():
    df = pd.read_json("scripts/new_pdbs.json")
    data = []
    for i, row in df.iterrows():
        for motif_name in row["pdbs"]:
            pdb_id = parse_motif_indentifier(motif_name)[-1]
            data.append(
                {
                    "m_sequence": row["m_sequence"],
                    "motif_name": motif_name,
                    "rev": False,
                    "pdb_id": pdb_id,
                }
            )
        for motif_name in row["rev_pdbs"]:
            pdb_id = parse_motif_indentifier(motif_name)[-1]
            data.append(
                {
                    "m_sequence": row["m_sequence"],
                    "motif_name": motif_name,
                    "rev": True,
                    "pdb_id": pdb_id,
                }
            )
    df_pdbs = pd.DataFrame(data)
    motifs = defaultdict(list)
    rev_motifs = defaultdict(list)
    for pdb_id, g in df_pdbs.groupby("pdb_id"):
        print(pdb_id)
        pdb_motifs = get_cached_motifs(pdb_id)
        motifs_by_name = {m.name: m for m in pdb_motifs}
        for i, row in g.iterrows():
            motif = motifs_by_name[row["motif_name"]]
            if row["rev"]:
                rev_motifs[row["m_sequence"]].append(motif)
            else:
                motifs[row["m_sequence"]].append(motif)
    save_dict_to_pickle(motifs, "scripts/motifs.pkl")
    save_dict_to_pickle(rev_motifs, "scripts/rev_motifs.pkl")


def get_other_pair_index(i, m_sequence_length):
    if m_sequence_length == 7:
        if i == 1:
            return 4
        if i == 4:
            return 1
    elif m_sequence_length == 9:
        if i == 1:
            return 6
        elif i == 2:
            return 5
        elif i == 6:
            return 1
        elif i == 5:
            return 2
    else:
        raise ValueError(f"Invalid sequence length: {m_sequence_length}")


def compute_distance(arr1, arr2):
    """
    Compute the Euclidean distance between two numpy arrays of length 3.
    """
    diff = arr1 - arr2
    return np.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)


def get_atom_distances(motif, reactivity, pair_index, other_pair_index):
    residues = motif.get_residues()
    res_1 = residues[pair_index]
    res_2 = residues[other_pair_index]
    data = []
    for i, (atom_1, coord_1) in enumerate(zip(res_1.atom_names, res_1.coords)):
        if atom_1.startswith("H"):
            continue
        for j, (atom_2, coord_2) in enumerate(zip(res_2.atom_names, res_2.coords)):
            if atom_2.startswith("H"):
                continue
            data.append(
                {
                    "atom_1": atom_1,
                    "atom_2": atom_2,
                    "distance": round(compute_distance(coord_1, coord_2), 2),
                    "reactivity": round(reactivity[pair_index], 4),
                    "reactivity_ratio": round(
                        reactivity[pair_index] / reactivity[other_pair_index], 4
                    ),
                }
            )
    return pd.DataFrame(data)


def get_atomic_distances():
    motifs = load_dict_from_pickle("scripts/motifs.pkl")
    rev_motifs = load_dict_from_pickle("scripts/rev_motifs.pkl")
    df = pd.read_json("scripts/new_pdbs.json")
    all_dfs = []
    for i, row in df.iterrows():
        row_motifs = motifs[row["m_sequence"]]
        rev_seq = "&".join(row["m_sequence"].split("&")[::-1])
        motif_names = [m.name for m in row_motifs]
        row_rev_motifs = []
        for m in rev_motifs[rev_seq]:
            if m.name not in motif_names:
                row_rev_motifs.append(m)
        # row_motifs = row_motifs + row_rev_motifs
        if len(row_rev_motifs) > 0:
            print(row_rev_motifs[0].sequence)
            print(row["m_sequence"])
            exit()
        pairs = list(row["pairs"])
        # Remove empty string pairs if present
        pairs = [p for p in pairs if p != ""]
        reactivity = list(row["m_data_avg"])
        # Remove 0.0 reactivity values if present
        reactivity = [r for r in reactivity if r != 0.0]
        if len(reactivity) != len(pairs):
            print("not equal", row["m_sequence"], len(reactivity), pairs)

        # Find all key pairs (i.e., not AU, UA, GC, CG, GU, UG)
        key_pairs = {
            "AG",
            "AC",
            "AA",
            "CA",
            "CC",
            "CU",
        }
        non_wc_indices = [
            idx for idx, pair in enumerate(pairs) if pair in key_pairs and pair != ""
        ]

        for idx in non_wc_indices:
            pair = pairs[idx]
            # Get the reverse pair string (e.g., AG <-> GA)
            reverse_pair = pair[::-1]

            other_pair_index = get_other_pair_index(idx, len(row["m_sequence"]))
            if not pairs[other_pair_index] == reverse_pair:
                print(pairs)
                print(row["m_sequence"], idx, other_pair_index)
                print(pairs[idx], pairs[other_pair_index])
                print("failed")
                exit()
            for m in row_motifs:
                if m.sequence.replace("-", "&") != row["m_sequence"]:
                    print("sequence mismatch", m.sequence, row["m_sequence"])
                    exit()
                df_dist = get_atom_distances(m, reactivity, idx, other_pair_index)
                df_dist["motif_name"] = m.name
                df_dist["m_sequence"] = row["m_sequence"]
                df_dist["pair"] = pair
                all_dfs.append(df_dist)
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv("scripts/non_wc_distances.csv", index=False)


def get_motif_hbonds():
    motifs = load_dict_from_pickle("scripts/motifs.pkl")
    motif_names = []
    for k, v in motifs.items():
        for m in v:
            motif_names.append(m.name)
    motifs = load_dict_from_pickle("scripts/rev_motifs.pkl")
    for k, v in motifs.items():
        for m in v:
            motif_names.append(m.name)
    motif_names = list(set(motif_names))
    df = pd.DataFrame(motif_names, columns=["motif_name"])
    df = add_motif_indentifier_columns(df, "motif_name")
    path = "/Users/jyesselman2/Library/CloudStorage/Dropbox/2_code/python/rna_motif_library/data/dataframes/motifs"
    motif_hbonds = {}
    for pdb_id, g in df.groupby("pdb_id"):
        print(pdb_id)
        try:
            df_motifs = pd.read_json(f"{path}/{pdb_id}.json")
        except:
            print(f"No motifs for {pdb_id}, {len(g)} motifs")
            continue
        for i, row in g.iterrows():
            motif_name = row["motif_name"]
            motif_row = df_motifs.query(f"motif_id == '{motif_name}'").iloc[0]
            motif_hbonds[motif_name] = {
                "is_isolatable": motif_row["is_isolatable"],
                "num_external_hbonds": motif_row["num_external_hbonds"],
                "num_external_phos_hbonds": motif_row["num_external_phos_hbonds"],
                "num_external_sugar_hbonds": motif_row["num_external_sugar_hbonds"],
                "num_external_base_hbonds": motif_row["num_external_base_hbonds"],
            }
    save_dict_to_pickle(motif_hbonds, "scripts/motif_hbonds.pkl")


def motifs_to_cif(unique, is_isolatable):
    motifs = load_dict_from_pickle("scripts/motifs.pkl")
    rev_motifs = load_dict_from_pickle("scripts/rev_motifs.pkl")
    for k, v in motifs.items():
        name = k.replace("&", "_")
        keep = []
        for m in v:
            if m.name in unique and m.name in is_isolatable:
                keep.append(m)
        if len(keep) == 0:
            continue
        os.makedirs(f"scripts/new_pdbs/{name}", exist_ok=True)
        for m in keep:
            m.to_cif(f"scripts/new_pdbs/{name}/{m.name}.cif")
    for k, v in rev_motifs.items():
        name = k.replace("&", "_")
        keep = []
        for m in v:
            if m.name in unique and m.name in is_isolatable:
                keep.append(m)
        if len(keep) == 0:
            continue
        os.makedirs(f"scripts/new_pdbs/{name}", exist_ok=True)
        for m in keep:
            m.to_cif(f"scripts/new_pdbs/{name}/{m.name}.cif")


def get_table_data():
    df_exclude = pd.read_csv("scripts/exclude.csv")
    exclude = {row["motif_name"]: 1 for i, row in df_exclude.iterrows()}
    motifs = load_dict_from_pickle("scripts/motifs.pkl")
    rev_motifs = load_dict_from_pickle("scripts/rev_motifs.pkl")
    table_data = []
    for k, v in motifs.items():
        for m in v:
            res_strs = []
            for res in m.get_residues():
                res_strs.append(res.chain_id + str(res.num))
            res_str = ";".join(res_strs)
            used = 0 if m.name in exclude else 1
            table_data.append(
                {
                    "sequence" : k,
                    "name" : m.name,
                    "residues" : res_str,
                    "used" : used,
                }
            )
    for k, v in rev_motifs.items():
        for m in v:
            res_strs = []
            for res in m.get_residues():
                res_strs.append(res.chain_id + str(res.num))
            res_str = ";".join(res_strs)
            used = 0 if m.name in exclude else 1
            table_data.append(
                {
                    "sequence" : k,
                    "name" : m.name,
                    "residues" : res_str,
                    "used" : used,
                }
            )
    df_table = pd.DataFrame(table_data)
    df_table.to_csv("scripts/table_data.csv", index=False)

def main():
    get_table_data()
    exit()
    # get_atomic_distances()
    df_dist = pd.read_csv("scripts/non_wc_distances.csv")
    path = "/Users/jyesselman2/Library/CloudStorage/Dropbox/2_code/python/rna_motif_library/data/summaries/non_redundant_motifs.csv"
    df_unique = pd.read_csv(path)
    unique = {row["motif_name"]: 1 for i, row in df_unique.iterrows()}
    hbonds_dict = load_dict_from_pickle("scripts/motif_hbonds.pkl")
    is_isolatable = {}
    for k, v in hbonds_dict.items():
        if v["is_isolatable"]:
            is_isolatable[k] = 1
    df_exclude = pd.read_csv("scripts/exclude.csv")
    exclude = {row["motif_name"]: 1 for i, row in df_exclude.iterrows()}
    data = []
    avg_data = []
    table_data = []
    for (atom_1, atom_2, pair), g in df_dist.groupby(["atom_1", "atom_2", "pair"]):
        g = g.query("motif_name in @unique")
        g = g.query("motif_name in @is_isolatable")
        g = g.query("motif_name not in @exclude")
        g = g.query("reactivity != 0")
        pair = pair[0] + "-" + pair[1]
        avg_reactivites = []
        avg_distances = []
        avg_reactivity_ratio = []
        for seq, g_seq in g.groupby("m_sequence"):
            avg_reactivites.append(g_seq["reactivity"].mean())
            avg_distances.append(g_seq["distance"].mean())
            avg_reactivity_ratio.append(g_seq["reactivity_ratio"].mean())
            avg_data.append(
                [
                    atom_1,
                    atom_2,
                    pair,
                    seq,
                    g_seq["reactivity"].mean(),
                    g_seq["distance"].mean(),
                    g_seq["reactivity_ratio"].mean(),
                ]
            )
        data.append(
            [
                atom_1,
                atom_2,
                pair,
                len(g),
                round(r2(np.log(avg_reactivites), avg_distances), 3),
                round(r2(np.log(g["reactivity"]), g["distance"]), 3),
                min(avg_distances),
                max(avg_distances),
                round(r2(np.log(avg_reactivity_ratio), avg_distances), 3),
                round(r2(np.log(g["reactivity_ratio"]), g["distance"]), 3),
            ]
        )
        # if atom_1 == "O3'" and atom_2 == "C2'" and pair == "CA":
        #    plt.scatter(avg_distances, np.log(avg_reactivites))
        #    plt.show()
    df = pd.DataFrame(
        data,
        columns=[
            "atom_1",
            "atom_2",
            "pair",
            "count",
            "r2_avg",
            "r2",
            "min_distance",
            "max_distance",
            "r2_avg_ratio",
            "r2_ratio",
        ],
    )
    df.sort_values(by="r2", ascending=False, inplace=True)
    df.to_csv("scripts/non_wc_distances_reactivity_correlation.csv", index=False)
    df_avg = pd.DataFrame(
        avg_data,
        columns=[
            "atom_1",
            "atom_2",
            "pair",
            "m_sequence",
            "reactivity",
            "distance",
            "reactivity_ratio",
        ],
    )
    df_avg.to_csv("scripts/non_wc_distances_avg_data.csv", index=False)


if __name__ == "__main__":
    main()
