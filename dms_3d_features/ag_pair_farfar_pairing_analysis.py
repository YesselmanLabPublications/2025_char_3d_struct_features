import os
import pandas as pd

# -----------------------------------
# SETTINGS
# -----------------------------------
USE_TOP10_FOLDER = True
TOP_N = 100


# -----------------------------------
# LOAD score file
# -----------------------------------
def load_sc(sc_file):
    header = None
    farfar_models = []

    with open(sc_file, "r") as f:
        for line in f:
            if not line.startswith("SCORE:"):
                continue

            parts = line.strip().split()

            if len(parts) > 1 and parts[1] == "score":
                header = parts
                continue

            if header is None:
                continue

            if len(parts) != len(header):
                continue

            farfar_models.append(dict(zip(header, parts)))

    df = pd.DataFrame(farfar_models)

    if df.empty or "description" not in df.columns or "score" not in df.columns:
        return pd.DataFrame()

    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    if "rms" in df.columns:
        df["rms"] = pd.to_numeric(df["rms"], errors="coerce")

    return df


# -----------------------------------
# GET MODEL TAGS
# -----------------------------------
def get_model_tags(condition_dir):

    # OPTION 1: top100 folder
    if USE_TOP10_FOLDER:
        top_dir = os.path.join(condition_dir, "top100")

        if os.path.exists(top_dir):
            tags = [
                d for d in os.listdir(top_dir)
                if os.path.isdir(os.path.join(top_dir, d))
            ]

            if len(tags) > 0:
                return tags, "top100"

    # OPTION 2: default.sc
    sc_file = os.path.join(condition_dir, "default.sc")

    if os.path.exists(sc_file):
        df = load_sc(sc_file)

        if not df.empty:
            n = min(TOP_N, len(df))
            df_top = df.nsmallest(n, "score")
            return df_top["description"].tolist(), "default.sc"

    return [], None


# -----------------------------------
# RMSD stats
# -----------------------------------
def get_rmsd_stats(condition_dir):
    sc_file = os.path.join(condition_dir, "default.sc")

    if not os.path.exists(sc_file):
        return None

    df = load_sc(sc_file)

    if df.empty or "rms" not in df.columns:
        return None

    df = df.dropna(subset=["rms", "score"])
    if df.empty:
        return None

    n = min(TOP_N, len(df))
    df_top = df.nsmallest(n, "score")

    return {
        "mean": df_top["rms"].mean(),
        "median": df_top["rms"].median(),
        "min": df_top["rms"].min(),
        "max": df_top["rms"].max()
    }


# -----------------------------------
# DSSR pair extraction
# -----------------------------------
def extract_pairs(out_file):
    pairs = []

    with open(out_file, "r") as f:
        lines = f.readlines()

    in_section = False
    passed_header = False

    for line in lines:
        s = line.strip()

        if s.startswith("List of") and "base pairs" in s:
            in_section = True
            passed_header = False
            continue

        if not in_section:
            continue

        if not passed_header:
            if "nt1" in line and "nt2" in line and "bp" in line and "LW" in line:
                passed_header = True
            continue

        if s == "":
            break

        parts = line.split()
        if len(parts) < 8:
            continue

        nt1_raw, nt2_raw, bp, lw = parts[1], parts[2], parts[3], parts[6]

        nt1 = nt1_raw[-2:]
        nt2 = nt2_raw[-2:]

        if bp in {"G-A", "A-G", "A+G", "G+A"}:
            key = tuple(sorted([nt1, nt2]))
            pairs.append({"key": key, "LW": lw})

    return pairs


# -----------------------------------
# ANALYZE ONE CONDITION
# -----------------------------------
def analyze_condition(motif, motif_root, condition, suffix):

    condition_dir = os.path.join(motif_root, condition)

    if not os.path.exists(condition_dir):
        print(f"Missing {condition} for {motif}")
        return None

    # ✅ crystal is in motif root
    crystal_out = os.path.join(motif_root, f"{motif.lower()}_gc_added.out")

    if not os.path.exists(crystal_out):
        print(f"Skipping {motif}: missing crystal")
        return None

    crystal_pairs = extract_pairs(crystal_out)

    if len(crystal_pairs) == 0:
        print(f"Skipping {motif}: no AG pairs")
        return None

    model_tags, source = get_model_tags(condition_dir)
    rmsd_stats = get_rmsd_stats(condition_dir)

    if len(model_tags) == 0:
        print(f"Skipping {motif} ({condition}): no_models")
        return None

    n_models = len(model_tags)
    total_possible = len(crystal_pairs) * n_models
    total_correct = 0

    for tag in model_tags:

        model_out = os.path.join(condition_dir, "top100", tag, f"{tag}.out")

        if not os.path.exists(model_out):
            model_out = os.path.join(condition_dir, "pdb", tag, f"{tag}.out")

        if not os.path.exists(model_out):
            continue

        model_pairs = extract_pairs(model_out)
        model_dict = {p["key"]: p["LW"] for p in model_pairs}

        for cp in crystal_pairs:
            if cp["key"] in model_dict and cp["LW"] == model_dict[cp["key"]]:
                total_correct += 1

    percent = (total_correct / total_possible) * 100 if total_possible > 0 else 0

    row = {
        "motif": motif,
        f"total_correct_{suffix}": total_correct,
        f"percent_true_{suffix}": percent,
        "n_crystal_pairs": len(crystal_pairs),
        "n_models": n_models,
        "total_possible": total_possible,
        "model_source": source
    }

    if rmsd_stats:
        row.update({
            f"rmsd_mean_{suffix}": rmsd_stats["mean"],
            f"rmsd_min_{suffix}": rmsd_stats["min"],
            f"rmsd_max_{suffix}": rmsd_stats["max"]
        })

    print(f"{motif} | {condition} → {percent:.2f}%")

    return row


# -----------------------------------
# MAIN RUNNER
# -----------------------------------
def run_train_test(root_dir, output_file):

    with_rows = []
    no_rows = []

    for motif in sorted(os.listdir(root_dir)):
        motif_root = os.path.join(root_dir, motif)

        if not os.path.isdir(motif_root):
            continue

        print(f"\nProcessing motif: {motif}")

        row_with = analyze_condition(motif, motif_root, "with_constraints", "with_cst")
        row_no   = analyze_condition(motif, motif_root, "without_constraints", "no_cst")

        if row_with:
            with_rows.append(row_with)

        if row_no:
            no_rows.append(row_no)

    df_with = pd.DataFrame(with_rows)
    df_no   = pd.DataFrame(no_rows)

    df_summary = pd.merge(df_with, df_no, on="motif", how="outer")

    # clean duplicate columns
    for col in ["n_crystal_pairs", "n_models", "total_possible", "model_source"]:
        x_col = f"{col}_x"
        y_col = f"{col}_y"

        if x_col in df_summary.columns and y_col in df_summary.columns:
            df_summary[col] = df_summary[x_col].combine_first(df_summary[y_col])
            df_summary.drop([x_col, y_col], axis=1, inplace=True)

    if "percent_true_with_cst" in df_summary.columns and "percent_true_no_cst" in df_summary.columns:
        df_summary["percent_true_pair_diff"] = (
            df_summary["percent_true_with_cst"] -
            df_summary["percent_true_no_cst"]
        )

    df_summary = df_summary.sort_values("motif").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_summary.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")


# -----------------------------------
# RUN
# -----------------------------------
run_train_test(
    root_dir="farfar-models/ag_pairs_train",
    output_file="farfar-models/csvs/ag_pair_train_analysis.csv"
)

run_train_test(
    root_dir="farfar-models/ag_pairs_test",
    output_file="farfar-models/csvs/ag_pair_test_analysis.csv"
)