import os
import json
import argparse
import numpy as np
import pandas as pd

# =========================================================
# SETTINGS
# =========================================================
EXCLUDE_HYDROGEN = True

BASE_ATOMS = {
    "A": {"N1","C2","N3","C4","C5","C6","N6","N7","C8","N9"},
    "G": {"N1","C2","N2","N3","C4","C5","C6","O6","N7","C8","N9"}
}

# =========================================================
# LOAD DISTANCE CSV
# =========================================================
def load_distance_data(csv_file):
    df = pd.read_csv(csv_file)
    df["m_sequence"] = df["m_sequence"].astype(str)
    df["pdb_r_pos"] = df["pdb_r_pos"].astype(str)
    return df

# =========================================================
# HELPERS
# =========================================================
def is_valid_outfile(filename):
    return filename.endswith(".out") and not filename.startswith(("._", "slurm-", "dssr_", "."))

def parse_residue(nt):
    nt = nt.strip()
    if not nt:
        return None
    if nt[0].isalpha() and nt[1:].isdigit():
        return nt[1:]
    if "." in nt:
        parts = nt.split(".")
        if len(parts) == 2:
            digits = ''.join(filter(str.isdigit, parts[1]))
            return digits if digits else None
    return None

def find_crystal_outfile_and_pdb(motif_dir, motif):
    outs = [f for f in os.listdir(motif_dir) if is_valid_outfile(f)]
    for o in outs:
        base = os.path.splitext(o)[0].lower()
        pdb = os.path.join(motif_dir, f"{base}.pdb")
        if "gc_added" in base:
            continue
        if os.path.exists(pdb):
            return os.path.join(motif_dir, o), pdb
    return None, None

def find_model_pdb(root, name):
    candidates = [
        os.path.join(root, name, f"{name}.pdb"),
        os.path.join(root, f"{name}.pdb")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# =========================================================
# DSSR PAIRS
# =========================================================
def extract_pairs(out_file):
    pairs = []
    with open(out_file) as f:
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
            if "nt1" in line and "nt2" in line:
                passed_header = True
            continue
        if s == "":
            break
        parts = line.split()
        if len(parts) < 8:
            continue
        nt1_raw, nt2_raw, bp, lw = parts[1], parts[2], parts[3], parts[6]
        nt1 = parse_residue(nt1_raw)
        nt2 = parse_residue(nt2_raw)
        if nt1 is None or nt2 is None:
            continue
        if bp in {"G-A", "A-G", "A+G", "G+A"}:
            pairs.append({
                "nt1": nt1,
                "nt2": nt2,
                "LW": lw,
                "res1_type": bp[0],
                "res2_type": bp[-1]
            })
    return pairs

# =========================================================
# COORDS
# =========================================================
def get_residue_coords(pdb_file, residues, residue_types):
    # Returns {res_id: {atom_name: coords}} — first occurrence of each atom wins (handles alt conformations)
    coords = {r: {} for r in residues}
    res_type_map = dict(zip(residues, residue_types))

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            res_id = line[22:26].strip()
            if res_id not in residues:
                continue
            atom_name = line[12:16].strip()
            if EXCLUDE_HYDROGEN and atom_name.startswith("H"):
                continue
            if atom_name not in BASE_ATOMS.get(res_type_map[res_id], set()):
                continue
            if atom_name in coords[res_id]:
                continue  # skip duplicate / alternate conformation
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            coords[res_id][atom_name] = np.array([x, y, z])
    return coords

# =========================================================
# P-P DISTANCE
# =========================================================
def compute_pp_distance(pdb_file, r1, r2):
    coords = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            res = line[22:26].strip()
            atom = line[12:16].strip()
            if atom == "P" and res in {r1, r2}:
                coords[res] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if r1 not in coords or r2 not in coords:
        return None
    return float(np.linalg.norm(coords[r1] - coords[r2]))

# =========================================================
# KABSCH RMSD
# =========================================================
def kabsch_rotation(P, Q):
    """Find rotation matrix R that best superimposes Q onto P.
    Both P and Q must already be centred (mean subtracted).
    H = P.T @ Q  (reference.T @ mobile) is the correct cross-covariance.
    R = Vt.T @ U.T maps Q -> P.
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Reflection correction: ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R

def kabsch_rmsd(P, Q):
    """Standard Kabsch RMSD — superimposes Q onto P over all atoms.
    P = crystal (reference), Q = model (mobile), both shape (N, 3).
    """
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    R  = kabsch_rotation(Pc, Qc)
    return np.sqrt(np.mean(np.sum((Qc @ R - Pc) ** 2, axis=1)))

def compute_whole_motif_rmsd(pdb_crystal, pdb_model, debug=False):
    def get_all_coords(pdb_file):
        coords = {}
        with open(pdb_file) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                res  = line[22:26].strip()
                atom = line[12:16].strip()
                if EXCLUDE_HYDROGEN and atom.startswith("H"):
                    continue
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                coords.setdefault(res, {})
                if atom not in coords[res]:
                    coords[res][atom] = xyz
        return coords

    coords_c = get_all_coords(pdb_crystal)
    coords_m = get_all_coords(pdb_model)

    if debug:
        print(f"  Crystal residues: {sorted(coords_c.keys())}")
        print(f"  Model residues:   {sorted(coords_m.keys())}")

    P_all, Q_all = [], []
    for res in coords_c:
        if res not in coords_m:
            if debug:
                print(f"  WARNING: residue {res} missing from model")
            continue
        shared_atoms = sorted(coords_c[res].keys() & coords_m[res].keys())
        if len(shared_atoms) < 3:
            continue
        for atom in shared_atoms:
            P_all.append(coords_c[res][atom])
            Q_all.append(coords_m[res][atom])

    if len(P_all) < 3:
        return None

    P = np.array(P_all)
    Q = np.array(Q_all)

    if debug:
        # Per-atom distances after superposition to spot bad pairs
        Pc = P - P.mean(0)
        Qc = Q - Q.mean(0)
        R  = kabsch_rotation(Pc, Qc)
        per_atom = np.sqrt(np.sum((Qc @ R - Pc) ** 2, axis=1))
        print(f"  Per-atom distances (sorted): {np.sort(per_atom).round(2)}")
        print(f"  Raw RMSD ({len(P)} atoms): {np.sqrt(np.mean(per_atom**2)):.3f}")

    return round(float(kabsch_rmsd(P, Q)), 3)

# =========================================================
# SCORE FILE
# =========================================================
def load_sc(sc_file):
    header = None
    data = []
    with open(sc_file) as f:
        for line in f:
            if not line.startswith("SCORE:"):
                continue
            parts = line.split()
            if len(parts) > 1 and parts[1] == "score":
                header = parts[1:]
                continue
            if header is None:
                continue
            row = parts[1:]
            if len(row) != len(header):
                continue
            data.append(dict(zip(header, row)))

    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["rms"] = pd.to_numeric(df["rms"], errors="coerce")
    return df.dropna(subset=["score", "rms", "description"])

def get_top_models(sc_file, frac=0.10):
    df = load_sc(sc_file)
    if df.empty:
        return df
    df = df.sort_values("score")
    return df.head(max(1, int(np.ceil(len(df) * frac))))

# =========================================================
# PAIR DATA INIT
# =========================================================
def init_pair_entry(pair):
    return {
        "nt1":             pair["nt1"],
        "nt2":             pair["nt2"],
        "LW":              pair["LW"],
        "rmsd":            [],
        "model_names":     [],
        "model_distances": [],
    }

def lookup_distances(pair, motif, distance_df):
    a_res = pair["nt1"] if pair["res1_type"] == "A" else pair["nt2"]
    match = distance_df[
        (distance_df["m_sequence"] == motif) &
        (distance_df["pdb_r_pos"] == str(a_res))
    ]
    crystal = round(float(match.iloc[0]["distance"]), 3)            if not match.empty else None
    pred    = round(float(match.iloc[0]["predicted_distance"]), 3)  if not match.empty else None
    return crystal, pred

# =========================================================
# RMSD + DISTANCE for one model/pair combination
# =========================================================
def build_point_clouds(coords_c, coords_m, res):
    """Return aligned (P, Q) atom coordinate arrays for one residue.
    Atoms are sorted so P[i] and Q[i] always correspond to the same atom name.
    Returns empty lists if fewer than 3 shared atoms (geometrically insufficient).
    """
    shared = sorted(coords_c[res].keys() & coords_m[res].keys())
    if len(shared) < 3:
        return [], []
    P = [coords_c[res][a] for a in shared]
    Q = [coords_m[res][a] for a in shared]
    return P, Q

def compute_pair_rmsd_and_dist(pdb_crystal, pdb_model, pair):
    res1, res2 = pair["nt1"], pair["nt2"]
    types = [pair["res1_type"], pair["res2_type"]]

    coords_c = get_residue_coords(pdb_crystal, [res1, res2], types)
    coords_m = get_residue_coords(pdb_model,   [res1, res2], types)

    # Check both residues found in both structures
    if any(len(coords_c[r]) == 0 for r in [res1, res2]):
        print(f"WARNING: missing crystal atoms for {res1}/{res2} — found: { {r: list(coords_c[r].keys()) for r in [res1,res2]} }")
        return None, None
    if any(len(coords_m[r]) == 0 for r in [res1, res2]):
        print(f"WARNING: missing model atoms for {res1}/{res2} in {pdb_model} — found: { {r: list(coords_m[r].keys()) for r in [res1,res2]} }")
        return None, None

    # Check atom sets match between crystal and model
    for r in [res1, res2]:
        c_atoms = set(coords_c[r].keys())
        m_atoms = set(coords_m[r].keys())
        if c_atoms != m_atoms:
            print(f"WARNING: atom mismatch for res {r}: crystal={sorted(c_atoms)} model={sorted(m_atoms)}")

    # Combined rigid-body fit with outlier rejection — matches PyMOL ExecutiveRMS
    P1, Q1 = build_point_clouds(coords_c, coords_m, res1)
    P2, Q2 = build_point_clouds(coords_c, coords_m, res2)

    if not P1 or not P2:
        return None, None

    P   = np.array(P1 + P2)
    Q   = np.array(Q1 + Q2)
    rms = kabsch_rmsd(P, Q)
    dist = compute_pp_distance(pdb_model, res1, res2)
    return round(float(rms), 3), round(float(dist), 3) if dist is not None else None

# =========================================================
# ACCUMULATE pair RMSD results across all top models
# =========================================================
def accumulate_pair_results(df_top, pdb_top, pdb_crystal, valid_pairs, pair_data):
    """Fills pair_data in-place. valid_pairs is a list of (label, pair) tuples."""
    for _, row in df_top.iterrows():
        name      = str(row["description"])
        pdb_model = find_model_pdb(pdb_top, name)
        if pdb_model is None:
            continue
        _fill_pairs_for_model(pdb_crystal, pdb_model, name, valid_pairs, pair_data)

def _fill_pairs_for_model(pdb_crystal, pdb_model, name, valid_pairs, pair_data):
    for label, pair in valid_pairs:
        rms, dist = compute_pair_rmsd_and_dist(pdb_crystal, pdb_model, pair)
        if rms is None:
            continue
        pair_data[label]["rmsd"].append(rms)
        pair_data[label]["model_names"].append(name)
        pair_data[label]["model_distances"].append(dist)

# =========================================================
# PROCESS ONE CONDITION (with_cst or no_cst)
# =========================================================
def process_condition(condition_dir, crystal_dir, motif, distance_df):
    sc_file = os.path.join(condition_dir, "default.sc")
    pdb_top = os.path.join(condition_dir, "top100")

    if not os.path.exists(sc_file):
        return None

    out_file, pdb_crystal = find_crystal_outfile_and_pdb(crystal_dir, motif)
    if out_file is None:
        return None

    df_all = load_sc(sc_file)
    df_top = get_top_models(sc_file)
    pairs  = extract_pairs(out_file)

    all_models = {
        "model_names": df_all["description"].astype(str).tolist(),
        "score":       [round(float(x), 3) for x in df_all["score"]],
        "rmsd":        [round(float(x), 3) for x in df_all["rms"]]
    }

    whole_rmsd_list = []
    model_names = []
    scores = []

    for _, row in df_top.iterrows():
        name = str(row["description"])
        pdb_model = find_model_pdb(pdb_top, name)

        if pdb_model is None:
            continue

        rms = compute_whole_motif_rmsd(pdb_crystal, pdb_model)
        if rms is None:
            continue

        whole_rmsd_list.append(rms)
        model_names.append(name)
        scores.append(round(float(row["score"]), 3))
        
    top_models = {
        "model_names": model_names,
        "score": scores,
        "rmsd": whole_rmsd_list,
        "mean_rmsd": round(float(np.mean(whole_rmsd_list)), 3) if whole_rmsd_list else None
    }

    crystal_distances   = {}
    predicted_distances = {}
    pair_data           = {}
    valid_pairs         = []   # list of (label, pair) tuples
    for i, p in enumerate(pairs, start=1):
        crystal, pred = lookup_distances(p, motif, distance_df)
        if crystal is None:
            continue  # skip pairs not in the CSV
        label = f"pair{i}"
        pair_data[label]           = init_pair_entry(p)
        crystal_distances[label]   = crystal
        predicted_distances[label] = pred
        valid_pairs.append((label, p))

    accumulate_pair_results(df_top, pdb_top, pdb_crystal, valid_pairs, pair_data)

    for entry in pair_data.values():
        vals = entry["rmsd"]
        entry["mean_rmsd"] = round(float(np.mean(vals)), 3) if vals else None

    return {
        "crystal_distances":   crystal_distances,
        "predicted_distances": predicted_distances,
        "all_models":          all_models,
        "top100_models":       top_models,
        **pair_data
    }

# =========================================================
# MAIN
# =========================================================
def main(root, output_json, distance_df):
    motifs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ])

    out = []
    for m in motifs:
        motif_dir   = os.path.join(root, m)
        with_dir    = os.path.join(motif_dir, "with_constraints")
        without_dir = os.path.join(motif_dir, "without_constraints")

        with_result = process_condition(with_dir,    motif_dir, m, distance_df) if os.path.isdir(with_dir)    else None
        no_result   = process_condition(without_dir, motif_dir, m, distance_df) if os.path.isdir(without_dir) else None

        dist_source = with_result or no_result
        entry = {
            "motif":               m,
            "crystal_distances":   dist_source["crystal_distances"]   if dist_source else None,
            "predicted_distances": dist_source["predicted_distances"] if dist_source else None,
            "with_constraint":     {k: v for k, v in with_result.items() if k not in ("crystal_distances", "predicted_distances")} if with_result else None,
            "no_constraint":       {k: v for k, v in no_result.items()   if k not in ("crystal_distances", "predicted_distances")} if no_result   else None,
        }
        out.append(entry)

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print("DONE")

# =========================================================
# ENTRY POINTS
# =========================================================
def run_train():
    root     = "farfar-models/ag_pairs_train"
    output   = "farfar-models/summary/ag_pairs_train_summary.json"
    dist_csv = "farfar-models/csvs/train_ag_actual_vs_predicted_distance.csv"
    main(root, output, load_distance_data(dist_csv))
 
def run_test():
    root     = "farfar-models/ag_pairs_test"
    output   = "farfar-models/summary/ag_pairs_test_summary.json"
    dist_csv = "farfar-models/csvs/test_ag_actual_vs_predicted_distance.csv"
    main(root, output, load_distance_data(dist_csv))
 
if __name__ == "__main__":
    run_train()
    run_test()