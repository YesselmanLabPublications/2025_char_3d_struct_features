import click
import warnings
import pandas as pd
import os

from dms_3d_features.library_build import build_pdb_library_from_motif_df
from dms_3d_features.sasa import generate_sasa_dataframe
from dms_3d_features.pdb_features import (
    process_basepair_details,
    generate_distance_dataframe,
    get_all_atom_distances,
    get_all_atom_distances_with_ratio,
    get_non_canonical_atom_distances_reactivity_correlation,
    get_non_canonical_atom_distances_reactivity_ratio_correlation,
    generate_data,
    generate_data_for_modified_cond,
)
from dms_3d_features.process_motifs import (
    process_mutation_histograms_to_json,
    generate_normalized_construct_dataframes,
    generate_threshold_motif_dataframes,
    generate_motif_dataframes,
)
from dms_3d_features.logger import setup_logging, get_logger
from dms_3d_features.paths import DATA_PATH

warnings.filterwarnings(
    "ignore", message="FreeSASA: warning: Found no matches to resn 'A', typo?"
)

log = get_logger("cli")


# cli functions #################################################################


@click.group()
def cli():
    pass


@cli.command()
@click.option("--motif-csv", type=str, default=None)
@click.option("--desired-sequences", type=int, default=10)
def generate_pdb_library(motif_csv: str, desired_sequences: int):
    """
    Generate a PDB library from motif data. Run with --desired-sequences 7,500 for full library.
    """
    setup_logging()
    if motif_csv is None:
        log.info("Using default motif file")
        motif_csv = f"{DATA_PATH}/csvs/motif_sequences.csv"
    df = pd.read_csv(motif_csv)
    build_pdb_library_from_motif_df(df, desired_sequences)


@cli.command()
def generate_processed_dataframes():
    """
    Takes raw mutation histograms from RNA-MaP and generates a JSON file with motif data.
    """
    setup_logging()

    # Check paths exist
    required_paths = [
        f"{DATA_PATH}/raw-jsons/constructs",
        f"{DATA_PATH}/raw-jsons/motifs",
        f"{DATA_PATH}/raw-jsons/residues",
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise ValueError(f"Required directory {path} does not exist")

    log.info("Processing mutation histograms #########################")
    process_mutation_histograms_to_json()
    log.info("Generating normalized construct dataframes #########################")
    generate_normalized_construct_dataframes()
    log.info("Generating threshold motif dataframes #########################")
    generate_threshold_motif_dataframes()
    log.info("Generating motif dataframes #########################")
    generate_motif_dataframes()


@cli.command()
def get_basepair_details():
    """
    Get basepair details for all PDB files in the pdbs directory.
    """
    setup_logging()
    log.info("Processing basepair details #########################")
    # process_basepair_details now takes the residue dataframe and RETURNS the
    # result instead of reading/writing internally.
    df_res = pd.read_json(f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues.json")
    df_wc = process_basepair_details(df_res)
    df_wc.to_csv(f"{DATA_PATH}/csvs/wc_details.csv", index=False)


@cli.command()
def get_non_canonical_atomic_distances():
    """
    Get non-canonical atomic distances for all PDB files in the pdbs directory.
    """
    setup_logging()
    log.info("Getting all distances")
    df = generate_distance_dataframe(max_distance=1000)
    df.to_csv(f"{DATA_PATH}/pdb-features/distances_all.csv", index=False)

    # The atom-distance / correlation functions now take dataframes and return
    # dataframes, so load the pdb-residue frame and thread it through.
    df_pdb = pd.read_json(
        f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues_pdb.json"
    )

    log.info("Getting non-canonical atomic distances")
    df_atom_dist = get_all_atom_distances(df_pdb)
    df_atom_dist.to_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances.csv", index=False
    )

    log.info("Getting non-canonical atomic distances with reactivity correlation")
    df_corr = get_non_canonical_atom_distances_reactivity_correlation(
        df_atom_dist, df_pdb
    )
    df_corr.to_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_reactivity_correlation.csv",
        index=False,
    )

    log.info("Getting non-canonical atomic distances with reactivity ratio")
    df_atom_dist_ratio = get_all_atom_distances_with_ratio(df_pdb)
    df_atom_dist_ratio.to_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_with_ratio.csv",
        index=False,
    )

    log.info("Getting non-canonical atomic distances with reactivity ratio correlation")
    df_ratio_corr = get_non_canonical_atom_distances_reactivity_ratio_correlation(
        df_atom_dist_ratio, df_pdb
    )
    df_ratio_corr.to_csv(
        f"{DATA_PATH}/pdb-features/non_canonical_atom_distances_reactivity_ratio_correlation.csv",
        index=False,
    )


@cli.command()
def get_pdb_features():
    """
    Get pdb features for all PDB files in the pdbs directory.
    """
    setup_logging()
    # get all distances for different max distances
    log.info("Getting all distances")
    df = generate_distance_dataframe(max_distance=1000)
    df.to_csv(f"{DATA_PATH}/pdb-features/distances_all.csv", index=False)
    # get all sasa values for different probe radii
    log.info("Getting all sasa values")
    df_sasa = generate_sasa_dataframe()
    df_sasa.to_csv("data/pdb-features/sasa.csv", index=False)
    log.info("Getting basepair details")
    df_res = pd.read_json(f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues.json")
    df_wc = process_basepair_details(df_res)
    df_wc.to_csv(f"{DATA_PATH}/csvs/wc_details.csv", index=False)


@cli.command()
def generate_pdb_feature_data():
    """
    Run the full pdb-feature data generation for the standard library
    (and the 37C/2min modified condition), mirroring pdb_features.__main__.
    """
    setup_logging()
    log.info("Generating pdb-feature data (standard) #########################")
    generate_data()
    log.info("Generating pdb-feature data (37C/2min) #########################")
    generate_data_for_modified_cond()


if __name__ == "__main__":
    cli()