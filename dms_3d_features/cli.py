import click
import warnings
import pandas as pd
import os

from dms_3d_features.library_build import build_pdb_library_from_motif_df
from dms_3d_features.sasa import generate_sasa_dataframe
from dms_3d_features.pdb_features import (
    process_basepair_details,
    generate_distance_dataframe,
    get_non_canonical_atom_distances_reactivity_correlation,
    get_non_canonical_atom_distances_reactivity_ratio_correlation,
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
@click.option("--motif-file", type=str, default=None)
@click.option("--desired-sequences", type=int, default=10)
def generate_pdb_library(motif_file: str, desired_sequences: int):
    """
    Generate a PDB library from motif data. Run with --desired-sequences 7,500 for full library.
    """
    setup_logging()
    if motif_file is None:
        log.info("Using default motif file")
        motif_file = f"{DATA_PATH}/csvs/motif_sequences.csv"
    df = pd.read_csv(motif_file)
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
    process_basepair_details()


@cli.command()
def get_non_canonical_atomic_distances():
    """
    Get non-canonical atomic distances for all PDB files in the pdbs directory.
    """
    setup_logging()
    log.info("Getting all distances")
    df = generate_distance_dataframe(max_distance=1000)
    df.to_csv(f"{DATA_PATH}/pdb-features/distances_all.csv", index=False)
    log.info("Getting non-canonical atomic distances with reactivity correlation")
    get_non_canonical_atom_distances_reactivity_correlation()
    log.info("Getting non-canonical atomic distances with reactivity ratio correlation")
    get_non_canonical_atom_distances_reactivity_ratio_correlation()


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
    process_basepair_details()


if __name__ == "__main__":
    cli()
