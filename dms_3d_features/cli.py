import click
import warnings
import pandas as pd
import os

from dms_3d_features.sasa import generate_sasa_dataframe
from dms_3d_features.hbond import calculate_hbond_strength_all
from dms_3d_features.pdb_features import (
    DSSRTorsionFileProcessor,
    process_basepair_details,
    generate_distance_dataframe,
)
from dms_3d_features.process_motifs import (
    process_mutation_histograms_to_json,
    GenerateMotifDataFrame,
    GenerateResidueDataFrame,
    generate_pdb_residue_dataframe,
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
def generate_motif_data():
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

    process_mutation_histograms_to_json()
    construct_file = f"{DATA_PATH}/raw-jsons/constructs/pdb_library_1.json"
    df = pd.read_json(construct_file)
    gen = GenerateMotifDataFrame()
    log.info("Generating motif dataframe")
    gen.run(df, "pdb_library_1")
    motif_file = f"{DATA_PATH}/raw-jsons/motifs/pdb_library_1_motifs_avg.json"
    df = pd.read_json(motif_file)
    log.info("Generating residue dataframe")
    gen = GenerateResidueDataFrame()
    gen.run(df, "pdb_library_1")
    residue_file = f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues.json"
    df = pd.read_json(residue_file)
    log.info("Generating pdb residue dataframe")
    df = generate_pdb_residue_dataframe(df)
    df.to_json(
        f"{DATA_PATH}/raw-jsons/residues/pdb_library_1_residues_pdb.json",
        orient="records",
    )


@cli.command()
def get_pdb_features():
    """
    Get the solvent accessibility features for all PDB files in the pdbs directory.
    """
    setup_logging()

    df = generate_distance_dataframe(max_distance=10)
    df.to_csv("data/pdb-features/distances_10a.csv", index=False)
    df = generate_distance_dataframe(max_distance=1000)
    df.to_csv("data/pdb-features/distances_all.csv", index=False)
    exit(0)
    # get all sasa values for different probe radii
    df_sasa = generate_sasa_dataframe()
    df_sasa.to_csv("data/pdb-features/sasa.csv", index=False)
    # get all hbonds
    df_hbonds = calculate_hbond_strength_all("data/pdbs")
    df_hbonds.to_csv("data/pdb-features/hbonds.csv", index=False)
    # df = calculate_hbond_strength("data/pdbs")
    # calculate_structural_parameters_with_dssr("data/pdbs")
    # df = get_all_torsional_parameters_from_dssr("data/pdbs")
    # df.to_csv("data/pdb-features/torsions.csv", index=False)
    # process_basepair_details()


if __name__ == "__main__":
    cli()
