import click
import warnings

from dms_3d_features.sasa import generate_sasa_dataframe
from dms_3d_features.pdb_features import (
    DSSRTorsionFileProcessor,
    calculate_hbond_strength_all,
    process_basepair_details,
)
from dms_3d_features.process_motifs import process_mutation_histograms_to_json
from dms_3d_features.logger import setup_logging, get_logger

warnings.filterwarnings(
    "ignore", message="FreeSASA: warning: Found no matches to resn 'A', typo?"
)

log = get_logger("cli")


# cli functions #################################################################


@click.group()
def cli():
    pass


@cli.command()
def process_mutation_histograms():
    setup_logging()
    process_mutation_histograms_to_json()


@cli.command()
def get_pdb_features():
    """
    Get the solvent accessibility features for all PDB files in the pdbs directory.
    """
    setup_logging()
    # get all sasa values for different probe radii
    # df_sasa = generate_sasa_dataframe()
    # df_sasa.to_csv("data/pdb-features/sasa.csv", index=False)
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
