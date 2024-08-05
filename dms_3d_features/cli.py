import click
import glob
import os

from dms_3d_features.pdb_features import (
    compute_solvent_accessibility_all,
    calculate_hbond_strength,
    calculate_structural_parameters_with_dssr,
    DSSRTorsionFileProcessor,
    get_all_torsional_parameters_from_dssr,
)

from dms_3d_features.logger import setup_logging, get_logger

log = get_logger("cli")


@click.group()
def cli():
    pass


@cli.command()
def get_pdb_features():
    """
    Get the solvent accessibility features for all PDB files in the pdbs directory.
    """
    setup_logging()
    # df = compute_solvent_accessibility_all("data/pdbs")
    # df.to_csv("data/pdb-features/sasa.csv", index=False)
    df = calculate_hbond_strength("data/pdbs")
    # calculate_structural_parameters_with_dssr("data/pdbs")
    # df = get_all_torsional_parameters_from_dssr("data/pdbs")
    # df.to_csv("data/pdb-features/torsions.csv", index=False)


if __name__ == "__main__":
    cli()
