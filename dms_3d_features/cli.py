import click

from dms_3d_features.pdb_features import (
    compute_solvent_accessibility_all,
    calculate_hbond_strength,
)


@click.group()
def cli():
    pass


@cli.command()
def get_pdb_features():
    """
    Get the solvent accessibility features for all PDB files in the pdbs directory.
    """
    # df = compute_solvent_accessibility_all("data/pdbs")
    # df.to_csv("data/pdb-features/sasa.csv", index=False)
    df = calculate_hbond_strength("data/pdbs")


if __name__ == "__main__":
    cli()
