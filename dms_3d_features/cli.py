import click
import glob
import os
import warnings

from dms_3d_features.pdb_features import (
    compute_solvent_accessibility_all,
    DSSRTorsionFileProcessor,
    calculate_hbond_strength_all,
)
from dms_3d_features.process_motifs import process_mutation_histograms_to_json
from dms_3d_features.logger import setup_logging, get_logger

warnings.filterwarnings(
    "ignore", message="FreeSASA: warning: Found no matches to resn 'A', typo?"
)

log = get_logger("cli")


def compute_sasa():
    dfs = []
    for probe_radius in [0.1, 0.25, 0.5, 1.5, 2.0, 2.5, 3.0]:
        df = compute_solvent_accessibility_all("data/pdbs", probe_radius=probe_radius)
        probe_radius = str(probe_radius).replace(".", "_")
        df.rename({"sasa": f"sasa_{probe_radius}"}, axis=1, inplace=True)
        dfs.append(df)
    df_final = dfs[0]
    dfs.pop(0)
    for df in dfs:
        df_final = df_final.merge(
            df, on=["m_sequence", "pdb_r_pos", "pdb_path", "r_nuc"]
        )
    df_final.to_csv("data/pdb-features/sasa.csv", index=False)


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
    # compute_sasa()
    # get all hbonds
    df_hbonds = calculate_hbond_strength_all("data/pdbs")
    df_hbonds.to_csv("data/pdb-features/hbonds.csv", index=False)
    # df = calculate_hbond_strength("data/pdbs")
    # calculate_structural_parameters_with_dssr("data/pdbs")
    # df = get_all_torsional_parameters_from_dssr("data/pdbs")
    # df.to_csv("data/pdb-features/torsions.csv", index=False)


if __name__ == "__main__":
    cli()
