"""Compute RDKit features for all the Therapeutics Data Commons (TDC) ADMET datasets"""
import subprocess
from pathlib import Path

from tqdm import tqdm

from constants import ADMET_ALL_SMILES_COLUMN


def rdkit_tdc_admet_all(
        data_dir: Path
) -> None:
    """Compute RDKit features for all the Therapeutics Data Commons (TDC) ADMET datasets.

    :param data_dir: A directory containing all the downloaded and prepared TDC ADMET data.
    """
    # Get dataset paths
    data_paths = sorted(data_dir.glob('*.csv'))

    # Compute features for each dataset using chemfunc
    for data_path in tqdm(data_paths):
        subprocess.run([
            'chemfunc', 'save_fingerprints',
            '--data_path', str(data_path),
            '--save_path', str(data_path.with_suffix('.npz')),
            '--smiles_column', ADMET_ALL_SMILES_COLUMN
        ])


if __name__ == '__main__':
    from tap import tapify

    tapify(rdkit_tdc_admet_all)
