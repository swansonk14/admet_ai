"""Compute RDKit features for the Therapeutics Data Commons (TDC) ADMET Benchmark Group datasets."""
import subprocess
from pathlib import Path

from tqdm import tqdm

from constants import ADMET_GROUP_SEEDS, ADMET_GROUP_SMILES_COLUMN


def rdkit_tdc_admet_group(
        data_dir: Path
) -> None:
    """Compute RDKit features for the Therapeutics Data Commons (TDC) ADMET Benchmark Group datasets.

    :param data_dir: A directory containing the downloaded and prepared TDC ADMET Benchmark Group data.
    """
    # Get dataset paths
    data_dirs = sorted(data_dir.iterdir())

    # Compute features for each dataset using chemfunc
    for data_dir in tqdm(data_dirs):
        # Compute features for train and val sets for each seed
        for seed in ADMET_GROUP_SEEDS:
            for split in ['train', 'val']:
                data_path = data_dir / str(seed) / f'{split}.csv'

                subprocess.run([
                    'chemfunc', 'save_fingerprints',
                    '--data_path', str(data_path),
                    '--save_path', str(data_path.with_suffix('.npz')),
                    '--smiles_column', ADMET_GROUP_SMILES_COLUMN
                ])

        # Compute features for test set
        subprocess.run([
            'chemfunc', 'save_fingerprints',
            '--data_path', str(data_dir / 'test.csv'),
            '--save_path', str(data_dir / 'test.npz'),
            '--smiles_column', ADMET_GROUP_SMILES_COLUMN
        ])


if __name__ == '__main__':
    from tap import tapify

    tapify(rdkit_tdc_admet_group)
