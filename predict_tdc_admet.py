"""Make predictions on a dataset using Chemprop models trained on TDC ADMET data (all or Benchmark Data group)."""
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm


def predict_tdc_admet(
        data_path: Path,
        save_path: Path,
        model_dir: Path,
        model_type: Literal['chemprop', 'chemprop_rdkit'],
        smiles_column: str = 'smiles'
) -> None:
    """Make predictions on a dataset using Chemprop models trained on TDC ADMET data (all or Benchmark Data group).

    Note: If using a Chemprop-RDKit model, this script first computes RDKit features.

    :param data_path: Path to a CSV file containing a dataset of molecules.
    :param save_path: Path to a CSV file where predictions will be saved.
    :param model_dir: Path to a directory containing Chemprop or Chemprop-RDKit models.
    :param model_type: The type of model to use (Chemprop or Chemprop-RDKit).
    :param smiles_column: Name of the column containing SMILES strings.
    """
    # Compute RDKit features if needed
    if model_type == 'chemprop_rdkit':
        subprocess.run([
            'chemfunc', 'save_features',
            '--data_path', str(data_path),
            '--save_path', str(data_path.with_suffix('.npz')),
            '--smiles_column', smiles_column
        ])

    # Get model directories
    model_dirs = sorted(model_dir.iterdir())

    # Create DataFrame to store predictions
    preds = None

    # Make predictions using each model
    with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
        for model_dir in tqdm(model_dirs):
            command = [
                'chemprop_predict',
                '--test_path', str(data_path),
                '--checkpoint_dir', str(model_dir),
                '--preds_path', temp_file.name,
                '--quiet'
            ]

            if model_type == 'chemprop_rdkit':
                command += [
                    '--features_path', str(data_path.with_suffix('.npz')),
                    '--no_features_scaling'
                ]

            subprocess.run(command)

            # Read predictions
            new_preds = pd.read_csv(temp_file.name)

            # Merge predictions
            if preds is None:
                preds = new_preds
            else:
                new_columns = new_preds.columns.difference(preds.columns)
                preds.merge(new_preds[new_columns])

    # Save predictions
    save_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(save_path, index=False)


if __name__ == '__main__':
    from tap import tapify

    tapify(predict_tdc_admet)
