"""Randomly samples molecules from a dataset."""
from pathlib import Path

import pandas as pd

from tap import tapify


def sample_molecules(
    data_path: Path,
    save_path: Path,
    num_molecules: int,
    max_smiles_length: int | None = None,
    smiles_column: str = "smiles",
    random_state: int = 0,
) -> None:
    """Samples molecules, either uniformly at random across the entire dataset or uniformly at random from each cluster.

    :param data_path: Path to CSV file containing SMILES.
    :param save_path: Path to CSV file where the selected molecules will be saved.
    :param num_molecules: Number of molecules to select.
    :param max_smiles_length: The maximum length of SMILES to include in the sample.
    :param smiles_column: Name of the column containing SMILES strings.
    :param random_state: Random seed for sampling.
    """
    print("Loading data")
    data = pd.read_csv(data_path)
    print(f"Data size = {len(data):,}")

    if max_smiles_length is not None:
        print(f"Removing SMILES strings longer than {max_smiles_length:,} characters")
        data = data[data[smiles_column].str.len() <= max_smiles_length]
        print(f"Data size = {len(data):,}")

    print(f"Selecting {num_molecules:,} molecules")
    sampled = data.sample(n=num_molecules, random_state=random_state)

    print("Saving data")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(save_path, index=False)


if __name__ == "__main__":
    tapify(sample_molecules)
