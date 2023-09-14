"""Download and prepare all the Therapeutics Data Commons (TDC) ADMET datasets."""
from pathlib import Path

import pandas as pd
from tdc.single_pred import ADME, Tox
from tqdm import tqdm

from constants import (
    ADME_DATASET_TO_TYPE,
    ADMET_GROUP_SMILES_COLUMN,
    ADMET_GROUP_TARGET_COLUMN,
    DATASET_TO_LABEL_NAMES,
    ADMET_ALL_SMILES_COLUMN,
    TOX_DATASET_TO_TYPE
)


def prepare_tdc_admet_all(
        save_dir: Path,
        skip_datasets: list[str] = None
) -> None:
    """Download and prepare all the Therapeutics Data Commons (TDC) ADMET datasets.

    :param save_dir: A directory where the TDC AMDET data will be saved.
    :param skip_datasets: A list of dataset names to skip.
    """
    # Map dataset to dataset class
    dataset_to_class = {
        tox_dataset: Tox
        for tox_dataset in TOX_DATASET_TO_TYPE
    } | {
        adme_dataset: ADME
        for adme_dataset in ADME_DATASET_TO_TYPE
    }

    # Skip datasets
    if skip_datasets is not None:
        dataset_to_class = {
            dataset: data_class
            for dataset, data_class in dataset_to_class.items()
            if dataset not in skip_datasets
        }

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Download and prepare each dataset
    for data_name, data_class in tqdm(dataset_to_class.items(), desc='Datasets'):
        # Get names of labels for dataset (or None if only one label)
        label_names = DATASET_TO_LABEL_NAMES.get(data_name, [None])
        data = []

        # Get data for each label
        for label_name in tqdm(label_names, desc='Labels'):
            label_data = data_class(name=data_name, label_name=label_name, path=save_dir).get_data()
            data.append(dict(zip(label_data[ADMET_GROUP_SMILES_COLUMN].tolist(), label_data[ADMET_GROUP_TARGET_COLUMN].tolist())))

        # Get all SMILES
        smiles = sorted(set.union(*[set(smiles_to_target) for smiles_to_target in data]))

        # Rename None label name to dataset name
        if label_names == [None]:
            label_names = [data_name]

        # Create DataFrame with all SMILES and targets
        data = pd.DataFrame({
            ADMET_ALL_SMILES_COLUMN: smiles
        } | {
            label_name: [smiles_to_target.get(smiles, None) for smiles in smiles]
            for label_name, smiles_to_target in zip(label_names, data)
        })

        # Save data
        data.to_csv(save_dir / f'{data_name}.csv', index=False)


if __name__ == '__main__':
    from tap import tapify

    tapify(prepare_tdc_admet_all)
