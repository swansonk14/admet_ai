"""Download and prepare all the Therapeutics Data Commons (TDC) ADMET datasets."""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tdc_constants import (
    ADMET_ALL_SMILES_COLUMN,
    ADMET_GROUP_SMILES_COLUMN,
    ADMET_GROUP_TARGET_COLUMN,
    DATASET_TO_LABEL_NAMES,
    DATASET_TO_TYPE,
    TDC_DATASET_TO_CLASS,
)


def prepare_tdc_admet_all(save_dir: Path, skip_datasets: list[str] = None) -> None:
    """Download and prepare all the Therapeutics Data Commons (TDC) ADMET datasets.

    :param save_dir: A directory where the TDC AMDET data will be saved.
    :param skip_datasets: A list of dataset names to skip.
    """
    # Map dataset to dataset class
    dataset_to_class = TDC_DATASET_TO_CLASS

    # Skip datasets
    if skip_datasets is not None:
        dataset_to_class = {
            dataset: data_class
            for dataset, data_class in dataset_to_class.items()
            if dataset not in skip_datasets
        }

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create list of dataset stats
    dataset_stats = []

    # Download and prepare each dataset
    for data_name, data_class in tqdm(dataset_to_class.items(), desc="Datasets"):
        # Get names of labels for dataset (or None if only one label)
        label_names = DATASET_TO_LABEL_NAMES.get(data_name, [None])
        data = []

        # Skin reaction exception
        if data_name == "Skin_Reaction":
            tdc_data_name = "Skin Reaction"
        else:
            tdc_data_name = data_name

        # Get data for each label
        for label_name in tqdm(label_names, desc="Labels"):
            label_data = data_class(
                name=tdc_data_name, label_name=label_name, path=save_dir
            ).get_data()
            data.append(
                dict(
                    zip(
                        label_data[ADMET_GROUP_SMILES_COLUMN].tolist(),
                        label_data[ADMET_GROUP_TARGET_COLUMN].tolist(),
                    )
                )
            )

        # Get all SMILES
        smiles = sorted(
            set.union(*[set(smiles_to_target) for smiles_to_target in data])
        )

        # Rename None label name to dataset name
        if label_names == [None]:
            label_names = [data_name]

        # Create DataFrame with all SMILES and targets
        data = pd.DataFrame(
            {ADMET_ALL_SMILES_COLUMN: smiles}
            | {
                label_name: [smiles_to_target.get(smiles, None) for smiles in smiles]
                for label_name, smiles_to_target in zip(label_names, data)
            }
        )

        # Process each label as a separate dataset
        for label_name in label_names:
            # Get label data
            label_data = data[[ADMET_ALL_SMILES_COLUMN, label_name]]
            label_data = label_data[label_data[label_name].notna()]

            # Compute class balance
            if DATASET_TO_TYPE[label_name] == "classification":
                class_balance = label_data[label_name].value_counts(normalize=True)[1]
            else:
                class_balance = None

            dataset_stats.append(
                {
                    "name": label_name,
                    "size": len(label_data),
                    "min": label_data[label_name].min(),
                    "max": label_data[label_name].max(),
                    "class_balance": class_balance,
                }
            )

            # Save data
            label_data.to_csv(save_dir / f"{label_name}.csv", index=False)

    # Print dataset stats
    dataset_stats = pd.DataFrame(dataset_stats).set_index("name")
    pd.set_option("display.max_rows", None)
    print(dataset_stats)

    # Clean up TDC data files
    tdc_files = save_dir.glob("*.tab")
    for tdc_data_name in tdc_files:
        tdc_data_name.unlink()


if __name__ == "__main__":
    from tap import tapify

    tapify(prepare_tdc_admet_all)
