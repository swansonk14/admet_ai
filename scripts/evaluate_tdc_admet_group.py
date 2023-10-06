"""Evaluate predictions from TDC ADMET Benchmark Group models."""
from pathlib import Path

import pandas as pd
from tdc.benchmark_group import admet_group

from admet_ai.constants import ADMET_GROUP_SEEDS, ADMET_GROUP_TARGET_COLUMN


def tdc_admet_group_evaluate(data_dir: Path, preds_dir: Path) -> None:
    """Evaluate predictions from TDC ADMET Benchmark Group models.

    :param data_dir: A directory containing the downloaded and prepared TDC ADMET data.
    :param preds_dir: A directory containing the predictions from chemprop models trained on the TDC ADMET datasets.
    """
    # Download/access the TDC ADMET Benchmark Group
    group = admet_group(path=data_dir)

    # Get dataset names
    names = [
        preds_subdir.name
        for preds_subdir in preds_dir.iterdir()
        if preds_subdir.is_dir()
    ]

    # Load predictions for each dataset
    predictions_list = []

    for seed in ADMET_GROUP_SEEDS:
        predictions = {}

        for name in names:
            preds_path = preds_dir / name / str(seed) / "test_preds.csv"
            preds = pd.read_csv(preds_path)
            predictions[name] = preds[ADMET_GROUP_TARGET_COLUMN].values

        predictions_list.append(predictions)

    # Evaluate predictions
    results = group.evaluate_many(predictions_list)

    # Print results
    print(results)


if __name__ == "__main__":
    from tap import tapify

    tapify(tdc_admet_group_evaluate)
