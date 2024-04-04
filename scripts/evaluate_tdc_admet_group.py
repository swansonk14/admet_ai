"""Evaluate predictions from TDC ADMET Benchmark Group models."""
from pathlib import Path

import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

from tdc_constants import ADMET_GROUP_SEEDS, ADMET_GROUP_TARGET_COLUMN


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

    # Evaluate predictions of single models across all folds
    single_results = group.evaluate_many(predictions_list)
    print(f"Results for single models across {len(predictions_list)} folds")
    print(single_results)
    print()

    # Evaluate predictions for ensemble of all folds
    ensemble_predictions = {
        name: np.mean(np.stack([preds[name] for preds in predictions_list]), axis=0)
        for name in names
    }
    ensemble_results = group.evaluate(ensemble_predictions)
    print(f"Results for ensemble model of all {len(predictions_list)} folds")
    print(ensemble_results)
    print()


if __name__ == "__main__":
    from tap import tapify

    tapify(tdc_admet_group_evaluate)
