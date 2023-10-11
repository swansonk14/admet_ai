"""Plot results from TDC ADMET Group and TDC ADMET All."""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def split_axes(axes: list[plt.Axes]) -> None:
    """Split axes into two disjoint plots.

    :param axes: List of two axes to split.
    """
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")

    axes[0].spines.bottom.set_visible(False)
    axes[1].spines.top.set_visible(False)
    axes[0].tick_params(axis="x", which="both", bottom=False, top=False)
    axes[1].xaxis.tick_bottom()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)


def plot_leaderboard_ranks(
    model_to_ranks: dict[str, list[int]],
    all_dataset_models: list[str],
    datasets: list[str],
    save_dir: Path,
) -> None:
    """Plot the ranks for each model that is evaluated on all datasets on the TDC ADMET group leaderboard.

    :param model_to_ranks: Dictionary mapping each model to its list of ranks.
    :param all_dataset_models: List of models evaluated on all datasets.
    :param datasets: List of datasets. Must be in the same order as the ranks in model_to_ranks.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Compute ranks for each model that is evaluated on all datasets
    dataset_model_ranks = pd.DataFrame(
        {model: model_to_ranks[model] for model in all_dataset_models}, index=datasets,
    )

    # Sort models by average rank
    models_by_average_rank = sorted(
        all_dataset_models, key=lambda model: np.mean(dataset_model_ranks[model])
    )
    dataset_model_ranks = dataset_model_ranks[models_by_average_rank]

    # Melt dataframe for plotting
    model_ranks = dataset_model_ranks.melt(var_name="Model", value_name="Rank")

    # Plot the ranks for each model that is evaluated on all datasets
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(x="Model", y="Rank", data=model_ranks, ax=ax, errorbar="se")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(save_dir / "leaderboard_model_ranks.pdf", bbox_inches="tight")


def plot_group_results(
    results: pd.ExcelFile,
    group_results: pd.DataFrame,
    all_dataset_models: list[str],
    save_dir: Path,
) -> None:
    """Plot results from TDC ADMET Group.

    :param results: Excel file containing results.
    :param group_results: DataFrame containing results from TDC ADMET Group.
    :param all_dataset_models: List of models evaluated on all datasets.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot TDC ADMET Group results
    for i, (metric, y_min, y_max) in enumerate(
        [
            ("AUROC", 0.5, 1.0),
            ("AUPRC", 0.0, 1.0),
            ("Spearman", 0.0, 1.0),
            ("MAE", None, None),
        ]
    ):
        if metric == "MAE":
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
            fig.subplots_adjust(hspace=0.05)
        else:
            fig, ax = plt.subplots(figsize=(6, 8))
            axes = [ax]

        metric_dataset_names = group_results[
            group_results["Leaderboard Metric"] == metric
        ]["Dataset"]

        for j, metric_dataset_name in enumerate(metric_dataset_names):
            dataset_results = results.parse(metric_dataset_name)
            dataset_results["Dataset"] = metric_dataset_name
            dataset_results[metric] = dataset_results[metric].apply(
                lambda val: float(val.split(" ")[0])
            )
            leaderboard_partial_data = dataset_results[
                ~dataset_results["Model"].isin(all_dataset_models)
            ]
            leaderboard_all_data = dataset_results[
                dataset_results["Model"].isin(all_dataset_models)
            ]
            chemprop_rdkit_data = dataset_results[
                dataset_results["Model"] == "Chemprop-RDKit"
            ]

            for k, ax in enumerate(axes):
                sns.scatterplot(
                    x="Dataset",
                    y=metric,
                    data=leaderboard_partial_data,
                    ax=ax,
                    color=sns.color_palette()[0],
                    s=60,
                    label="Leaderboard (partial)" if j == 0 and k == 0 else None,
                    marker="x",
                )
                sns.scatterplot(
                    x="Dataset",
                    y=metric,
                    data=leaderboard_all_data,
                    ax=ax,
                    color=sns.color_palette()[0],
                    s=60,
                    label="Leaderboard (all)" if j == 0 and k == 0 else None,
                    marker="o",
                )
                sns.scatterplot(
                    x="Dataset",
                    y=metric,
                    data=chemprop_rdkit_data,
                    ax=ax,
                    color=sns.color_palette()[3],
                    s=300,
                    label="Chemprop-RDKit" if j == 0 and k == 0 else None,
                    marker="*",
                )

        if metric == "MAE":
            split_axes(axes)

            axes[0].set_ylim(6.5, 13.5)
            axes[1].set_ylim(0.0, 1.5)

            fig.text(0.04, 0.6, metric, va="center", rotation="vertical", fontsize=12)
        else:
            axes[0].set_ylim(y_min, y_max)
            axes[0].set_xlabel("")

        axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=90)

        plt.tight_layout()

        if metric == "MAE":
            fig.subplots_adjust(left=0.15)

        plt.savefig(save_dir / f"leaderboard_{metric.lower()}.pdf", bbox_inches="tight")


def plot_all_results(results: pd.ExcelFile, save_dir: Path) -> None:
    """Plot results from TDC ADMET All.

    :param results: Excel file containing results.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get regression and classification results
    regression_all = results.parse("TDC ADMET All Regression")
    classification_all = results.parse("TDC ADMET All Classification")

    # Create a seaborn barplot comparing single task and multitask performance for each dataset
    for results_all, metric in [
        (classification_all, "AUROC"),
        (classification_all, "AUPRC"),
        (regression_all, "R^2"),
        (regression_all, "MAE"),
    ]:
        results_all_mean_single_vs_multi = results_all.melt(
            id_vars="Dataset",
            value_vars=[f"Single Task {metric} Mean", f"Multitask {metric} Mean"],
            var_name="Task Type",
            value_name=f"{metric} Mean",
        )
        results_all_std_single_vs_multi = results_all.melt(
            id_vars="Dataset",
            value_vars=[
                f"Single Task {metric} Standard Deviation",
                f"Multitask {metric} Standard Deviation",
            ],
            var_name="Task Type",
            value_name=f"{metric} Standard Deviation",
        )

        if metric == "MAE":
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
            fig.subplots_adjust(hspace=0.05)
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            axes = [ax]

        for ax in axes:
            sns.barplot(
                x="Dataset",
                y=f"{metric} Mean",
                hue="Task Type",
                data=results_all_mean_single_vs_multi,
                ax=ax,
            )

            for i, dataset in enumerate(results_all["Dataset"]):
                ax.errorbar(
                    x=[i - 0.2, i + 0.2],
                    y=results_all_mean_single_vs_multi.loc[
                        (results_all_mean_single_vs_multi["Dataset"] == dataset),
                        f"{metric} Mean",
                    ].values,
                    yerr=results_all_std_single_vs_multi.loc[
                        (results_all_std_single_vs_multi["Dataset"] == dataset),
                        f"{metric} Standard Deviation",
                    ].values,
                    fmt="none",
                    c="black",
                    capsize=5,
                )

        if metric == "MAE":
            axes[0].set_ylim(3.0, 40.0)
            axes[1].set_ylim(0.0, 1.0)

            axes[1].legend().remove()

            split_axes(axes)

            fig.text(
                0.04,
                0.6,
                f"{metric} Mean",
                va="center",
                rotation="vertical",
                fontsize=12,
            )
        else:
            axes[0].set_xlabel("")

        axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=90)

        plt.tight_layout()

        if metric == "MAE":
            fig.subplots_adjust(left=0.1)

        plt.savefig(save_dir / f"all_{metric.lower()}.pdf", bbox_inches="tight")


def plot_results(results_path: Path, save_dir: Path) -> None:
    """Plot results from TDC ADMET Group and TDC ADMET All.

    :param results_path: Path to an Excel file containing results.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = pd.ExcelFile(results_path)

    # Get TDC ADMET group results
    regression_group = results.parse("TDC ADMET Group Regression")
    classification_group = results.parse("TDC ADMET Group Classification")
    group_results = pd.concat([regression_group, classification_group])

    # Map each model to its set of ranks
    model_to_ranks = defaultdict(list)
    datasets = sorted(set(group_results["Dataset"]))
    for dataset in datasets:
        data = results.parse(dataset)
        for index, model in enumerate(data["Model"]):
            model_to_ranks[model].append(index + 1)

    # Determine which models are evaluated on all datasets
    all_dataset_models = sorted(
        {
            model
            for model, ranks in model_to_ranks.items()
            if len(ranks) == len(datasets)
        }
    )
    print(f"Number of models: {len(model_to_ranks):,}")
    print(f"Number of models evaluated on all datasets: {len(all_dataset_models):,}")

    # Plot TDC ADMET Group leaderboard ranks
    plot_leaderboard_ranks(
        model_to_ranks=model_to_ranks,
        all_dataset_models=all_dataset_models,
        datasets=datasets,
        save_dir=save_dir,
    )

    # Plot TDC ADMET Group results
    plot_group_results(
        results=results,
        group_results=group_results,
        all_dataset_models=all_dataset_models,
        save_dir=save_dir,
    )

    # Plot TDC ADMET All results
    plot_all_results(results=results, save_dir=save_dir)


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_results)
