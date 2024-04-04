"""Plot results from TDC Leaderboard and TDC Single-Task and Multi-Task."""
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGSIZE = (14, 10)
matplotlib.rcParams["font.size"] = 18
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
SCATTER_SIZE = 150


def split_axes(axes: list[plt.Axes]) -> None:
    """Split axes (two columns) into two disjoint plots.

    :param axes: List of two axes (two columns) to split.
    """
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")

    axes[0].spines.right.set_visible(False)
    axes[1].spines.left.set_visible(False)
    axes[1].tick_params(axis="y", which="both", left=False, right=False)

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-d, -1), (d, 1)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    axes[0].plot([1, 1], [0, 1], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 0], [0, 1], transform=axes[1].transAxes, **kwargs)


def plot_tdc_leaderboard_ranks(
    model_to_ranks: dict[str, list[int]],
    all_dataset_models: list[str],
    datasets: list[str],
    save_dir: Path,
) -> None:
    """Plot the ranks for each model that is evaluated on all datasets on the TDC Leaderboard.

    :param model_to_ranks: Dictionary mapping each model to its list of ranks.
    :param all_dataset_models: List of models evaluated on all datasets.
    :param datasets: List of datasets. Must be in the same order as the ranks in model_to_ranks.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Adjust font size
    matplotlib.rcParams["font.size"] = 28

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

    # Rename models
    model_to_new_name = {
        "Chemprop-RDKit": "ADMET-AI",
        "RDKit2D + MLP (DeepPurpose)": "RDKit2D + MLP",
        "Morgan + MLP (DeepPurpose)": "Morgan + MLP",
        "CNN (DeepPurpose)": "CNN",
    }

    model_ranks["Model"] = model_ranks["Model"].apply(
        lambda model: model_to_new_name.get(model, model)
    )

    # Plot the ranks for each model that is evaluated on all datasets
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        x="Rank", y="Model", data=model_ranks, hue="Model", ax=ax, errorbar="se",
    )
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_dir / "leaderboard_model_ranks.pdf", bbox_inches="tight")
    plt.close()


def plot_tdc_leaderboard_results(
    results: pd.ExcelFile,
    leaderboard_results: pd.DataFrame,
    all_dataset_models: list[str],
    save_dir: Path,
) -> None:
    """Plot results from TDC Leaderboard.

    :param results: Excel file containing results.
    :param leaderboard_results: DataFrame containing results from TDC Leaderboard.
    :param all_dataset_models: List of models evaluated on all datasets.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Adjust font size
    matplotlib.rcParams["font.size"] = 18

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot TDC Leaderboard results
    for i, (metric, val_min, val_max) in enumerate(
        [
            ("AUROC", 0.5, 1.0),
            ("AUPRC", 0.0, 1.0),
            ("Spearman", 0.0, 1.0),
            ("MAE", None, None),
        ]
    ):
        if metric == "MAE":
            fig, axes = plt.subplots(1, 2, sharey=True, figsize=FIGSIZE)
            fig.subplots_adjust(wspace=0.05)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            axes = [ax]

        metric_dataset_names = leaderboard_results[
            leaderboard_results["Leaderboard Metric"] == metric
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
                    x=metric,
                    y="Dataset",
                    data=leaderboard_partial_data,
                    ax=ax,
                    color=sns.color_palette()[0],
                    s=SCATTER_SIZE,
                    label="Leaderboard (partial)"
                    if j == 0 and k == len(axes) - 1
                    else None,
                    marker="x",
                )
                sns.scatterplot(
                    x=metric,
                    y="Dataset",
                    data=leaderboard_all_data,
                    ax=ax,
                    color=sns.color_palette()[0],
                    s=SCATTER_SIZE,
                    label="Leaderboard (all)"
                    if j == 0 and k == len(axes) - 1
                    else None,
                    marker="o",
                )
                sns.scatterplot(
                    x=metric,
                    y="Dataset",
                    data=chemprop_rdkit_data,
                    ax=ax,
                    color=sns.color_palette()[3],
                    s=4 * SCATTER_SIZE,
                    label="Chemprop-RDKit" if j == 0 and k == len(axes) - 1 else None,
                    marker="*",
                )

        if metric == "MAE":
            split_axes(axes)

            axes[0].set_xlim(0.0, 1.5)
            axes[1].set_xlim(6.5, 13.5)

            fig.text(0.63, 0.04, metric, ha="center")
        else:
            axes[0].set_xlim(val_min, val_max)
            axes[0].set_ylabel("")

        plt.tight_layout()

        if metric == "MAE":
            fig.subplots_adjust(bottom=0.1)

        plt.savefig(save_dir / f"leaderboard_{metric.lower()}.pdf", bbox_inches="tight")
        plt.close()


def plot_tdc_leaderboard_single_vs_ensemble(
    results: pd.ExcelFile, save_dir: Path
) -> None:
    """Plot results from TDC Leaderboard single model vs ensemble model.

    :param results: Excel file containing results.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Adjust font size
    matplotlib.rcParams["font.size"] = 18

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get regression and classification results
    regression = results.parse("TDC Leaderboard Regression")
    classification = results.parse("TDC Leaderboard Classification")

    # Create a seaborn barplot comparing single model and ensemble model performance for each dataset
    for i, (metric, val_min, val_max) in enumerate(
        [
            ("AUROC", 0.5, 1.0),
            ("AUPRC", 0.0, 1.0),
            ("Spearman", 0.0, 1.0),
            ("MAE", None, None),
        ]
    ):
        # Adapt plot based on metric
        if metric == "MAE":
            fig, axes = plt.subplots(1, 2, sharey=True, figsize=FIGSIZE)
            fig.subplots_adjust(wspace=0.05)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            axes = [ax]

        # Select results based on metric
        results = classification if metric in {"AUROC", "AUPRC"} else regression
        results = results[results["Leaderboard Metric"] == metric]

        # Reformat results
        results = results.rename(
            columns={
                "Leaderboard Mean": "Single",
                "Leaderboard Standard Deviation": "Single Std",
                "Leaderboard Ensemble": "Ensemble",
            },
        )
        means = results.melt(
            id_vars="Dataset",
            value_vars=["Single", "Ensemble",],
            var_name="Model Type",
            value_name=metric,
        )

        # Plot results
        for ax in axes:
            # Add mean bars
            sns.barplot(
                x=metric, y="Dataset", hue="Model Type", data=means, ax=ax,
            )

            # Add standard deviation for single model
            for i, dataset in enumerate(results["Dataset"]):
                ax.errorbar(
                    x=results.iloc[i]["Single"],
                    y=[i - 0.2],
                    xerr=results.iloc[i]["Single Std"],
                    fmt="none",
                    c="black",
                    capsize=3,
                )

        # Adapt plot limits
        if metric == "MAE":
            axes[0].set_xlim(0.0, 1.0)
            axes[1].set_xlim(3.0, 10.0)

            axes[0].legend().remove()

            split_axes(axes)

            fig.text(
                0.625, 0.04, metric, ha="center",
            )
        else:
            axes[0].set_ylabel("")

        plt.tight_layout()

        if metric == "MAE":
            fig.subplots_adjust(bottom=0.1)

        plt.savefig(
            save_dir / f"single_vs_ensemble_{metric.lower()}.pdf", bbox_inches="tight"
        )
        plt.close()


def plot_tdc_single_task_vs_multi_task_results(
    results: pd.ExcelFile, save_dir: Path
) -> None:
    """Plot results from TDC Single-Task and Multi-Task.

    :param results: Excel file containing results.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Adjust font size
    matplotlib.rcParams["font.size"] = 18

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get regression and classification results
    regression_all = results.parse("TDC Single-Multi Regression")
    classification_all = results.parse("TDC Single-Multi Classification")

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

        # Rename task type column to remove metric
        task_type_mapping = {
            f"Single Task {metric} Mean": "Single Task",
            f"Multitask {metric} Mean": "Multi Task",
            f"Single Task {metric} Standard Deviation": "Single Task",
            f"Multitask {metric} Standard Deviation": "Multi Task",
        }

        results_all_mean_single_vs_multi[
            "Task Type"
        ] = results_all_mean_single_vs_multi["Task Type"].apply(
            lambda task_type: task_type_mapping[task_type]
        )
        results_all_std_single_vs_multi["Task Type"] = results_all_std_single_vs_multi[
            "Task Type"
        ].apply(lambda task_type: task_type_mapping[task_type])

        if metric == "MAE":
            fig, axes = plt.subplots(1, 2, sharey=True, figsize=FIGSIZE)
            fig.subplots_adjust(wspace=0.05)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            axes = [ax]

        for ax in axes:
            sns.barplot(
                x=f"{metric} Mean",
                y="Dataset",
                hue="Task Type",
                data=results_all_mean_single_vs_multi,
                ax=ax,
            )

            for i, dataset in enumerate(results_all["Dataset"]):
                ax.errorbar(
                    x=results_all_mean_single_vs_multi.loc[
                        (results_all_mean_single_vs_multi["Dataset"] == dataset),
                        f"{metric} Mean",
                    ].values,
                    y=[i - 0.2, i + 0.2],
                    xerr=results_all_std_single_vs_multi.loc[
                        (results_all_std_single_vs_multi["Dataset"] == dataset),
                        f"{metric} Standard Deviation",
                    ].values,
                    fmt="none",
                    c="black",
                    capsize=3,
                )

        if metric == "MAE":
            axes[0].set_xlim(0.0, 1.0)
            axes[1].set_xlim(3.0, 40.0)

            axes[0].legend().remove()

            split_axes(axes)

            fig.text(
                0.65, 0.04, f"{metric} Mean", ha="center",
            )
        else:
            axes[0].set_ylabel("")

        # Change R^2 metric to $R^2$ for plotting
        if metric == "R^2":
            axes[0].set_xlabel(r"$R^2$ Mean")

        plt.tight_layout()

        if metric == "MAE":
            fig.subplots_adjust(bottom=0.1)

        plt.savefig(save_dir / f"all_{metric.lower()}.pdf", bbox_inches="tight")
        plt.close()


def plot_tdc_results(results_path: Path, save_dir: Path) -> None:
    """Plot results from TDC Leaderboard and TDC Single-Task and Multi-Task.

    :param results_path: Path to an Excel file containing results.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = pd.ExcelFile(results_path)

    # Get TDC Leaderboard results
    regression_leaderboard = results.parse("TDC Leaderboard Regression")
    classification_leaderboard = results.parse("TDC Leaderboard Classification")
    leaderboard_results = pd.concat(
        [regression_leaderboard, classification_leaderboard]
    )

    # Map each model to its set of ranks
    model_to_ranks = defaultdict(list)
    datasets = sorted(set(leaderboard_results["Dataset"]))
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

    # Plot TDC Leaderboard leaderboard ranks
    plot_tdc_leaderboard_ranks(
        model_to_ranks=model_to_ranks,
        all_dataset_models=all_dataset_models,
        datasets=datasets,
        save_dir=save_dir,
    )

    # Plot TDC Leaderboard results
    plot_tdc_leaderboard_results(
        results=results,
        leaderboard_results=leaderboard_results,
        all_dataset_models=all_dataset_models,
        save_dir=save_dir,
    )

    # Plot TDC Leaderboard single vs ensemble
    plot_tdc_leaderboard_single_vs_ensemble(results=results, save_dir=save_dir)

    # Plot TDC Single-Task vs Multi-Task results
    plot_tdc_single_task_vs_multi_task_results(results=results, save_dir=save_dir)


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_tdc_results)
