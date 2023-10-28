"""Plot statistics about the DrugBank approved molecules."""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import trange

from tdc_constants import DRUGBANK_ATC_DELIMITER, DRUGBANK_ATC_PREFIX


def plot_drugbank_approved(
    data_path: Path, save_dir: Path, top_k_atc_codes: int = 25
) -> None:
    """Plot statistics about the DrugBank approved molecules.

    :param data_path: Path to a CSV file containing the DrugBank approved molecules.
    :param save_dir: Path to directory where plots will be saved.
    :param top_k_atc_codes: The number of ATC codes to plot at each ATC level.
    """
    # Load data
    data = pd.read_csv(data_path)

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot distribution of ATC codes at each level
    for level in trange(1, 5, desc="ATC levels"):
        # Compute ATC code counts at this level and keep only the top k
        atc_column = f"{DRUGBANK_ATC_PREFIX}_{level}"
        atc_code_counts = Counter(
            atc_code
            for atc_list in data[atc_column].dropna()
            for atc_code in atc_list.split(DRUGBANK_ATC_DELIMITER)
        )
        atc_code_df = pd.DataFrame.from_dict(
            atc_code_counts, orient="index", columns=["count"]
        )
        atc_code_df.sort_values("count", ascending=False, inplace=True)
        num_atc_codes = len(atc_code_df)

        # Create a Seaborn barplot with ATC code counts
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x=atc_code_df["count"].values[:top_k_atc_codes],
            y=atc_code_df.index[:top_k_atc_codes].str.upper(),
            palette="viridis",
        )
        plt.xlabel("Count")
        plt.ylabel("")

        if num_atc_codes <= top_k_atc_codes:
            plt.title(f"All {num_atc_codes:,} ATC Codes at Level {level}")
        else:
            plt.title(
                f"Top {top_k_atc_codes:,} of {num_atc_codes:,} ATC Codes at Level {level}"
            )

        plt.savefig(
            save_dir / f"atc_level_{level}_distribution.pdf", bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_drugbank_approved)
