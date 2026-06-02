from argparse import ArgumentParser
from pathlib import Path
from pickle import load

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from dataframes import df_tot

plt.style.use(hep.style.LHCb2)


PROCESS_LABELS = {
    "ww": r"$W^+W^-$",
    "dfdy": r"$Z \rightarrow \tau\tau$",
    "ttbar": r"$t\overline{t}$",
}

def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot WW BDT score distributions and S/sqrt(S+B), using WW as signal "
            "and DFDY+ttbar as background."
        )
    )
    parser.add_argument("pickle_file", help="Path to a trained BDT pickle file.")
    parser.add_argument(
        "--output-dir",
        default="Plots",
        help="Directory where plots will be saved. Defaults to Plots.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins. Defaults to 100.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold step for the significance scan. Defaults to 0.01.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for output plot names. Defaults to the pickle filename stem.",
    )
    return parser.parse_args()


def load_bdt(pickle_file):
    with open(pickle_file, "rb") as handle:
        imported_bdt = load(handle)

    required_keys = {"Model", "Features"}
    missing_keys = required_keys - set(imported_bdt)
    if missing_keys:
        raise ValueError(
            f"{pickle_file} is missing required key(s): {', '.join(sorted(missing_keys))}"
        )

    return imported_bdt


def get_ww_probability(classifier, dataframe, feature_names):
    missing_features = [name for name in feature_names if name not in dataframe.columns]
    if missing_features:
        raise ValueError(
            "Input dataframe is missing BDT feature(s): "
            + ", ".join(sorted(missing_features))
        )

    probabilities = classifier.predict_proba(dataframe[feature_names])

    if hasattr(classifier, "classes_"):
        classes = list(classifier.classes_)
        if len(classes) == 2 and 1 in classes:
            return probabilities[:, classes.index(1)]
        if len(classes) == 3 and 0 in classes:
            return probabilities[:, classes.index(0)]

    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    if probabilities.shape[1] == 3:
        return probabilities[:, 0]

    raise ValueError(
        "Could not determine the WW probability column from classifier output."
    )


def run_bdt(classifier, dataframe, feature_names):
    results = dataframe.copy()
    results["ww_prob"] = get_ww_probability(classifier, results, feature_names)
    return results


def calculate_significances(dataframe, step):
    thresholds = np.arange(0, 1.0, step)
    significances = []

    for threshold in thresholds:
        passed = dataframe[dataframe["ww_prob"] > threshold]
        signal = passed[passed["Process"] == "ww"]["Weight"].sum()
        background = passed[passed["Process"].isin(["dfdy", "ttbar"])]["Weight"].sum()

        if signal + background > 0:
            significance = signal / np.sqrt(signal + background)
        else:
            significance = 0

        significances.append(significance)

    best_index = int(np.argmax(significances))
    return thresholds[best_index], significances[best_index], thresholds, significances


def plot_score_distribution(dataframe, best_cut, bins, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    for process in ["ww", "dfdy", "ttbar"]:
        process_data = dataframe[dataframe["Process"] == process]
        weight_sum = process_data["Weight"].sum()
        if weight_sum == 0:
            raise ValueError(f"Cannot normalize {process} histogram because its weight sum is 0.")

        ax.hist(
            process_data["ww_prob"],
            bins=bins,
            range=(0, 1),
            weights=process_data["Weight"] / weight_sum,
            histtype="step",
            linewidth=2.5,
            label=PROCESS_LABELS[process],
        )

    ax.axvline(
        best_cut,
        linestyle="--",
        color="black",
        linewidth=2.5,
        label=f"best cut ({best_cut:.2f})",
    )
    ax.set_xlabel("WW Probability")
    ax.set_ylabel("Fractional Counts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_significance(thresholds, significances, best_cut, max_significance, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(thresholds, significances, linewidth=2.5)
    ax.axvline(
        best_cut,
        linestyle="--",
        color="black",
        linewidth=2.5,
        label=f"best cut ({best_cut:.2f})",
    )
    ax.scatter([best_cut], [max_significance], color="red", zorder=3)
    ax.set_xlabel("WW Probability Cut")
    ax.set_ylabel(r"$S/\sqrt{S+B}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    pickle_file = Path(args.pickle_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix if args.prefix else pickle_file.stem
    distribution_path = output_dir / f"{prefix}_ww_score_distribution.png"
    significance_path = output_dir / f"{prefix}_significance.png"

    imported_bdt = load_bdt(pickle_file)
    classifier = imported_bdt["Model"]
    feature_names = imported_bdt["Features"]

    required_columns = {"Process", "Weight"}
    missing_columns = required_columns - set(df_tot.columns)
    if missing_columns:
        raise ValueError(
            "Input dataframe is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    dataframe = df_tot[df_tot["Process"].isin(["ww", "dfdy", "ttbar"])]
    results = run_bdt(classifier, dataframe, feature_names)
    # To score only the saved test dataframe instead of all df_tot, use:
    # results = run_bdt(classifier, imported_bdt["Test Dataframe"], feature_names)

    best_cut, max_significance, thresholds, significances = calculate_significances(
        results, args.step
    )

    plot_score_distribution(results, best_cut, args.bins, distribution_path)
    plot_significance(
        thresholds, significances, best_cut, max_significance, significance_path
    )

    print("\nBDT WW score summary")
    print("=" * len("BDT WW score summary"))
    print(f"Pickle file: {pickle_file}")
    print(f"Best cut: {best_cut:.4f}")
    print(f"Maximum S/sqrt(S+B): {max_significance:.4f}")
    print(f"Score distribution plot: {distribution_path}")
    print(f"Significance plot: {significance_path}")


if __name__ == "__main__":
    main()
