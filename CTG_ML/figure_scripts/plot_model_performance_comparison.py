from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = "figures/generated"

OUTCOME_ORDER = [
    "Apgar <7 after 1 min",
    "Apgar <7 after 5 min",
    "Apgar <7 after 10 min",
    "Umbilical pH <7",
    "Shoulder dystocia",
    "Hypoglycemia treatment",
    "Neonatal sepsis or pneumonia",
    "Neonatal anemia",
    "Respirator treatment",
]

MODEL_ORDER = ["Full multimodal", "Registry only", "CTG only"]

MODEL_COLORS = {
    "Full multimodal": "#2B6CB0",
    "Registry only": "#2C7A7B",
    "CTG only": "#C05621",
}

RESULTS = [
    {
        "model": "Full multimodal",
        "outcome": "Apgar <7 after 1 min",
        "prevalence": 0.0478,
        "roc_auc_mean": 0.7687,
        "roc_auc_sd": 0.0177,
        "pr_auc_mean": 0.1557,
        "pr_auc_sd": 0.0099,
    },
    {
        "model": "Full multimodal",
        "outcome": "Apgar <7 after 5 min",
        "prevalence": 0.0114,
        "roc_auc_mean": 0.7854,
        "roc_auc_sd": 0.0066,
        "pr_auc_mean": 0.0854,
        "pr_auc_sd": 0.0129,
    },
    {
        "model": "Full multimodal",
        "outcome": "Apgar <7 after 10 min",
        "prevalence": 0.0019,
        "roc_auc_mean": 0.8710,
        "roc_auc_sd": 0.0221,
        "pr_auc_mean": 0.0547,
        "pr_auc_sd": 0.0216,
    },
    {
        "model": "Full multimodal",
        "outcome": "Umbilical pH <7",
        "prevalence": 0.0094,
        "roc_auc_mean": 0.6841,
        "roc_auc_sd": 0.0383,
        "pr_auc_mean": 0.0219,
        "pr_auc_sd": 0.0044,
    },
    {
        "model": "Full multimodal",
        "outcome": "Shoulder dystocia",
        "prevalence": 0.0125,
        "roc_auc_mean": 0.8016,
        "roc_auc_sd": 0.0055,
        "pr_auc_mean": 0.0686,
        "pr_auc_sd": 0.0110,
    },
    {
        "model": "Full multimodal",
        "outcome": "Hypoglycemia treatment",
        "prevalence": 0.0218,
        "roc_auc_mean": 0.7932,
        "roc_auc_sd": 0.0048,
        "pr_auc_mean": 0.1609,
        "pr_auc_sd": 0.0131,
    },
    {
        "model": "Full multimodal",
        "outcome": "Neonatal sepsis or pneumonia",
        "prevalence": 0.0041,
        "roc_auc_mean": 0.7470,
        "roc_auc_sd": 0.0058,
        "pr_auc_mean": 0.1553,
        "pr_auc_sd": 0.0135,
    },
    {
        "model": "Full multimodal",
        "outcome": "Neonatal anemia",
        "prevalence": 0.0041,
        "roc_auc_mean": 0.9661,
        "roc_auc_sd": 0.0075,
        "pr_auc_mean": 0.5120,
        "pr_auc_sd": 0.0134,
    },
    {
        "model": "Full multimodal",
        "outcome": "Respirator treatment",
        "prevalence": 0.0268,
        "roc_auc_mean": 0.7368,
        "roc_auc_sd": 0.0042,
        "pr_auc_mean": 0.1368,
        "pr_auc_sd": 0.0051,
    },
    {
        "model": "Registry only",
        "outcome": "Apgar <7 after 1 min",
        "prevalence": 0.0478,
        "roc_auc_mean": 0.7249,
        "roc_auc_sd": 0.0222,
        "pr_auc_mean": 0.1411,
        "pr_auc_sd": 0.0095,
    },
    {
        "model": "Registry only",
        "outcome": "Apgar <7 after 5 min",
        "prevalence": 0.0114,
        "roc_auc_mean": 0.7443,
        "roc_auc_sd": 0.0138,
        "pr_auc_mean": 0.0756,
        "pr_auc_sd": 0.0107,
    },
    {
        "model": "Registry only",
        "outcome": "Apgar <7 after 10 min",
        "prevalence": 0.0019,
        "roc_auc_mean": 0.8915,
        "roc_auc_sd": 0.0313,
        "pr_auc_mean": 0.0417,
        "pr_auc_sd": 0.0247,
    },
    {
        "model": "Registry only",
        "outcome": "Umbilical pH <7",
        "prevalence": 0.0094,
        "roc_auc_mean": 0.5559,
        "roc_auc_sd": 0.0289,
        "pr_auc_mean": 0.0149,
        "pr_auc_sd": 0.0022,
    },
    {
        "model": "Registry only",
        "outcome": "Shoulder dystocia",
        "prevalence": 0.0125,
        "roc_auc_mean": 0.7766,
        "roc_auc_sd": 0.0190,
        "pr_auc_mean": 0.0465,
        "pr_auc_sd": 0.0087,
    },
    {
        "model": "Registry only",
        "outcome": "Hypoglycemia treatment",
        "prevalence": 0.0218,
        "roc_auc_mean": 0.7819,
        "roc_auc_sd": 0.0064,
        "pr_auc_mean": 0.1531,
        "pr_auc_sd": 0.0058,
    },
    {
        "model": "Registry only",
        "outcome": "Neonatal sepsis or pneumonia",
        "prevalence": 0.0041,
        "roc_auc_mean": 0.7604,
        "roc_auc_sd": 0.0362,
        "pr_auc_mean": 0.1307,
        "pr_auc_sd": 0.0196,
    },
    {
        "model": "Registry only",
        "outcome": "Neonatal anemia",
        "prevalence": 0.0041,
        "roc_auc_mean": 0.9479,
        "roc_auc_sd": 0.0094,
        "pr_auc_mean": 0.5232,
        "pr_auc_sd": 0.0213,
    },
    {
        "model": "Registry only",
        "outcome": "Respirator treatment",
        "prevalence": 0.0268,
        "roc_auc_mean": 0.7105,
        "roc_auc_sd": 0.0227,
        "pr_auc_mean": 0.1271,
        "pr_auc_sd": 0.0061,
    },
    {
        "model": "CTG only",
        "outcome": "Apgar <7 after 1 min",
        "prevalence": 0.0478,
        "roc_auc_mean": 0.7116,
        "roc_auc_sd": 0.0089,
        "pr_auc_mean": 0.1207,
        "pr_auc_sd": 0.0061,
    },
    {
        "model": "CTG only",
        "outcome": "Apgar <7 after 5 min",
        "prevalence": 0.0114,
        "roc_auc_mean": 0.7357,
        "roc_auc_sd": 0.0336,
        "pr_auc_mean": 0.0540,
        "pr_auc_sd": 0.0119,
    },
    {
        "model": "CTG only",
        "outcome": "Apgar <7 after 10 min",
        "prevalence": 0.0019,
        "roc_auc_mean": 0.8098,
        "roc_auc_sd": 0.0538,
        "pr_auc_mean": 0.0200,
        "pr_auc_sd": 0.0097,
    },
    {
        "model": "CTG only",
        "outcome": "Umbilical pH <7",
        "prevalence": 0.0094,
        "roc_auc_mean": 0.6621,
        "roc_auc_sd": 0.0629,
        "pr_auc_mean": 0.0505,
        "pr_auc_sd": 0.0220,
    },
    {
        "model": "CTG only",
        "outcome": "Shoulder dystocia",
        "prevalence": 0.0125,
        "roc_auc_mean": 0.7003,
        "roc_auc_sd": 0.0443,
        "pr_auc_mean": 0.0802,
        "pr_auc_sd": 0.0273,
    },
    {
        "model": "CTG only",
        "outcome": "Hypoglycemia treatment",
        "prevalence": 0.0218,
        "roc_auc_mean": 0.6633,
        "roc_auc_sd": 0.0094,
        "pr_auc_mean": 0.0697,
        "pr_auc_sd": 0.0092,
    },
    {
        "model": "CTG only",
        "outcome": "Neonatal sepsis or pneumonia",
        "prevalence": 0.0041,
        "roc_auc_mean": 0.5376,
        "roc_auc_sd": 0.0500,
        "pr_auc_mean": 0.0157,
        "pr_auc_sd": 0.0139,
    },
    {
        "model": "CTG only",
        "outcome": "Neonatal anemia",
        "prevalence": 0.0041,
        "roc_auc_mean": 0.8885,
        "roc_auc_sd": 0.0214,
        "pr_auc_mean": 0.1246,
        "pr_auc_sd": 0.0242,
    },
    {
        "model": "CTG only",
        "outcome": "Respirator treatment",
        "prevalence": 0.0268,
        "roc_auc_mean": 0.6648,
        "roc_auc_sd": 0.0197,
        "pr_auc_mean": 0.0648,
        "pr_auc_sd": 0.0117,
    },
]


def results_df() -> pd.DataFrame:
    df = pd.DataFrame(RESULTS)
    df["outcome"] = pd.Categorical(df["outcome"], categories=OUTCOME_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["outcome", "model"]).reset_index(drop=True)


def plot_grouped_pr_auc(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    exclude_outcomes: set[str] | None = None,
) -> None:
    if exclude_outcomes:
        df = df[~df["outcome"].astype(str).isin(exclude_outcomes)].copy()

    outcomes = [outcome for outcome in OUTCOME_ORDER if outcome in set(df["outcome"].astype(str))]
    prevalence_by_outcome = (
        df.drop_duplicates("outcome")
        .assign(outcome_str=lambda x: x["outcome"].astype(str))
        .set_index("outcome_str")["prevalence"]
        .to_dict()
    )
    y = np.arange(len(outcomes), dtype=float)
    bar_height = 0.22
    offsets = {
        "Full multimodal": -bar_height,
        "Registry only": 0.0,
        "CTG only": bar_height,
    }

    fig_height = max(5.4, 0.52 * len(outcomes) + 1.7)
    fig, ax = plt.subplots(figsize=(11.2, fig_height))

    for model in MODEL_ORDER:
        model_mask = df["model"].astype(str) == model
        part = df[model_mask].set_index(df.loc[model_mask, "outcome"].astype(str))
        means = np.array([float(part.loc[outcome, "pr_auc_mean"]) for outcome in outcomes])
        sds = np.array([float(part.loc[outcome, "pr_auc_sd"]) for outcome in outcomes])
        ax.barh(
            y + offsets[model],
            means,
            height=bar_height,
            xerr=sds,
            color=MODEL_COLORS[model],
            alpha=0.92,
            label=model,
            error_kw={"elinewidth": 1.1, "capsize": 2.5, "capthick": 1.1},
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{outcome}\nprev. {100 * float(prevalence_by_outcome[outcome]):.2f}%"
            for outcome in outcomes
        ],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("PR-AUC, mean +/- SD across five seeds", fontsize=11)
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.grid(axis="x", color="#CBD5E0", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    x_max = max(0.18, float((df["pr_auc_mean"] + df["pr_auc_sd"]).max()) * 1.12)
    ax.set_xlim(0, x_max)
    for idx, outcome in enumerate(outcomes):
        prevalence = float(prevalence_by_outcome[outcome])
        ax.vlines(
            prevalence,
            idx - 0.39,
            idx + 0.39,
            color="#2D3748",
            linewidth=1.2,
            alpha=0.65,
            zorder=4,
        )
    ax.plot(
        [],
        [],
        color="#2D3748",
        linewidth=1.2,
        alpha=0.65,
        label="Prevalence baseline",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=10)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#718096")

    note = (
        "Bars show repeated-seed mean PR-AUC; error bars show standard deviation. "
        "Grey ticks mark outcome prevalence, the baseline expected PR-AUC for random ranking."
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color="#2D3748")
    fig.subplots_adjust(left=0.30, right=0.96, top=0.88, bottom=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot repeated-seed PR-AUC comparison across model variants."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df = results_df()
    plot_grouped_pr_auc(
        df,
        output_dir / "model_performance_pr_auc.png",
        "Binary outcome prediction by model input modality",
    )
    plot_grouped_pr_auc(
        df,
        output_dir / "model_performance_pr_auc_zoom_no_anemia.png",
        "Binary outcome prediction by model input modality, excluding neonatal anemia",
        exclude_outcomes={"Neonatal anemia"},
    )
    print(f"Wrote performance figures to {output_dir}")


if __name__ == "__main__":
    main()
