from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_INPUT = "artifacts_ctg3/xgboost_registry_plus_ctg_embedding/xgboost_grouped_importance.csv"
DEFAULT_OUTPUT = "artifacts_ctg3/xgboost_registry_plus_ctg_embedding/xgboost_importance_heatmap.png"

TARGET_LABELS = {
    "apgar1_below7": "Apgar 1 < 7",
    "apgar5_below7": "Apgar 5 < 7",
    "apgar10_below7": "Apgar 10 < 7",
    "ph_navel_below7": "Umbilical pH < 7",
    "shoulder_dystocia": "Shoulder dystocia",
    "treatment_for_hypoglycemia": "Hypoglycemia treatment",
    "neonatal_sepsis_or_pneumonia": "Sepsis/pneumonia",
    "neonatal_anemia": "Neonatal anemia",
    "respiratorbehandling": "Respiratory treatment",
}

TARGET_ORDER = list(TARGET_LABELS)

FEATURE_LABELS = {
    "ctg_embedding": "CTG embedding",
    "gestational_days": "Gestational age",
    "etablerade_varkar_seconds": "Labor duration",
    "bmi_inskrivning": "Maternal BMI",
    "langd_inskrivning_cm": "Maternal height",
    "maternal_age": "Maternal age",
    "para_mhv1": "Parity",
    "fodelseland": "Country of birth",
    "utbildningsniva": "Education",
    "forlossningsstart": "Labor onset",
    "use_of_oxytocin": "Oxytocin use",
    "labor_dystocia": "Labor dystocia",
    "previous_c_section": "Previous C-section",
    "preeclampsia": "Preeclampsia",
    "gestational_or_pregestational_diabetes": "Gest./pregest. diabetes",
    "diabetes_mellitus": "Diabetes mellitus",
    "gestational_hypertension_without_significant_proteinuria": "Gest. hypertension",
    "heavy_vaginal_bleeding_before_or_during_delivery": "Heavy bleeding",
    "is_girl": "Infant sex",
    "is_smoker": "Tobacco use",
    "alkohol_audit_poang": "Alcohol audit score",
    "tobacco_use": "Tobacco use",
}

FEATURE_GROUPS = {
    "is_smoker": "tobacco_use",
    "tobak_3_manader_fore_graviditet": "tobacco_use",
    "tobak_inskrivning": "tobacco_use",
    "tobak_vecka_30_32": "tobacco_use",
}


def wrap_label(label: str, width: int) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def select_feature_order(matrix: pd.DataFrame, top_n: int) -> list[str]:
    if top_n <= 0 or top_n >= matrix.shape[1]:
        selected = list(matrix.columns)
    else:
        max_importance = matrix.max(axis=0).sort_values(ascending=False)
        selected = max_importance.head(top_n).index.tolist()

    if "ctg_embedding" in matrix.columns:
        selected = [
            "ctg_embedding",
            *[feature for feature in selected if feature != "ctg_embedding"],
        ]
    return selected


def build_matrix(
    input_path: Path,
    top_n: int,
    exclude_features: set[str],
    renormalize_rows: bool,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"target", "raw_feature", "gain_fraction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["raw_feature"] = df["raw_feature"].replace(FEATURE_GROUPS)
    if exclude_features:
        df = df[~df["raw_feature"].isin(exclude_features)].copy()

    matrix = (
        df.pivot_table(
            index="target",
            columns="raw_feature",
            values="gain_fraction",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex([target for target in TARGET_ORDER if target in set(df["target"])])
        .fillna(0.0)
    )
    if renormalize_rows:
        row_sums = matrix.sum(axis=1).replace(0.0, np.nan)
        matrix = matrix.div(row_sums, axis=0).fillna(0.0)
    feature_order = select_feature_order(matrix, top_n)
    return matrix.loc[:, feature_order]


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    transpose: bool,
    annotate: bool,
) -> None:
    plot_matrix = matrix.T if transpose else matrix
    values = plot_matrix.to_numpy(dtype=float)

    row_labels = [
        FEATURE_LABELS.get(idx, idx) if transpose else TARGET_LABELS.get(idx, idx)
        for idx in plot_matrix.index
    ]
    col_labels = [
        TARGET_LABELS.get(col, col) if transpose else FEATURE_LABELS.get(col, col)
        for col in plot_matrix.columns
    ]

    fig_width = max(8.0, 0.62 * len(col_labels) + 3.2)
    fig_height = max(6.5, 0.52 * len(row_labels) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.14, right=0.90, top=0.88, bottom=0.34)

    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("XGBoost grouped gain fraction", fontsize=10)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([wrap_label(label, 16) for label in col_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([wrap_label(label, 24) for label in row_labels])

    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_title(title, fontsize=13, pad=14)

    if annotate:
        threshold = np.nanmax(values) * 0.55 if values.size else 0.0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                val = values[row, col]
                if val <= 0:
                    continue
                color = "white" if val > threshold else "black"
                ax.text(
                    col,
                    row,
                    f"{100 * val:.0f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )

    note = (
        "Cell values are percentages of each outcome's total XGBoost gain. "
        "CTG embedding groups 128 learned TCN features."
    )
    fig.text(0.01, 0.02, note, ha="left", va="bottom", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a heatmap from grouped XGBoost feature importances."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Grouped importance CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of feature columns to show, ranked by max gain fraction.",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Put features on rows and outcomes on columns.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable percent labels inside heatmap cells.",
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Grouped feature to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--registry-only-view",
        action="store_true",
        help=(
            "Exclude CTG embedding and renormalize each row across remaining registry "
            "features. This only changes the plot; it does not retrain XGBoost."
        ),
    )
    parser.add_argument(
        "--title",
        default="XGBoost Importance: Registry Variables and Frozen CTG Embedding",
    )
    args = parser.parse_args()

    exclude_features = set(args.exclude_feature)
    if args.registry_only_view:
        exclude_features.add("ctg_embedding")
    matrix = build_matrix(
        Path(args.input),
        args.top_n,
        exclude_features,
        renormalize_rows=args.registry_only_view,
    )
    plot_heatmap(
        matrix=matrix,
        output_path=Path(args.output),
        title=args.title,
        transpose=args.transpose,
        annotate=not args.no_annotate,
    )
    print(f"Wrote heatmap: {args.output}")


if __name__ == "__main__":
    main()
