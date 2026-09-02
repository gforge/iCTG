from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ctg_ml.multimodal_config import load_multimodal_config  # noqa: E402
from ctg_ml.multimodal_registry import _clean_boolean_series, load_registry_for_multimodal  # noqa: E402


DEFAULT_OUTPUT = "figures/generated/registry_input_summary.png"

FEATURE_LABELS = {
    "maternal_age": "Maternal age",
    "gestational_days": "Gestational age",
    "etablerade_varkar_seconds": "Labor duration",
    "para_mhv1": "Parity",
    "langd_inskrivning_cm": "Maternal height",
    "bmi_inskrivning": "Maternal BMI",
    "alkohol_audit_poang": "Alcohol audit score",
    "is_smoker": "Tobacco use",
    "diabetes_mellitus": "Diabetes mellitus",
    "previous_c_section": "Previous C-section",
    "is_girl": "Infant sex: female",
    "gestational_hypertension_without_significant_proteinuria": "Gestational hypertension",
    "preeclampsia": "Preeclampsia",
    "gestational_or_pregestational_diabetes": "Gest./pregest. diabetes",
    "heavy_vaginal_bleeding_before_or_during_delivery": "Heavy bleeding",
    "labor_dystocia": "Labor dystocia",
    "use_of_oxytocin": "Oxytocin use",
    "forlossningsstart": "Labor onset",
    "fodelseland": "Country of birth",
    "utbildningsniva": "Education",
    "tobak_3_manader_fore_graviditet": "Tobacco before pregnancy",
    "tobak_inskrivning": "Tobacco at registration",
    "tobak_vecka_30_32": "Tobacco week 30-32",
}


def label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def build_summary(config_path: str, registry_csv: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_multimodal_config(config_path)
    csv_path = registry_csv or str(cfg.paths.registry_csv)
    df = load_registry_for_multimodal(csv_path, cfg.registry)

    input_columns = [
        *cfg.registry.input_numeric,
        *cfg.registry.input_boolean,
        *cfg.registry.input_categorical,
    ]
    missing_rows = []
    for col in input_columns:
        if col not in df.columns:
            continue
        missing_rows.append(
            {
                "variable": col,
                "label": label(col),
                "missing_fraction": float(df[col].isna().mean()),
                "available_fraction": float(df[col].notna().mean()),
                "kind": (
                    "numeric"
                    if col in cfg.registry.input_numeric
                    else "boolean"
                    if col in cfg.registry.input_boolean
                    else "categorical"
                ),
            }
        )

    boolean_rows = []
    for col in cfg.registry.input_boolean:
        if col not in df.columns:
            continue
        clean = _clean_boolean_series(df[col])
        known = clean.notna()
        known_n = int(known.sum())
        true_n = int(clean[known].fillna(False).sum())
        boolean_rows.append(
            {
                "variable": col,
                "label": label(col),
                "known_n": known_n,
                "true_n": true_n,
                "true_fraction_all": float(clean.fillna(False).mean()),
                "true_fraction_known": true_n / known_n if known_n else float("nan"),
            }
        )

    missing_summary = pd.DataFrame(missing_rows).sort_values(
        ["missing_fraction", "label"], ascending=[False, True]
    )
    boolean_summary = pd.DataFrame(boolean_rows).sort_values("true_fraction_all", ascending=False)
    return missing_summary, boolean_summary


def plot_summary(
    missing_summary: pd.DataFrame,
    boolean_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2), gridspec_kw={"width_ratios": [1, 1]})

    missing_plot = missing_summary[missing_summary["missing_fraction"] > 0].copy()
    if missing_plot.empty:
        missing_plot = missing_summary.copy()
    missing_plot = missing_plot.sort_values("missing_fraction", ascending=True)
    y_missing = np.arange(len(missing_plot))
    kind_colors = {
        "numeric": "#2B6CB0",
        "boolean": "#2C7A7B",
        "categorical": "#C05621",
    }
    axes[0].barh(
        y_missing,
        100 * missing_plot["missing_fraction"],
        color=[kind_colors[kind] for kind in missing_plot["kind"]],
        alpha=0.9,
    )
    axes[0].set_yticks(y_missing)
    axes[0].set_yticklabels(missing_plot["label"], fontsize=9)
    axes[0].set_xlabel("Missing values (%)")
    axes[0].set_title("A. Missingness of registry inputs", fontsize=13, weight="bold")
    axes[0].grid(axis="x", color="#CBD5E0", linewidth=0.8, alpha=0.7)
    axes[0].set_axisbelow(True)
    max_missing = max(5.0, 100 * float(missing_plot["missing_fraction"].max()) * 1.25)
    axes[0].set_xlim(0, max_missing)
    for idx, value in enumerate(100 * missing_plot["missing_fraction"]):
        axes[0].text(value + max_missing * 0.015, idx, f"{value:.1f}", va="center", fontsize=8)

    bool_plot = boolean_summary.sort_values("true_fraction_all", ascending=True)
    y_bool = np.arange(len(bool_plot))
    axes[1].barh(y_bool, 100 * bool_plot["true_fraction_all"], color="#2C7A7B", alpha=0.9)
    axes[1].set_yticks(y_bool)
    axes[1].set_yticklabels(bool_plot["label"], fontsize=9)
    axes[1].set_xlabel("True / positive values (%)")
    axes[1].set_title("B. Prevalence of boolean registry inputs", fontsize=13, weight="bold")
    axes[1].grid(axis="x", color="#CBD5E0", linewidth=0.8, alpha=0.7)
    axes[1].set_axisbelow(True)
    max_bool = max(5.0, 100 * float(bool_plot["true_fraction_all"].max()) * 1.25)
    axes[1].set_xlim(0, max_bool)
    for idx, row in enumerate(bool_plot.itertuples(index=False)):
        value = 100 * float(row.true_fraction_all)
        axes[1].text(
            value + max_bool * 0.015,
            idx,
            f"{value:.1f}",
            va="center",
            fontsize=8,
        )

    for ax in axes:
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#718096")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=kind_colors["numeric"], label="Numeric"),
        plt.Rectangle((0, 0), 1, 1, color=kind_colors["boolean"], label="Boolean"),
        plt.Rectangle((0, 0), 1, 1, color=kind_colors["categorical"], label="Categorical"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Registry input availability and boolean prevalence", fontsize=15, weight="bold")
    fig.text(
        0.01,
        0.01,
        "Panel B shows prevalence among all rows after missing booleans are treated as "
        "false for plotting. "
        "Categorical and numeric inputs are summarized by missingness rather than prevalence.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#2D3748",
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.13, wspace=0.45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot registry input summary figures.")
    parser.add_argument("--config", default="configs/ctg3_multimodal.toml")
    parser.add_argument("--registry-csv", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    missing_summary, boolean_summary = build_summary(args.config, args.registry_csv)
    plot_summary(missing_summary, boolean_summary, Path(args.output))
    print(f"Wrote registry input summary: {args.output}")


if __name__ == "__main__":
    main()
