from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_INPUT_DIR = "artifacts_ctg3/tcn_multimodal_60m"
DEFAULT_OUTPUT = "figures/generated/train_val_test_outcome_balance.png"

OUTCOME_LABELS = {
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

OUTCOME_ORDER = list(OUTCOME_LABELS)
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {
    "train": "#2B6CB0",
    "val": "#2C7A7B",
    "test": "#C05621",
}


def split_rows(path: Path, split_name: str) -> list[dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    rows: list[dict[str, object]] = []

    apgar_names = [str(x) for x in data["apgar_target_names"].tolist()]
    y_apgar = data["y_apgar"]
    y_apgar_mask = data["y_apgar_mask"].astype(bool)
    for idx, name in enumerate(apgar_names):
        outcome = f"{name}_below7"
        valid = y_apgar_mask[:, idx]
        positives = int((y_apgar[valid, idx] < 7).sum())
        total = int(valid.sum())
        rows.append(
            {
                "split": split_name,
                "outcome": outcome,
                "label": OUTCOME_LABELS[outcome],
                "positives": positives,
                "total": total,
                "prevalence": positives / total if total else float("nan"),
            }
        )

    binary_names = [str(x) for x in data["binary_target_names"].tolist()]
    y_bin = data["y_bin"]
    y_bin_mask = data["y_bin_mask"].astype(bool)
    for idx, name in enumerate(binary_names):
        valid = y_bin_mask[:, idx]
        positives = int(y_bin[valid, idx].sum())
        total = int(valid.sum())
        rows.append(
            {
                "split": split_name,
                "outcome": name,
                "label": OUTCOME_LABELS[name],
                "positives": positives,
                "total": total,
                "prevalence": positives / total if total else float("nan"),
            }
        )
    return rows


def load_balance(input_dir: Path) -> pd.DataFrame:
    rows = []
    for split_name in SPLIT_ORDER:
        path = input_dir / f"{split_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing NPZ file: {path}")
        rows.extend(split_rows(path, split_name))
    df = pd.DataFrame(rows)
    df["outcome"] = pd.Categorical(df["outcome"], categories=OUTCOME_ORDER, ordered=True)
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    return df.sort_values(["outcome", "split"]).reset_index(drop=True)


def plot_balance(df: pd.DataFrame, output_path: Path) -> None:
    outcomes = [outcome for outcome in OUTCOME_ORDER if outcome in set(df["outcome"].astype(str))]
    y = np.arange(len(outcomes), dtype=float)
    bar_height = 0.22
    offsets = {"train": -bar_height, "val": 0.0, "test": bar_height}

    fig_height = max(5.4, 0.52 * len(outcomes) + 1.7)
    fig, ax = plt.subplots(figsize=(11.2, fig_height))

    for split_name in SPLIT_ORDER:
        mask = df["split"].astype(str) == split_name
        part = df[mask].set_index(df.loc[mask, "outcome"].astype(str))
        values = np.array([100 * float(part.loc[outcome, "prevalence"]) for outcome in outcomes])
        ax.barh(
            y + offsets[split_name],
            values,
            height=bar_height,
            color=SPLIT_COLORS[split_name],
            alpha=0.92,
            label=split_name.capitalize(),
        )

    ax.set_yticks(y)
    ax.set_yticklabels([OUTCOME_LABELS[outcome] for outcome in outcomes], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Prevalence within split (%)", fontsize=11)
    ax.set_title("Outcome prevalence by train/validation/test split", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#CBD5E0", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    x_max = max(6.0, 100 * float(df["prevalence"].max()) * 1.25)
    ax.set_xlim(0, x_max)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#718096")

    note = (
        "Apgar 5 < 7 was used for stratified splitting. Other rare outcomes were not "
        "explicitly stratified and may vary more between splits."
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color="#2D3748")
    fig.subplots_adjust(left=0.27, right=0.96, top=0.88, bottom=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val/test outcome balance.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_balance(Path(args.input_dir))
    plot_balance(df, Path(args.output))
    print(f"Wrote split balance plot: {args.output}")


if __name__ == "__main__":
    main()
