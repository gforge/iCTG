from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_INPUT_DIR = "artifacts_ctg3/tcn_multimodal_60m"
DEFAULT_OUTPUT = "figures/generated/outcome_prevalence.png"

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


def load_split(path: Path) -> list[dict[str, object]]:
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
                "outcome": outcome,
                "positives": positives,
                "total": total,
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
                "outcome": name,
                "positives": positives,
                "total": total,
            }
        )

    return rows


def aggregate_prevalence(input_dir: Path, split: str) -> list[dict[str, object]]:
    split_names = ["train", "val", "test"] if split == "all" else [split]
    totals: dict[str, dict[str, int]] = {}
    for split_name in split_names:
        path = input_dir / f"{split_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing NPZ file: {path}")
        for row in load_split(path):
            outcome = str(row["outcome"])
            target = totals.setdefault(outcome, {"positives": 0, "total": 0})
            target["positives"] += int(row["positives"])
            target["total"] += int(row["total"])

    rows = []
    for outcome in OUTCOME_ORDER:
        if outcome not in totals:
            continue
        positives = totals[outcome]["positives"]
        total = totals[outcome]["total"]
        prevalence = positives / total if total else float("nan")
        rows.append(
            {
                "outcome": outcome,
                "label": OUTCOME_LABELS[outcome],
                "positives": positives,
                "total": total,
                "prevalence": prevalence,
            }
        )
    return rows


def plot_prevalence(
    rows: list[dict[str, object]],
    output_path: Path,
    title: str,
    sort: bool,
) -> None:
    if sort:
        rows = sorted(rows, key=lambda row: float(row["prevalence"]), reverse=True)

    labels = [str(row["label"]) for row in rows]
    prevalence_pct = np.array([100.0 * float(row["prevalence"]) for row in rows])
    annotations = [
        f"{int(row['positives'])}/{int(row['total'])} ({100 * float(row['prevalence']):.2f}%)"
        for row in rows
    ]

    fig_height = max(5.2, 0.45 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y = np.arange(len(rows))

    colors = ["#2B6CB0" if "Apgar" in label or "pH" in label else "#2C7A7B" for label in labels]
    ax.barh(y, prevalence_pct, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Prevalence (%)", fontsize=11)
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.grid(axis="x", color="#CBD5E0", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    x_max = max(float(prevalence_pct.max()) * 1.35, 6.0)
    ax.set_xlim(0, x_max)
    for idx, (value, text) in enumerate(zip(prevalence_pct, annotations, strict=True)):
        ax.text(value + x_max * 0.015, idx, text, va="center", ha="left", fontsize=9)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#718096")

    note = (
        "Counts are calculated among valid labels. For pH < 7, the denominator is lower "
        "because umbilical pH was not available for all cases."
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color="#2D3748")
    fig.subplots_adjust(left=0.28, right=0.94, top=0.88, bottom=0.16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot binary outcome prevalence.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--keep-order", action="store_true", help="Use configured outcome order.")
    args = parser.parse_args()

    rows = aggregate_prevalence(Path(args.input_dir), args.split)
    split_label = "all processed splits" if args.split == "all" else f"{args.split} split"
    title = f"Binary outcome prevalence ({split_label})"
    plot_prevalence(rows, Path(args.output), title, sort=not args.keep_order)
    print(f"Wrote prevalence plot: {args.output}")


if __name__ == "__main__":
    main()
