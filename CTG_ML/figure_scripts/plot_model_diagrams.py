from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


BLUE = "#2B6CB0"
TEAL = "#2C7A7B"
GREEN = "#2F855A"
ORANGE = "#C05621"
PURPLE = "#6B46C1"
GRAY = "#4A5568"
LIGHT_BLUE = "#EBF8FF"
LIGHT_TEAL = "#E6FFFA"
LIGHT_GREEN = "#F0FFF4"
LIGHT_ORANGE = "#FFFAF0"
LIGHT_PURPLE = "#FAF5FF"
LIGHT_GRAY = "#F7FAFC"


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: int = 10,
    weight: str = "normal",
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1A202C",
        weight=weight,
        linespacing=1.18,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GRAY,
    style: str = "-|>",
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=1.4,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.6, 7.2))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(
        7.3,
        6.85,
        "Multimodal multitask TCN architecture",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="#1A202C",
    )

    add_box(
        ax,
        (0.45, 4.65),
        (2.35, 1.0),
        "Last-hour CTG\n5 channels x 3600 s\nFHR, toco, quality,\npadding mask",
        LIGHT_BLUE,
        BLUE,
    )
    add_box(
        ax,
        (3.25, 4.65),
        (2.35, 1.0),
        "TCN sequence encoder\n8 residual dilated\nConv1D blocks",
        LIGHT_BLUE,
        BLUE,
    )
    add_box(
        ax,
        (6.05, 4.65),
        (2.0, 1.0),
        "Pooled CTG\nembedding\n128 features",
        LIGHT_BLUE,
        BLUE,
    )

    add_box(
        ax,
        (0.45, 2.25),
        (2.35, 1.0),
        "Registry variables\nnumeric, boolean,\ncategorical inputs",
        LIGHT_TEAL,
        TEAL,
    )
    add_box(
        ax,
        (3.25, 2.25),
        (2.35, 1.0),
        "Tabular encoder\nLinear 79 -> 64\nReLU + dropout",
        LIGHT_TEAL,
        TEAL,
    )
    add_box(
        ax,
        (6.05, 2.25),
        (2.0, 1.0),
        "Registry\nembedding\n64 features",
        LIGHT_TEAL,
        TEAL,
    )

    add_box(
        ax,
        (8.55, 3.45),
        (1.65, 1.0),
        "Concatenate\n128 + 64",
        LIGHT_GRAY,
        GRAY,
    )
    add_box(
        ax,
        (10.75, 3.45),
        (1.65, 1.0),
        "Fusion layer\nLinear -> 128\nReLU + dropout",
        LIGHT_GRAY,
        GRAY,
    )

    output_y = [5.7, 4.7, 3.7, 2.7]
    output_text = [
        "Apgar heads\n3 x 11 classes",
        "pH regression\nartery + vein",
        "Binary outcomes\n6 logits",
        "Delivery mode\ncategorical head",
    ]
    output_colors = [
        (LIGHT_PURPLE, PURPLE),
        (LIGHT_GREEN, GREEN),
        (LIGHT_ORANGE, ORANGE),
        (LIGHT_GRAY, GRAY),
    ]
    for y, text, (face, edge) in zip(output_y, output_text, output_colors, strict=True):
        add_box(ax, (12.95, y), (1.35, 0.68), text, face, edge, fontsize=9)

    add_arrow(ax, (2.8, 5.15), (3.25, 5.15), BLUE)
    add_arrow(ax, (5.6, 5.15), (6.05, 5.15), BLUE)
    add_arrow(ax, (2.8, 2.75), (3.25, 2.75), TEAL)
    add_arrow(ax, (5.6, 2.75), (6.05, 2.75), TEAL)
    add_arrow(ax, (8.05, 5.15), (8.55, 4.1), BLUE)
    add_arrow(ax, (8.05, 2.75), (8.55, 3.8), TEAL)
    add_arrow(ax, (10.2, 3.95), (10.75, 3.95), GRAY)

    for y in output_y:
        add_arrow(ax, (12.4, 3.95), (12.95, y + 0.34), GRAY, rad=0.0)

    ax.text(
        0.45,
        0.55,
        "The model jointly optimizes classification, regression, and binary prediction heads. "
        "Ablation experiments replace either CTG or registry inputs while retaining the "
        "same architecture.",
        ha="left",
        va="center",
        fontsize=9,
        color="#2D3748",
    )

    save_figure(fig, output_dir, "model_architecture")


def plot_tcn_block(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    ax.text(
        5.4,
        5.45,
        "Residual dilated TCN block",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="#1A202C",
    )

    add_box(ax, (0.45, 2.35), (1.15, 0.75), "Input\nsequence", LIGHT_GRAY, GRAY)
    add_box(ax, (2.0, 2.35), (1.3, 0.75), "Dilated\nConv1D", LIGHT_BLUE, BLUE)
    add_box(ax, (3.7, 2.35), (1.05, 0.75), "Chomp\n+ ReLU", LIGHT_BLUE, BLUE)
    add_box(ax, (5.15, 2.35), (1.0, 0.75), "Dropout", LIGHT_BLUE, BLUE)
    add_box(ax, (6.55, 2.35), (1.3, 0.75), "Dilated\nConv1D", LIGHT_BLUE, BLUE)
    add_box(ax, (8.25, 2.35), (1.05, 0.75), "Chomp\n+ ReLU", LIGHT_BLUE, BLUE)
    add_box(ax, (9.7, 2.35), (0.75, 0.75), "Output", LIGHT_GRAY, GRAY)

    add_arrow(ax, (1.6, 2.72), (2.0, 2.72), GRAY)
    add_arrow(ax, (3.3, 2.72), (3.7, 2.72), GRAY)
    add_arrow(ax, (4.75, 2.72), (5.15, 2.72), GRAY)
    add_arrow(ax, (6.15, 2.72), (6.55, 2.72), GRAY)
    add_arrow(ax, (7.85, 2.72), (8.25, 2.72), GRAY)
    add_arrow(ax, (9.3, 2.72), (9.7, 2.72), GRAY)

    ax.plot([1.05, 1.05, 9.55, 9.55], [2.35, 1.25, 1.25, 2.35], color=TEAL, linewidth=1.6)
    add_arrow(ax, (9.55, 1.25), (9.55, 2.35), TEAL)
    ax.text(
        5.4,
        1.0,
        "Residual connection, with 1x1 convolution if channel dimensions change",
        ha="center",
        va="center",
        fontsize=9,
        color=TEAL,
    )

    add_box(
        ax,
        (1.6, 4.05),
        (7.65, 0.55),
        "Dilation increases across stacked blocks: 1, 2, 4, 8, 16, 32, 64, 128",
        LIGHT_ORANGE,
        ORANGE,
        fontsize=10,
    )
    add_arrow(ax, (5.4, 4.05), (5.4, 3.18), ORANGE)

    ax.text(
        0.45,
        0.35,
        "Chomp removes right-side padding introduced by causal/dilated convolution, "
        "preventing future time steps from entering the representation.",
        ha="left",
        va="center",
        fontsize=9,
        color="#2D3748",
    )

    save_figure(fig, output_dir, "tcn_residual_block")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model architecture diagrams.")
    parser.add_argument("--output-dir", default="figures/generated")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_architecture(output_dir)
    plot_tcn_block(output_dir)
    print(f"Wrote diagrams to {output_dir}")


if __name__ == "__main__":
    main()
