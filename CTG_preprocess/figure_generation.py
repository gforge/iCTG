from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402


@dataclass(frozen=True)
class StageData:
    key: str
    label: str
    method: str
    rows: int | None
    patients: int | None
    babies: int | None
    pregnancies: int | None
    row_retention_pct: float | None
    baby_retention_pct: float | None
    registry_overlap_patients: int | None
    files: int | None
    size_mb: float | None
    group: str


# Counts are from the current spreadsheet summary. Stage 1/2 pregnancy counts
# are normalized to 133,793 to avoid over-emphasizing a one-episode estimation
# difference before BabyID assignment.
STAGES: dict[str, StageData] = {
    "raw": StageData(
        key="raw",
        label="Raw CTG parquet",
        method="Raw CTG rows before reduction.",
        rows=8_232_038_694,
        patients=168_397,
        babies=None,
        pregnancies=220_816,
        row_retention_pct=None,
        baby_retention_pct=None,
        registry_overlap_patients=83_640,
        files=None,
        size_mb=88_904.6,
        group="raw",
    ),
    "stage1": StageData(
        key="stage1",
        label="Stage 1: time filter",
        method="Keep rows with Timestamp >= 2014-12-31.",
        rows=5_101_595_694,
        patients=100_077,
        babies=None,
        pregnancies=133_793,
        row_retention_pct=61.97,
        baby_retention_pct=60.59,
        registry_overlap_patients=81_955,
        files=None,
        size_mb=75_802.5,
        group="filter",
    ),
    "stage2": StageData(
        key="stage2",
        label="Stage 2: signal and column reduction",
        method="Keep analysis columns; derive 1 Hz FHR and toco.",
        rows=5_101_595_694,
        patients=100_077,
        babies=None,
        pregnancies=133_793,
        row_retention_pct=100.00,
        baby_retention_pct=100.00,
        registry_overlap_patients=81_955,
        files=None,
        size_mb=49_007.8,
        group="filter",
    ),
    "stage3": StageData(
        key="stage3",
        label="Stage 3: session and pregnancy filter",
        method="Sessionize; keep final 60 min of final CTG session; assign BabyID.",
        rows=403_499_262,
        patients=100_077,
        babies=133_793,
        pregnancies=133_793,
        row_retention_pct=7.91,
        baby_retention_pct=100.00,
        registry_overlap_patients=81_955,
        files=None,
        size_mb=2_415.5,
        group="episode",
    ),
    "stage4": StageData(
        key="stage4",
        label="Stage 4: duplicate filter",
        method="Drop BabyIDs with >30% duplicate timestamps; aggregate duplicates.",
        rows=158_389_086,
        patients=70_991,
        babies=87_632,
        pregnancies=87_632,
        row_retention_pct=39.25,
        baby_retention_pct=65.50,
        registry_overlap_patients=58_443,
        files=None,
        size_mb=1_656.5,
        group="quality",
    ),
    "stage5": StageData(
        key="stage5",
        label="Stage 5: FHR quality filter",
        method="Keep BabyIDs with >=1200 non-zero FHR seconds.",
        rows=133_358_545,
        patients=41_140,
        babies=44_453,
        pregnancies=44_453,
        row_retention_pct=84.20,
        baby_retention_pct=50.73,
        registry_overlap_patients=34_480,
        files=None,
        size_mb=1_382.8,
        group="quality",
    ),
    "stage5_5": StageData(
        key="stage5_5",
        label="Stage 5.5: sort and anchor date",
        method="Add ctg_date from max Timestamp; sort by date, BabyID, Timestamp.",
        rows=133_358_545,
        patients=41_140,
        babies=44_453,
        pregnancies=44_453,
        row_retention_pct=100.00,
        baby_retention_pct=100.00,
        registry_overlap_patients=34_480,
        files=None,
        size_mb=1_069.9,
        group="structure",
    ),
    "stage6": StageData(
        key="stage6",
        label="Stage 6: date partitioning",
        method="Write cleaned CTG dataset partitioned by ctg_date.",
        rows=133_358_545,
        patients=41_140,
        babies=44_453,
        pregnancies=44_453,
        row_retention_pct=100.00,
        baby_retention_pct=100.00,
        registry_overlap_patients=34_480,
        files=None,
        size_mb=1_251.8,
        group="structure",
    ),
    "stage7_match": StageData(
        key="stage7_match",
        label="Stage 7: registry matching",
        method="Match on PatientID and birth day/day before; export anonymized data.",
        rows=None,
        patients=None,
        babies=30_871,
        pregnancies=30_871,
        row_retention_pct=72.59,
        baby_retention_pct=69.45,
        registry_overlap_patients=None,
        files=None,
        size_mb=None,
        group="match",
    ),
    "stage7_ctg": StageData(
        key="stage7_ctg",
        label="Final CTG parquet",
        method="Matched anonymized CTG signal rows.",
        rows=96_804_761,
        patients=None,
        babies=30_871,
        pregnancies=30_871,
        row_retention_pct=72.59,
        baby_retention_pct=69.45,
        registry_overlap_patients=None,
        files=None,
        size_mb=779.6,
        group="output",
    ),
    "stage7_registry": StageData(
        key="stage7_registry",
        label="Final registry CSV",
        method="Matched registry variables; direct identifiers removed.",
        rows=30_871,
        patients=None,
        babies=30_871,
        pregnancies=30_871,
        row_retention_pct=0.03,
        baby_retention_pct=100.00,
        registry_overlap_patients=None,
        files=None,
        size_mb=12.0,
        group="output",
    ),
}

FLOW_KEYS = [
    "raw",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage5_5",
    "stage6",
    "stage7_match",
]

SUMMARY_KEYS = [
    "raw",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage5_5",
    "stage6",
    "stage7_ctg",
]

PALETTE = {
    "raw": {"fill": "#f3f4f6", "edge": "#4b5563"},
    "filter": {"fill": "#e0f2fe", "edge": "#0369a1"},
    "episode": {"fill": "#dcfce7", "edge": "#15803d"},
    "quality": {"fill": "#fef3c7", "edge": "#b45309"},
    "structure": {"fill": "#ede9fe", "edge": "#6d28d9"},
    "match": {"fill": "#fae8ff", "edge": "#a21caf"},
    "output": {"fill": "#fee2e2", "edge": "#b91c1c"},
}


def _format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _format_compact(value: int | float | None) -> str:
    if value is None:
        return "-"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,.0f}"


def _format_size(size_mb: float | None) -> str:
    if size_mb is None:
        return "-"
    if size_mb >= 1000:
        return f"{size_mb / 1000:.2f} GB"
    return f"{size_mb:.1f} MB"


def _cohort_value(stage: StageData) -> int | None:
    return stage.babies if stage.babies is not None else stage.pregnancies


def _cohort_unit(stage: StageData) -> str:
    return "babies" if stage.babies is not None else "pregnancies"


def _stage_lines(stage: StageData) -> list[str]:
    cohort = _cohort_value(stage)
    cohort_unit = _cohort_unit(stage)
    cohort_text = f"{_format_int(cohort)} {cohort_unit}"
    if stage.baby_retention_pct is not None:
        cohort_text += f" ({stage.baby_retention_pct:.2f}% retained)"

    lines = [cohort_text]

    if stage.rows is not None:
        row_text = f"Rows: {_format_compact(stage.rows)}"
        if stage.row_retention_pct is not None:
            row_text += f" ({stage.row_retention_pct:.2f}% retained)"
        lines.append(row_text)
    if stage.patients is not None:
        lines.append(f"Patients: {_format_int(stage.patients)}")
    if stage.registry_overlap_patients is not None:
        lines.append(f"Registry-overlap patients: {_format_int(stage.registry_overlap_patients)}")
    if stage.files is not None and stage.size_mb is not None:
        lines.append(f"Files: {_format_int(stage.files)} | Size: {_format_size(stage.size_mb)}")
    elif stage.files is not None:
        lines.append(f"Files: {_format_int(stage.files)}")
    elif stage.size_mb is not None:
        lines.append(f"Size: {_format_size(stage.size_mb)}")

    return lines


def _transition_lines(previous: StageData, current: StageData) -> list[str]:
    previous_count = _cohort_value(previous)
    current_count = _cohort_value(current)
    unit = "babies" if previous.babies is not None or current.babies is not None else "pregnancies"
    lines: list[str] = []

    if previous_count is not None and current_count is not None:
        removed = previous_count - current_count
        lines.append(f"Removed: {_format_int(max(removed, 0))} {unit}")

    if previous.rows is not None and current.rows is not None:
        removed_rows = previous.rows - current.rows
        if removed_rows > 0:
            lines.append(f"Rows: -{_format_compact(removed_rows)}")

    if previous.size_mb is not None and current.size_mb is not None:
        size_delta = current.size_mb - previous.size_mb
        if size_delta < -1 and (previous.rows == current.rows or current.key == "stage5_5"):
            lines.append(f"Size: -{abs(size_delta):,.1f} MB")

    return lines


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _draw_stage_box(
    ax: plt.Axes,
    stage: StageData,
    x: float,
    y: float,
    width: float,
    height: float,
    wrap_chars: int,
    title_size: float = 8.7,
    text_size: float = 7.2,
) -> None:
    colors = PALETTE[stage.group]
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=colors["fill"],
            edgecolor=colors["edge"],
            linewidth=1.25,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            0.12,
            height,
            facecolor=colors["edge"],
            edgecolor=colors["edge"],
            linewidth=0,
        )
    )

    title_y = y + height - 0.16
    ax.text(
        x + 0.22,
        title_y,
        stage.label,
        ha="left",
        va="top",
        fontsize=title_size,
        weight="bold",
        color="#111827",
    )

    method = "\n".join(textwrap.wrap(stage.method, width=wrap_chars))
    ax.text(
        x + 0.22,
        y + height - 0.42,
        method,
        ha="left",
        va="top",
        fontsize=text_size,
        color="#1f2937",
        linespacing=1.12,
    )

    metrics = " | ".join(_stage_lines(stage))
    metric_lines = "\n".join(textwrap.wrap(metrics, width=wrap_chars + 18))
    ax.text(
        x + 0.22,
        y + 0.12,
        metric_lines,
        ha="left",
        va="bottom",
        fontsize=text_size,
        color="#374151",
        linespacing=1.12,
    )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color="#374151",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="left",
            va="center",
            fontsize=7.0,
            color="#374151",
            bbox={
                "boxstyle": "square,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "linewidth": 0.6,
            },
            linespacing=1.12,
        )


def create_flowchart(output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    _set_publication_style()
    stages = [STAGES[key] for key in FLOW_KEYS]

    box_x = 0.7
    box_width = 7.25
    box_height = 0.98
    y_step = 1.48
    top_y = 13.0
    final_y = -0.65

    fig, ax = plt.subplots(figsize=(10.5, 15.2))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(-1.45, 15.0)
    ax.axis("off")

    ax.text(
        0.7,
        14.75,
        "CTG preprocessing cohort reduction and registry matching",
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
        color="#111827",
    )
    ax.text(
        0.7,
        14.37,
        "Box percentages are stage-to-stage retentions; arrow labels show pregnancy/BabyID exclusions and major row/size reductions.",
        ha="left",
        va="top",
        fontsize=8.2,
        color="#4b5563",
    )

    centers: dict[str, tuple[float, float]] = {}
    for index, stage in enumerate(stages):
        y = top_y - index * y_step
        _draw_stage_box(ax, stage, box_x, y, box_width, box_height, wrap_chars=78)
        centers[stage.key] = (box_x + box_width / 2, y + box_height / 2)

        if index > 0:
            previous = stages[index - 1]
            previous_y = top_y - (index - 1) * y_step
            arrow_start = (box_x + box_width / 2, previous_y)
            arrow_end = (box_x + box_width / 2, y + box_height)
            transition = "\n".join(_transition_lines(previous, stage))
            _draw_arrow(
                ax,
                arrow_start,
                arrow_end,
                transition,
                (8.3, (arrow_start[1] + arrow_end[1]) / 2),
            )

    match_stage = STAGES["stage7_match"]
    match_center = centers[match_stage.key]
    output_width = 4.05
    output_height = 1.15
    left_output_x = 0.7
    right_output_x = 5.0

    ctg_stage = STAGES["stage7_ctg"]
    reg_stage = STAGES["stage7_registry"]
    _draw_stage_box(
        ax,
        ctg_stage,
        left_output_x,
        final_y,
        output_width,
        output_height,
        wrap_chars=42,
        title_size=8.2,
        text_size=6.8,
    )
    _draw_stage_box(
        ax,
        reg_stage,
        right_output_x,
        final_y,
        output_width,
        output_height,
        wrap_chars=42,
        title_size=8.2,
        text_size=6.8,
    )

    branch_start = (match_center[0], top_y - (len(stages) - 1) * y_step)
    left_end = (left_output_x + output_width / 2, final_y + output_height)
    right_end = (right_output_x + output_width / 2, final_y + output_height)
    _draw_arrow(ax, branch_start, left_end)
    _draw_arrow(ax, branch_start, right_end)

    ax.text(
        0.7,
        -1.18,
        "Note: before Stage 3, BabyID does not yet exist, so the cohort denominator is estimated pregnancy episodes. Stage 5.5 and Stage 6 reorganize data without baby-level exclusion.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#4b5563",
        wrap=True,
    )

    return _save_figure(fig, output_dir, "ctg_reduction_flowchart", formats, dpi)


def create_retention_summary(output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    _set_publication_style()
    stages = [STAGES[key] for key in SUMMARY_KEYS]
    labels = ["Raw", "S1", "S2", "S3", "S4", "S5", "S5.5", "S6", "S7 CTG"]
    cohort_counts = [_cohort_value(stage) or 0 for stage in stages]
    row_counts = [stage.rows or 0 for stage in stages]
    sizes = [stage.size_mb or 0 for stage in stages]
    raw_cohort_count = cohort_counts[0] if cohort_counts else 0
    cumulative_pct = [
        (value / raw_cohort_count * 100.0) if raw_cohort_count else 0.0 for value in cohort_counts
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.2),
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.38},
    )

    bar_colors = [PALETTE[stage.group]["edge"] for stage in stages]
    axes[0].bar(labels, cohort_counts, color=bar_colors, alpha=0.82)
    axes[0].set_title(
        "Cumulative pregnancy/BabyID retention by preprocessing stage", loc="left", weight="bold"
    )
    axes[0].set_ylabel("Pregnancies or babies")
    axes[0].grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for x_pos, value, pct in zip(range(len(labels)), cohort_counts, cumulative_pct, strict=True):
        axes[0].text(
            x_pos,
            value + max(cohort_counts) * 0.018,
            f"{_format_compact(value)}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#374151",
        )
    axes[0].set_ylim(0, max(cohort_counts) * 1.17)

    ax_rows = axes[1]
    ax_size = ax_rows.twinx()
    ax_rows.plot(labels, row_counts, marker="o", linewidth=1.8, color="#0369a1", label="CTG rows")
    ax_size.plot(labels, sizes, marker="s", linewidth=1.8, color="#b45309", label="File size")
    ax_rows.set_yscale("log")
    ax_size.set_yscale("log")
    ax_rows.set_ylabel("CTG rows, log scale", color="#0369a1")
    ax_size.set_ylabel("File size in MB, log scale", color="#b45309")
    ax_rows.tick_params(axis="y", labelcolor="#0369a1")
    ax_size.tick_params(axis="y", labelcolor="#b45309")
    ax_rows.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax_rows.set_title("Data volume after each stage", loc="left", weight="bold")

    lines = ax_rows.get_lines() + ax_size.get_lines()
    ax_rows.legend(lines, [line.get_label() for line in lines], loc="upper right", frameon=False)

    fig.suptitle(
        "CTG preprocessing reduction summary",
        x=0.075,
        ha="left",
        y=0.985,
        weight="bold",
        fontsize=14,
    )
    fig.text(
        0.075,
        0.018,
        "Bar percentages are relative to raw pregnancy episodes. Pre-Stage 3 counts are estimated pregnancy episodes; Stage 3 onward counts are BabyID episodes. Final registry CSV has 30,871 rows and 12.0 MB.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#4b5563",
    )

    return _save_figure(fig, output_dir, "ctg_reduction_retention_summary", formats, dpi)


def _draw_plain_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    fill: str,
    edge: str,
    title_size: float = 9.0,
    text_size: float = 7.4,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.25,
        )
    )
    ax.text(
        x + 0.16,
        y + height - 0.18,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        weight="bold",
        color="#111827",
    )
    ax.text(
        x + 0.16,
        y + height - 0.52,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=text_size,
        color="#374151",
        linespacing=1.18,
    )


def create_schema_evolution(output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    _set_publication_style()
    fig, ax = plt.subplots(figsize=(12.4, 9.3))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 9.3)
    ax.axis("off")

    ax.text(
        0.55,
        9.02,
        "CTG data structure and schema evolution",
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
        color="#111827",
    )
    ax.text(
        0.55,
        8.70,
        "Raw device-specific columns are condensed into a cleaned time-series table, then linked to one registry row per BabyID.",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#4b5563",
    )

    columns = [
        "Raw\nCTG",
        "Stage 2\nsignals",
        "Stage 3\nepisodes",
        "Stages 4-6\nclean CTG",
        "Stage 7\nCTG parquet",
        "Stage 7\nregistry CSV",
    ]
    rows: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "PatientID",
            [
                ("internal", "identifier"),
                ("internal", "identifier"),
                ("internal", "identifier"),
                ("internal", "identifier"),
                ("removed", "absent"),
                ("removed", "absent"),
            ],
        ),
        (
            "RegistrationID",
            [
                ("raw ID", "raw"),
                ("kept", "raw"),
                ("kept", "raw"),
                ("removed", "absent"),
                ("removed", "absent"),
                ("removed", "absent"),
            ],
        ),
        (
            "BabyID",
            [
                ("-", "absent"),
                ("-", "absent"),
                ("created", "created"),
                ("kept", "created"),
                ("join key", "exported"),
                ("join key", "exported"),
            ],
        ),
        (
            "Time fields",
            [
                ("Timestamp", "raw"),
                ("Timestamp", "raw"),
                ("final hour", "derived"),
                ("Timestamp\nctg_date", "derived"),
                ("Timestamp", "exported"),
                ("birth dates", "registry"),
            ],
        ),
        (
            "Raw FHR channels",
            [
                ("Hr1_0..3", "raw"),
                ("to FHR", "derived"),
                ("removed", "absent"),
                ("removed", "absent"),
                ("removed", "absent"),
                ("-", "absent"),
            ],
        ),
        (
            "Raw toco payload",
            [
                ("Toco_Values", "raw"),
                ("to toco", "derived"),
                ("removed", "absent"),
                ("removed", "absent"),
                ("removed", "absent"),
                ("-", "absent"),
            ],
        ),
        (
            "Derived CTG signals",
            [
                ("-", "absent"),
                ("FHR\ntoco", "derived"),
                ("FHR\ntoco", "derived"),
                ("cleaned", "derived"),
                ("exported", "exported"),
                ("-", "absent"),
            ],
        ),
        (
            "Signal metadata",
            [
                ("optional", "optional"),
                ("kept if\npresent", "optional"),
                ("kept", "optional"),
                ("kept", "optional"),
                ("optional", "optional"),
                ("-", "absent"),
            ],
        ),
        (
            "Registry variables",
            [
                ("-", "absent"),
                ("-", "absent"),
                ("-", "absent"),
                ("-", "absent"),
                ("-", "absent"),
                ("exported", "registry"),
            ],
        ),
    ]
    style = {
        "raw": {"fill": "#dbeafe", "edge": "#2563eb", "text": "#1e3a8a"},
        "identifier": {"fill": "#ffedd5", "edge": "#ea580c", "text": "#7c2d12"},
        "created": {"fill": "#ede9fe", "edge": "#7c3aed", "text": "#4c1d95"},
        "derived": {"fill": "#dcfce7", "edge": "#16a34a", "text": "#14532d"},
        "exported": {"fill": "#ccfbf1", "edge": "#0f766e", "text": "#134e4a"},
        "registry": {"fill": "#fee2e2", "edge": "#dc2626", "text": "#7f1d1d"},
        "optional": {"fill": "#fef3c7", "edge": "#d97706", "text": "#78350f"},
        "absent": {"fill": "#f9fafb", "edge": "#d1d5db", "text": "#6b7280"},
    }

    matrix_x = 0.55
    matrix_y = 3.55
    label_w = 2.35
    cell_w = 1.50
    row_h = 0.43
    header_h = 0.58

    ax.text(
        matrix_x,
        matrix_y + header_h + len(rows) * row_h + 0.28,
        "A. How column groups change during preprocessing",
        ha="left",
        va="bottom",
        fontsize=10.3,
        weight="bold",
        color="#111827",
    )
    ax.add_patch(
        Rectangle(
            (matrix_x, matrix_y + len(rows) * row_h),
            label_w,
            header_h,
            facecolor="#f3f4f6",
            edgecolor="#d1d5db",
            linewidth=0.8,
        )
    )
    ax.text(
        matrix_x + 0.12,
        matrix_y + len(rows) * row_h + header_h / 2,
        "Column group",
        ha="left",
        va="center",
        fontsize=7.0,
        weight="bold",
        color="#374151",
    )
    for col_idx, col_label in enumerate(columns):
        x = matrix_x + label_w + col_idx * cell_w
        group_key = ["raw", "filter", "episode", "structure", "output", "output"][col_idx]
        ax.add_patch(
            Rectangle(
                (x, matrix_y + len(rows) * row_h),
                cell_w,
                header_h,
                facecolor=PALETTE[group_key]["fill"],
                edgecolor=PALETTE[group_key]["edge"],
                linewidth=0.9,
            )
        )
        ax.text(
            x + cell_w / 2,
            matrix_y + len(rows) * row_h + header_h / 2,
            col_label,
            ha="center",
            va="center",
            fontsize=6.8,
            weight="bold",
            color="#111827",
            linespacing=1.05,
        )

    for row_idx, (row_label, cells) in enumerate(rows):
        y = matrix_y + (len(rows) - row_idx - 1) * row_h
        ax.add_patch(
            Rectangle(
                (matrix_x, y),
                label_w,
                row_h,
                facecolor="#ffffff" if row_idx % 2 else "#f9fafb",
                edgecolor="#d1d5db",
                linewidth=0.6,
            )
        )
        ax.text(
            matrix_x + 0.12,
            y + row_h / 2,
            row_label,
            ha="left",
            va="center",
            fontsize=6.8,
            color="#111827",
        )
        for col_idx, (cell_text, kind) in enumerate(cells):
            x = matrix_x + label_w + col_idx * cell_w
            cell_style = style[kind]
            ax.add_patch(
                Rectangle(
                    (x, y),
                    cell_w,
                    row_h,
                    facecolor=cell_style["fill"],
                    edgecolor=cell_style["edge"],
                    linewidth=0.45,
                )
            )
            ax.text(
                x + cell_w / 2,
                y + row_h / 2,
                cell_text,
                ha="center",
                va="center",
                fontsize=6.0,
                color=cell_style["text"],
                linespacing=0.95,
            )

    legend_items = [
        ("raw/input", "raw"),
        ("internal identifier", "identifier"),
        ("created key", "created"),
        ("derived/cleaned", "derived"),
        ("final export", "exported"),
        ("registry data", "registry"),
        ("optional", "optional"),
        ("absent/removed", "absent"),
    ]
    legend_x = 0.72
    legend_y = 3.05
    for idx, (label, kind) in enumerate(legend_items):
        x = legend_x + idx * 1.42
        cell_style = style[kind]
        ax.add_patch(
            Rectangle(
                (x, legend_y),
                0.18,
                0.18,
                facecolor=cell_style["fill"],
                edgecolor=cell_style["edge"],
                linewidth=0.5,
            )
        )
        ax.text(
            x + 0.24, legend_y + 0.09, label, ha="left", va="center", fontsize=6.2, color="#374151"
        )

    ax.text(
        0.55,
        2.88,
        "B. Final ML-ready matched dataset",
        ha="left",
        va="bottom",
        fontsize=10.3,
        weight="bold",
        color="#111827",
    )

    ctg_rows = STAGES["stage7_ctg"].rows or 0
    babies = STAGES["stage7_ctg"].babies or 0
    mean_rows = ctg_rows / babies if babies else 0
    _draw_plain_box(
        ax,
        0.70,
        1.14,
        3.55,
        1.46,
        "Final CTG parquet",
        [
            f"{_format_int(ctg_rows)} timestamped rows",
            f"{_format_int(babies)} BabyIDs, mean {mean_rows:,.0f} rows/BabyID",
            "Columns: BabyID, Timestamp, FHR, toco,",
            "optional signal metadata",
        ],
        fill="#ecfeff",
        edge="#0f766e",
        title_size=8.5,
        text_size=6.7,
    )
    _draw_plain_box(
        ax,
        8.05,
        1.14,
        3.55,
        1.46,
        "Final registry CSV",
        [
            f"{_format_int(STAGES['stage7_registry'].rows)} rows",
            "Exactly one row per BabyID",
            "Columns: BabyID plus maternal, delivery,",
            "neonatal, and outcome variables",
        ],
        fill="#fef2f2",
        edge="#dc2626",
        title_size=8.5,
        text_size=6.7,
    )
    _draw_plain_box(
        ax,
        4.75,
        1.00,
        2.75,
        1.82,
        "Join key",
        [
            "BabyID",
            "Links CTG time series",
            "to registry variables",
            "Direct identifiers removed",
            "from final outputs",
        ],
        fill="#f5f3ff",
        edge="#7c3aed",
        title_size=8.5,
        text_size=6.7,
    )
    _draw_arrow(ax, (4.25, 1.87), (4.75, 1.87))
    _draw_arrow(ax, (8.05, 1.87), (7.50, 1.87))

    ax.text(
        0.55,
        0.28,
        "Interpretation: the model can use repeated CTG signal rows for each BabyID together with one matched row of registry covariates/outcomes.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#4b5563",
    )

    return _save_figure(fig, output_dir, "ctg_dataset_structure_schema_evolution", formats, dpi)


RASTER_FORMATS = frozenset({"png", "jpg", "jpeg", "tif", "tiff"})


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        if fmt.lower() in RASTER_FORMATS:
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
        else:
            fig.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(fig)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-oriented figures for the CTG preprocessing pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--figure",
        choices=["all", "flowchart", "retention", "schema"],
        default="all",
        help="Which figure to generate.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["svg", "pdf", "png"],
        help="Output formats. SVG/PDF are recommended for thesis and manuscript use.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI for PNG output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths: list[Path] = []

    if args.figure in {"all", "flowchart"}:
        saved_paths.extend(create_flowchart(args.output_dir, args.formats, args.dpi))
    if args.figure in {"all", "retention"}:
        saved_paths.extend(create_retention_summary(args.output_dir, args.formats, args.dpi))
    if args.figure in {"all", "schema"}:
        saved_paths.extend(create_schema_evolution(args.output_dir, args.formats, args.dpi))

    for path in saved_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
