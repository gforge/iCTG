from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ctg_ml.ctg2_config import load_ctg2_config


APGAR_OUTCOMES = ["apgar1_below7", "apgar5_below7", "apgar10_below7"]
BINARY_OUTCOMES = [
    "ph_navel_below7",
    "shoulder_dystocia",
    "treatment_for_hypoglycemia",
    "neonatal_sepsis_or_pneumonia",
    "respiratorbehandling",
]
SUMMARY_OUTCOMES = ["apgar_mean_below7", *APGAR_OUTCOMES, *BINARY_OUTCOMES]

OUTCOME_LABELS = {
    "apgar_mean_below7": "Apgar mean (<7 average)",
    "apgar1_below7": "Apgar 1 min < 7",
    "apgar5_below7": "Apgar 5 min < 7",
    "apgar10_below7": "Apgar 10 min < 7",
    "ph_navel_below7": "Cord pH < 7",
    "shoulder_dystocia": "Shoulder dystocia",
    "treatment_for_hypoglycemia": "Treatment for hypoglycemia",
    "neonatal_sepsis_or_pneumonia": "Neonatal sepsis/pneumonia",
    "respiratorbehandling": "Respirator treatment",
}

ABLATION_LABELS = {
    "baseline": "Baseline",
    "CTG": "CTG removed",
    "all_registry": "All registry removed",
    "maternal_background": "Maternal background removed",
    "smoking": "Smoking-related variables removed",
    "maternal_conditions": "Maternal conditions removed",
    "labour_context": "Labour context removed",
}


@dataclass(frozen=True)
class AblationSpec:
    name: str
    kind: str  # baseline | sequence | tabular_all | tabular_group
    columns: list[str]


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _default_group_specs(cfg) -> list[AblationSpec]:
    grouped = [
        AblationSpec("baseline", "baseline", []),
        AblationSpec("CTG", "sequence", []),
        AblationSpec("all_registry", "tabular_all", []),
        AblationSpec(
            "maternal_background",
            "tabular_group",
            [
                "maternal_age",
                "para_mhv1",
                "langd_inskrivning_cm",
                "bmi_inskrivning",
                "alkohol_audit_poang",
                "fodelseland",
                "utbildningsniva",
                "is_girl",
            ],
        ),
        AblationSpec(
            "smoking",
            "tabular_group",
            [
                "is_smoker",
                "tobak_3_manader_fore_graviditet",
                "tobak_inskrivning",
                "tobak_vecka_30_32",
            ],
        ),
        AblationSpec(
            "maternal_conditions",
            "tabular_group",
            [
                "diabetes_mellitus",
                "gestational_hypertension_without_significant_proteinuria",
                "preeclampsia",
                "gestational_or_pregestational_diabetes",
            ],
        ),
        AblationSpec(
            "labour_context",
            "tabular_group",
            [
                "etablerade_varkar_seconds",
                "forlossningsstart",
                "use_of_oxytocin",
                "heavy_vaginal_bleeding_before_or_during_delivery",
                "labor_dystocia",
            ],
        ),
    ]
    single = [AblationSpec("baseline", "baseline", []), AblationSpec("CTG", "sequence", []), AblationSpec("all_registry", "tabular_all", [])]
    all_raw = cfg.registry.input_numeric + cfg.registry.input_boolean + cfg.registry.input_categorical
    for col in all_raw:
        single.append(AblationSpec(col, "tabular_group", [col]))
    return grouped, single


def _dedupe_specs(specs: list[AblationSpec]) -> list[AblationSpec]:
    seen: set[str] = set()
    out: list[AblationSpec] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out.append(spec)
    return out


def _filter_specs(specs: list[AblationSpec], requested_names: list[str] | None) -> list[AblationSpec]:
    if not requested_names:
        return specs
    requested = {name.strip() for name in requested_names if name.strip()}
    requested.add("baseline")
    filtered = [spec for spec in specs if spec.name in requested]
    missing = sorted(requested - {spec.name for spec in filtered})
    if missing:
        raise ValueError(f"Unknown ablation names requested: {missing}")
    return filtered


def _extract_test_metric(payload: dict, metric_name: str) -> dict[str, float]:
    test = payload["test_metrics"]
    result: dict[str, float] = {}
    apgar_vals = []
    for name in APGAR_OUTCOMES:
        val = float(test["derived_binary"].get(name, {}).get(metric_name, float("nan")))
        result[name] = val
        if math.isfinite(val):
            apgar_vals.append(val)
    result["apgar_mean_below7"] = float(sum(apgar_vals) / len(apgar_vals)) if apgar_vals else float("nan")
    for name in BINARY_OUTCOMES:
        result[name] = float(test["binary"].get(name, {}).get(metric_name, float("nan")))
    return result


def _build_command(train_script: Path, config: str, device: str, seed: int, spec: AblationSpec, metrics_out: Path, show_inner_progress: bool) -> list[str]:
    cmd = [sys.executable, str(train_script), "--config", config, "--device", device, "--seed-override", str(seed), "--metrics-out", str(metrics_out), "--run-name", f"ablation_{_sanitize(spec.name)}_seed{seed}"]
    if not show_inner_progress:
        cmd.append("--no-progress")
    if spec.kind == "sequence":
        cmd.append("--ablate-sequence")
    elif spec.kind == "tabular_all":
        cmd.append("--ablate-tabular")
    elif spec.kind == "tabular_group":
        cmd.extend(["--ablate-tabular-columns", ",".join(spec.columns)])
    return cmd


def _fmt_value(mean: float, sd: float, decimals: int = 3) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(sd):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def _build_readable_markdown(config_path: str, seeds: list[int], mode: str, summary_df: pd.DataFrame) -> str:
    non_baseline = summary_df[summary_df["ablation"] != "baseline"].copy()
    lines = [
        "# CTG2 Ablation Study",
        "",
        f"- Config: `{config_path}`",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Mode: {mode}",
        "- Interpretation: positive drop means the removed input group was helping the model.",
        "- Main comparison metric remains ROC-AUC. PR-AUC is included as a second view for rare outcomes.",
        "",
        "## Compact ROC-AUC Table",
        "",
        "| Ablation | Apgar mean | Apgar1 | Apgar5 | Apgar10 | pH<7 | Shoulder dystocia | Hypoglycemia | Sepsis/pneumonia | Respirator |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in non_baseline.iterrows():
        lines.append(
            "| "
            + ABLATION_LABELS.get(row["ablation"], str(row["ablation"]))
            + " | "
            + " | ".join(
                _fmt_value(row[f"{outcome}_mean_drop_roc_auc"], row[f"{outcome}_sd_drop_roc_auc"])
                for outcome in SUMMARY_OUTCOMES
            )
            + " |"
        )

    lines.extend(["", "## Outcome Ranking By ROC-AUC Drop", ""])
    for outcome in SUMMARY_OUTCOMES:
        lines.append(f"### {OUTCOME_LABELS.get(outcome, outcome)}")
        lines.append("")
        lines.append("| Rank | Ablation | Mean ROC-AUC drop | Mean PR-AUC drop |")
        lines.append("|---:|---|---:|---:|")
        ranked = non_baseline.sort_values(by=f"{outcome}_mean_drop_roc_auc", ascending=False)
        for i, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"| {i} | {ABLATION_LABELS.get(row['ablation'], str(row['ablation']))} | "
                f"{_fmt_value(row[f'{outcome}_mean_drop_roc_auc'], row[f'{outcome}_sd_drop_roc_auc'])} | "
                f"{_fmt_value(row[f'{outcome}_mean_drop_pr_auc'], row[f'{outcome}_sd_drop_pr_auc'])} |"
            )
        lines.append("")

    lines.extend(["## Notes", ""])
    lines.append("- Negative values mean the ablated run did slightly better than the baseline on that metric.")
    lines.append("- With one seed, all standard deviations are 0 and small negative values are often just training noise.")
    return "\n".join(lines)


def run_study(
    config_path: str,
    device: str,
    seeds: list[int],
    mode: str,
    output_dir: Path,
    force: bool,
    show_inner_progress: bool,
    only_ablations: list[str] | None,
) -> None:
    cfg = load_ctg2_config(config_path)
    grouped, single = _default_group_specs(cfg)
    if mode == "grouped":
        specs = grouped
    elif mode == "single":
        specs = single
    else:
        specs = _dedupe_specs(grouped + single)
    specs = _filter_specs(specs, only_ablations)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).with_name("train_ctg2_multimodal.py")

    tasks: list[tuple[int, AblationSpec]] = [(seed, spec) for seed in seeds for spec in specs]
    total_runs = len(tasks)
    completed = 0
    durations: list[float] = []
    raw_rows: list[dict] = []

    definitions_path = output_dir / "group_definitions.json"
    definitions_path.write_text(json.dumps({spec.name: {"kind": spec.kind, "columns": spec.columns} for spec in specs}, indent=2))

    for seed, spec in tasks:
        metrics_out = output_dir / "run_metrics" / f"{_sanitize(spec.name)}__seed{seed}.json"
        metrics_out.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        prefix = f"[{completed + 1}/{total_runs}] seed={seed} ablation={spec.name}"
        if metrics_out.exists() and not force:
            print(f"{prefix} -> reusing existing metrics {metrics_out.name}")
        else:
            eta = "unknown"
            if durations:
                avg = sum(durations) / len(durations)
                eta = _format_duration(avg * (total_runs - completed))
            print(f"{prefix} -> starting (ETA {eta})")
            cmd = _build_command(train_script, config_path, device, seed, spec, metrics_out, show_inner_progress)
            subprocess.run(cmd, check=True)
        payload = json.loads(metrics_out.read_text())
        roc_aucs = _extract_test_metric(payload, "roc_auc")
        pr_aucs = _extract_test_metric(payload, "pr_auc")
        elapsed = time.time() - start
        durations.append(elapsed)
        completed += 1
        row = {
            "seed": seed,
            "ablation": spec.name,
            "kind": spec.kind,
            "columns": ",".join(spec.columns),
            "metrics_json": str(metrics_out),
            "elapsed_seconds": round(elapsed, 2),
        }
        row.update({f"{k}_roc_auc": v for k, v in roc_aucs.items()})
        row.update({f"{k}_pr_auc": v for k, v in pr_aucs.items()})
        raw_rows.append(row)
        avg = sum(durations) / len(durations)
        remaining = _format_duration(avg * (total_runs - completed)) if completed < total_runs else "0s"
        print(f"{prefix} -> done in {_format_duration(elapsed)} (remaining ~{remaining})")

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = output_dir / "ablation_raw_results.csv"
    raw_df.to_csv(raw_csv, index=False)

    baseline_df = raw_df[raw_df["ablation"] == "baseline"].set_index("seed")
    delta_rows = []
    for _, row in raw_df.iterrows():
        seed = int(row["seed"])
        baseline = baseline_df.loc[seed]
        out = {
            "seed": seed,
            "ablation": row["ablation"],
            "kind": row["kind"],
            "columns": row["columns"],
        }
        for outcome in SUMMARY_OUTCOMES:
            for metric in ["roc_auc", "pr_auc"]:
                base_val = float(baseline[f"{outcome}_{metric}"])
                cur_val = float(row[f"{outcome}_{metric}"])
                out[f"{outcome}_{metric}_baseline"] = base_val
                out[f"{outcome}_{metric}_ablated"] = cur_val
                out[f"{outcome}_{metric}_drop"] = (
                    base_val - cur_val if math.isfinite(base_val) and math.isfinite(cur_val) else float("nan")
                )
        delta_rows.append(out)
    delta_df = pd.DataFrame(delta_rows)
    delta_csv = output_dir / "ablation_seed_deltas.csv"
    delta_df.to_csv(delta_csv, index=False)

    summary_records = []
    for ablation, part in delta_df.groupby("ablation", sort=False):
        record = {"ablation": ablation, "kind": part["kind"].iloc[0], "columns": part["columns"].iloc[0]}
        for outcome in SUMMARY_OUTCOMES:
            for metric in ["roc_auc", "pr_auc"]:
                vals = pd.to_numeric(part[f"{outcome}_{metric}_drop"], errors="coerce").dropna()
                record[f"{outcome}_mean_drop_{metric}"] = float(vals.mean()) if not vals.empty else float("nan")
                record[f"{outcome}_sd_drop_{metric}"] = (
                    float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan"))
                )
        summary_records.append(record)
    summary_df = pd.DataFrame(summary_records)
    summary_csv = output_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    md_lines = [
        "# CTG2 Ablation Study Summary",
        "",
        f"- Config: `{config_path}`",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Mode: {mode}",
        "",
        "## Mean drop in test ROC-AUC versus full baseline",
        "",
        "| Ablation | apgar_mean | apgar1 | apgar5 | apgar10 | ph_below7 | shoulder_dystocia | hypoglycemia | neonatal_sepsis_or_pneumonia | respiratorbehandling |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        def fmt(outcome: str, metric: str) -> str:
            mean = row[f"{outcome}_mean_drop_{metric}"]
            sd = row[f"{outcome}_sd_drop_{metric}"]
            if pd.isna(mean):
                return "NA"
            if pd.isna(sd):
                return f"{mean:.4f}"
            return f"{mean:.4f} ± {sd:.4f}"

        md_lines.append(
            f"| {row['ablation']} | {fmt('apgar_mean_below7', 'roc_auc')} | {fmt('apgar1_below7', 'roc_auc')} | {fmt('apgar5_below7', 'roc_auc')} | {fmt('apgar10_below7', 'roc_auc')} | {fmt('ph_navel_below7', 'roc_auc')} | {fmt('shoulder_dystocia', 'roc_auc')} | {fmt('treatment_for_hypoglycemia', 'roc_auc')} | {fmt('neonatal_sepsis_or_pneumonia', 'roc_auc')} | {fmt('respiratorbehandling', 'roc_auc')} |"
        )
    md_lines.extend(
        [
            "",
            "## Mean drop in test PR-AUC versus full baseline",
            "",
            "| Ablation | apgar_mean | apgar1 | apgar5 | apgar10 | ph_below7 | shoulder_dystocia | hypoglycemia | neonatal_sepsis_or_pneumonia | respiratorbehandling |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['ablation']} | {fmt('apgar_mean_below7', 'pr_auc')} | {fmt('apgar1_below7', 'pr_auc')} | {fmt('apgar5_below7', 'pr_auc')} | {fmt('apgar10_below7', 'pr_auc')} | {fmt('ph_navel_below7', 'pr_auc')} | {fmt('shoulder_dystocia', 'pr_auc')} | {fmt('treatment_for_hypoglycemia', 'pr_auc')} | {fmt('neonatal_sepsis_or_pneumonia', 'pr_auc')} | {fmt('respiratorbehandling', 'pr_auc')} |"
        )
    md_path = output_dir / "ablation_summary.md"
    md_path.write_text("\n".join(md_lines))
    readable_md_path = output_dir / "ablation_summary_readable.md"
    readable_md_path.write_text(_build_readable_markdown(config_path, seeds, mode, summary_df))

    print(f"Wrote raw results: {raw_csv}")
    print(f"Wrote per-seed deltas: {delta_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote markdown summary: {md_path}")
    print(f"Wrote readable markdown summary: {readable_md_path}")
    print(f"Wrote group definitions: {definitions_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated grouped/single ablation study for CTG2 multimodal training.")
    parser.add_argument("--config", default="configs/ctg2_multimodal.toml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to the training seed from config.")
    parser.add_argument("--mode", choices=["grouped", "single", "both"], default="grouped")
    parser.add_argument(
        "--only-ablations",
        default=None,
        help="Comma-separated ablation names to run. baseline is added automatically.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for study outputs. Defaults to <artifacts_dir>/ablation_study/<mode>")
    parser.add_argument("--force", action="store_true", help="Rerun even if per-run metrics JSON already exists.")
    parser.add_argument("--show-inner-progress", action="store_true", help="Show the full batch progress bars from each underlying training run.")
    args = parser.parse_args()

    cfg = load_ctg2_config(args.config)
    seeds = [int(x.strip()) for x in args.seeds.split(",")] if args.seeds else [int(cfg.train.seed)]
    only_ablations = [x.strip() for x in args.only_ablations.split(",")] if args.only_ablations else None
    output_dir = Path(args.output_dir) if args.output_dir else (cfg.paths.artifacts_dir / "ablation_study" / args.mode)
    run_study(args.config, args.device, seeds, args.mode, output_dir, args.force, args.show_inner_progress, only_ablations)


if __name__ == "__main__":
    main()
