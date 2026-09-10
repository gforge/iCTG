#!/usr/bin/env python3
"""Clinical-style evaluation of a run's test predictions: calibration, sensitivity and
positive predictive value at fixed flag rates, and per-year stability.

    uv run python scripts/evaluate_clinical.py --predictions artifacts_ctg3/metrics_random_init_predictions.npz \\
        --registry data/CTG3/registry.csv --out artifacts_ctg3/clinical_random_init.md

The predictions file is written by train_multimodal_tcn.py next to --metrics-out. Only
aggregate numbers are printed or written (BabyIDs are used for the join, never listed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FLAG_RATES = (0.05, 0.10, 0.20)
MIN_POSITIVES = 10


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": b + 1,
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_rate": float(y[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def at_flag_rate(y: np.ndarray, p: np.ndarray, rate: float) -> dict[str, float]:
    n_flag = max(1, int(round(rate * len(p))))
    order = np.argsort(-p)
    flagged = np.zeros(len(p), dtype=bool)
    flagged[order[:n_flag]] = True
    tp = float((flagged & (y == 1)).sum())
    return {
        "flag_rate": rate,
        "threshold": float(p[order[n_flag - 1]]),
        "sensitivity": tp / max(float((y == 1).sum()), 1.0),
        "ppv": tp / n_flag,
        "flagged": n_flag,
    }


def evaluate_outcome(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    n_pos = int(y.sum())
    result: dict = {
        "outcome": name,
        "n": int(len(y)),
        "positives": n_pos,
        "prevalence": float(y.mean()),
    }
    if n_pos < MIN_POSITIVES or n_pos == len(y):
        result["skipped"] = f"fewer than {MIN_POSITIVES} positives"
        return result
    result["roc_auc"] = float(roc_auc_score(y, p))
    result["pr_auc"] = float(average_precision_score(y, p))
    result["brier"] = float(brier_score_loss(y, p))
    result["brier_baseline"] = float(y.mean() * (1 - y.mean()))
    result["mean_predicted"] = float(p.mean())
    result["calibration"] = calibration_table(y, p).to_dict("records")
    result["flag_rates"] = [at_flag_rate(y, p, r) for r in FLAG_RATES]
    return result


def per_year(name: str, y: np.ndarray, p: np.ndarray, years: np.ndarray) -> list[dict]:
    rows = []
    for yr in sorted({int(v) for v in years if np.isfinite(v)}):
        m = years == yr
        if m.sum() < 200 or y[m].sum() < MIN_POSITIVES or y[m].sum() == m.sum():
            continue
        rows.append(
            {
                "year": yr,
                "n": int(m.sum()),
                "positives": int(y[m].sum()),
                "roc_auc": float(roc_auc_score(y[m], p[m])),
                "pr_auc": float(average_precision_score(y[m], p[m])),
            }
        )
    return rows


def render(results: list[dict], years: dict[str, list[dict]], source: str) -> str:
    lines = [f"# Clinical evaluation: {source}", ""]
    lines += [
        "Test split. Sensitivity and PPV are reported at fixed flag rates (share of births",
        "the model would flag); Brier baseline is the score of predicting the prevalence.",
        "Years are the time-shifted birth years, so they are approximate (+/- 1 year).",
        "",
        "| Outcome | Positives | Prevalence | ROC-AUC | PR-AUC | Brier (baseline) | Sens@5% | PPV@5% | Sens@10% | PPV@10% | Sens@20% | PPV@20% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if "skipped" in r:
            lines.append(
                f"| {r['outcome']} | {r['positives']} | {100 * r['prevalence']:.2f} % | skipped: {r['skipped']} |||||||||"
            )
            continue
        fr = {f["flag_rate"]: f for f in r["flag_rates"]}
        lines.append(
            f"| {r['outcome']} | {r['positives']} | {100 * r['prevalence']:.2f} % | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
            f"{r['brier']:.4f} ({r['brier_baseline']:.4f}) | "
            + " | ".join(f"{fr[k]['sensitivity']:.2f} | {fr[k]['ppv']:.2f}" for k in FLAG_RATES)
            + " |"
        )
    lines += ["", "## Calibration (deciles of predicted risk)", ""]
    for r in results:
        if "skipped" in r:
            continue
        lines.append(
            f"### {r['outcome']} (mean predicted {r['mean_predicted']:.4f}, observed {r['prevalence']:.4f})"
        )
        lines.append("")
        lines.append("| Decile | n | Mean predicted | Observed |")
        lines.append("|---:|---:|---:|---:|")
        for c in r["calibration"]:
            lines.append(
                f"| {c['bin']} | {c['n']} | {c['mean_predicted']:.4f} | {c['observed_rate']:.4f} |"
            )
        lines.append("")
    if years:
        lines += ["## Per (shifted) birth year", ""]
        for name, rows in years.items():
            if not rows:
                continue
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Year | n | Positives | ROC-AUC | PR-AUC |")
            lines.append("|---:|---:|---:|---:|---:|")
            for row in rows:
                lines.append(
                    f"| {row['year']} | {row['n']} | {row['positives']} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--registry", default=None, help="registry.csv with birth_day for the per-year table"
    )
    parser.add_argument(
        "--out", default=None, help="markdown output (default: next to the predictions)"
    )
    parser.add_argument(
        "--year-outcomes", default="severe_neonatal_outcome,apgar5_below7,neonatal_care_admission"
    )
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=False)
    baby_ids = data["baby_ids"].astype(str)
    outcomes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for names_key, prob_key, true_key, mask_key in (
        ("apgar_names", "apgar_below7_prob", "apgar_below7_true", "apgar_mask"),
        ("binary_names", "binary_prob", "binary_true", "binary_mask"),
    ):
        names = data[names_key].astype(str)
        for i, name in enumerate(names):
            valid = data[mask_key][:, i] > 0
            outcomes[name] = (
                data[true_key][valid, i].astype(int),
                data[prob_key][valid, i].astype(float),
            )

    results = [evaluate_outcome(name, y, p) for name, (y, p) in outcomes.items()]

    years_by_outcome: dict[str, list[dict]] = {}
    if args.registry:
        reg = pd.read_csv(args.registry, usecols=["BabyID", "birth_day"], dtype={"BabyID": str})
        year_map = dict(
            zip(reg["BabyID"], pd.to_datetime(reg["birth_day"], errors="coerce").dt.year)
        )
        years_all = np.array([year_map.get(b, np.nan) for b in baby_ids], dtype=float)
        for name in [x.strip() for x in args.year_outcomes.split(",") if x.strip()]:
            if name not in outcomes:
                continue
            names_key, mask_key = (
                ("apgar_names", "apgar_mask")
                if name.endswith("_below7")
                else ("binary_names", "binary_mask")
            )
            i = list(data[names_key].astype(str)).index(name)
            valid = data[mask_key][:, i] > 0
            y, p = outcomes[name]
            years_by_outcome[name] = per_year(name, y, p, years_all[valid])

    out = Path(args.out) if args.out else Path(args.predictions).with_suffix(".clinical.md")
    out.write_text(render(results, years_by_outcome, Path(args.predictions).name))
    out.with_suffix(".json").write_text(
        json.dumps({"results": results, "per_year": years_by_outcome}, indent=2)
    )
    print(f"Wrote {out} and {out.with_suffix('.json')}")
    for r in results:
        if "skipped" in r:
            continue
        s10 = next(f for f in r["flag_rates"] if f["flag_rate"] == 0.10)
        print(
            f"{r['outcome']:40s} PR-AUC {r['pr_auc']:.3f}  sens@10% {s10['sensitivity']:.2f}  ppv@10% {s10['ppv']:.2f}  Brier {r['brier']:.4f} vs {r['brier_baseline']:.4f}"
        )


if __name__ == "__main__":
    main()
