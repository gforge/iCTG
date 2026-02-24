from __future__ import annotations

import argparse

from ctg_ml.config import load_config
from ctg_ml.data import load_registry_labels, print_label_summary
from ctg_ml.splits import SplitFractions, create_stratified_splits, print_split_summary, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create BabyID-level train/val/test splits.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--out", default=None, help="Output CSV path (defaults to artifacts/splits.csv)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = args.out or str(cfg.paths.artifacts_dir / "splits.csv")

    labels = load_registry_labels(cfg.paths.registry_csv, cfg.target.at_risk_max_apgar)
    print_label_summary(labels)

    splits = create_stratified_splits(
        labels=labels,
        fractions=SplitFractions(
            train_fraction=cfg.split.train_fraction,
            val_fraction=cfg.split.val_fraction,
            test_fraction=cfg.split.test_fraction,
        ),
        random_seed=cfg.split.random_seed,
    )
    save_splits(splits, out_path)
    print(f"Saved splits to {out_path}")
    print_split_summary(splits)


if __name__ == "__main__":
    main()
