from __future__ import annotations

import argparse
from pathlib import Path

from ctg_ml.multimodal_config import load_multimodal_config
from ctg_ml.pretrain_preprocess import build_pretrain_windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cut unlabeled CTG windows (all sessions of all pregnancies) into NPZ shards for "
            "self-supervised TCN encoder pretraining. Val/test BabyIDs from splits.csv are "
            "always excluded."
        )
    )
    parser.add_argument("--config", default="configs/ctg3_multimodal.toml")
    parser.add_argument(
        "--pretrain-parquet",
        default=None,
        help="Override [pretrain].pretrain_parquet (file or directory of *.parquet buckets)",
    )
    parser.add_argument("--splits", default=None, help="Path to train/val/test split CSV")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument(
        "--allow-no-splits",
        action="store_true",
        help="Proceed without splits.csv (nothing excluded). Only for unlabeled-only setups.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = load_multimodal_config(args.config)
    pretrain_parquet = (
        Path(args.pretrain_parquet) if args.pretrain_parquet else cfg.pretrain.pretrain_parquet
    )
    splits_path = Path(args.splits) if args.splits else cfg.paths.artifacts_dir / "splits.csv"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.paths.artifacts_dir / cfg.pretrain.out_subdir
    )

    print(
        f"Building pretraining windows: window={cfg.pretrain.window_minutes} min, "
        f"stride={cfg.pretrain.stride_minutes} min, "
        f"min_signal_fraction={cfg.pretrain.min_signal_fraction}, "
        f"exclude_final_window={cfg.pretrain.exclude_final_window}"
    )
    print(f"Pretrain parquet: {pretrain_parquet}")
    print(f"Splits csv:       {splits_path} (val/test BabyIDs excluded)")
    print(f"Output dir:       {output_dir}")

    stats = build_pretrain_windows(
        pretrain_parquet=pretrain_parquet,
        splits_csv=splits_path,
        output_dir=output_dir,
        seq_cfg=cfg.sequence,
        pretrain_cfg=cfg.pretrain,
        allow_no_splits=args.allow_no_splits,
        show_progress=not args.no_progress,
    )

    print("\nPretraining window summary")
    print(
        f"windows={stats.n_windows} babies={stats.n_babies} sessions={stats.n_sessions} "
        f"excluded_val_test_babies={stats.n_excluded_baby_ids}"
    )
    print(
        f"dropped: low_signal={stats.n_windows_dropped_low_signal} "
        f"final_window_overlap={stats.n_windows_dropped_final_overlap} "
        f"sessions_too_long={stats.n_sessions_skipped_too_long}"
    )
    print(f"channels={stats.channel_names} n_steps={stats.n_steps}")
    print(
        f"normalization: FHR mean/std={stats.means[0]:.3f}/{stats.stds[0]:.3f} "
        f"toco mean/std={stats.means[1]:.3f}/{stats.stds[1]:.3f}"
    )
    print(f"shards={len(stats.shard_paths)} -> {stats.meta_path}")


if __name__ == "__main__":
    main()
