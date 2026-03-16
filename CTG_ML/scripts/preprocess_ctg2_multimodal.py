from __future__ import annotations

import argparse
from pathlib import Path

from ctg_ml.ctg2_config import load_ctg2_config
from ctg_ml.ctg2_preprocess import build_ctg2_multimodal_npz_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess CTG2 parquet + registry CSV into multimodal NPZ files for TCN training."
    )
    parser.add_argument("--config", default="configs/ctg2_multimodal.toml")
    parser.add_argument("--splits", default=None, help="Path to train/val/test split CSV")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_ctg2_config(args.config)
    splits_path = Path(args.splits) if args.splits else cfg.paths.artifacts_dir / "splits.csv"
    output_dir = Path(args.output_dir) if args.output_dir else cfg.sequence.output_dir

    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing splits file: {splits_path}. Run `uv run python scripts/make_splits_ctg2.py --config {args.config}` first."
        )

    print(
        f"Building CTG2 multimodal NPZ: window={cfg.sequence.window_minutes} min, "
        f"steps={cfg.sequence.window_minutes * 60 * cfg.sequence.sample_rate_hz}, "
        f"pad_short={cfg.sequence.pad_short}, quality_channels={cfg.sequence.include_signal_quality_channels}, "
        f"padding_mask={cfg.sequence.include_padding_mask}"
    )
    print(f"CTG parquet:   {cfg.paths.ctg_parquet}")
    print(f"Registry csv:  {cfg.paths.registry_csv}")
    print(f"Splits csv:    {splits_path}")
    print(f"Output dir:    {output_dir}")

    stats = build_ctg2_multimodal_npz_files(
        ctg_parquet=cfg.paths.ctg_parquet,
        registry_csv=cfg.paths.registry_csv,
        splits_csv=splits_path,
        output_dir=output_dir,
        seq_cfg=cfg.sequence,
        registry_cfg=cfg.registry,
    )

    print("\nPreprocessing summary")
    for s in stats:
        print(
            f"{s.split_name:>5}: kept={s.kept_babies}/{s.total_babies} dropped_short={s.dropped_short_babies} "
            f"rows[min/med/max]={s.min_rows}/{s.median_rows:.1f}/{s.max_rows} "
            f"X_seq=({s.kept_babies},{s.seq_channels},{s.n_steps}) "
            f"X_tab=({s.kept_babies},{s.tab_features}) -> {s.output_path}"
        )


if __name__ == "__main__":
    main()
