from __future__ import annotations

import argparse
from pathlib import Path

from ctg_ml.config import load_config
from ctg_ml.sequence_preprocess import SequenceBuildConfig, build_tcn_npz_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess CTG parquet into fixed-length NPZ sequences for TCN training."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--splits", default=None, help="Path to artifacts/splits.csv")
    parser.add_argument("--output-dir", default=None, help="Override sequence output dir")
    parser.add_argument("--window-minutes", type=int, default=None, help="Override sequence window length")
    parser.add_argument("--pad-short", action="store_true", help="Left-pad short sequences instead of dropping")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_path = Path(args.splits) if args.splits else cfg.paths.artifacts_dir / "splits.csv"
    output_dir = Path(args.output_dir) if args.output_dir else cfg.sequence.output_dir
    window_minutes = args.window_minutes if args.window_minutes is not None else cfg.sequence.window_minutes
    pad_short = bool(args.pad_short or cfg.sequence.pad_short)

    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing splits file: {splits_path}. Run `uv run python scripts/make_splits.py` first."
        )

    seq_cfg = SequenceBuildConfig(
        window_minutes=window_minutes,
        sample_rate_hz=cfg.sequence.sample_rate_hz,
        pad_short=pad_short,
        treat_fhr_zero_as_missing=cfg.sequence.treat_fhr_zero_as_missing,
        include_fhr_missing_mask=cfg.sequence.include_fhr_missing_mask,
        chunk_vectors_per_batch=cfg.sequence.chunk_vectors_per_batch,
    )

    print(
        f"Building TCN sequences: window={seq_cfg.window_minutes} min, "
        f"sample_rate={seq_cfg.sample_rate_hz} Hz, steps={seq_cfg.n_steps}, "
        f"pad_short={seq_cfg.pad_short}, fhr_mask={seq_cfg.include_fhr_missing_mask}"
    )
    print(f"Input parquet: {cfg.paths.ctg_parquet}")
    print(f"Splits csv:    {splits_path}")
    print(f"Output dir:    {output_dir}")

    stats = build_tcn_npz_files(
        ctg_parquet=cfg.paths.ctg_parquet,
        splits_csv=splits_path,
        output_dir=output_dir,
        cfg=seq_cfg,
    )

    print("\nPreprocessing summary")
    for s in stats:
        print(
            f"{s.split_name:>5}: kept={s.kept_babies}/{s.total_babies} "
            f"dropped_short={s.dropped_short_babies} rows[min/med/max]={s.min_rows}/{s.median_rows:.1f}/{s.max_rows} "
            f"X=({s.kept_babies},{s.channels},{s.n_steps}) -> {s.output_path}"
        )


if __name__ == "__main__":
    main()
