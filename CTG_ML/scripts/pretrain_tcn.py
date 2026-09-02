from __future__ import annotations

import argparse
from pathlib import Path

from ctg_ml.multimodal_config import load_multimodal_config
from ctg_ml.pretrain import run_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Self-supervised masked-reconstruction pretraining of the TCN sequence encoder on "
            "the NPZ shards written by scripts/preprocess_pretrain.py."
        )
    )
    parser.add_argument("--config", default="configs/ctg3_multimodal.toml")
    parser.add_argument(
        "--windows-dir",
        default=None,
        help="Directory with windows_*.npz + windows_meta.json (default: artifacts/<out_subdir>)",
    )
    parser.add_argument(
        "--out-dir", default=None, help="Where encoder.pt is written (default: same as windows)"
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = load_multimodal_config(args.config)
    windows_dir = (
        Path(args.windows_dir)
        if args.windows_dir
        else cfg.paths.artifacts_dir / cfg.pretrain.out_subdir
    )
    out_dir = Path(args.out_dir) if args.out_dir else windows_dir
    if not (windows_dir / "windows_meta.json").exists():
        raise FileNotFoundError(
            f"Missing {windows_dir / 'windows_meta.json'}. Run "
            f"`uv run python scripts/preprocess_pretrain.py --config {args.config}` first."
        )

    print(f"Windows dir: {windows_dir}")
    print(f"Output dir:  {out_dir}")
    print(
        f"Pretraining: epochs={cfg.pretrain.epochs} batch_size={cfg.pretrain.batch_size} "
        f"lr={cfg.pretrain.lr} mask_ratio={cfg.pretrain.mask_ratio} "
        f"mask_span={cfg.pretrain.mask_span_seconds}s tcn_channels={cfg.model.tcn_channels}"
    )
    metrics = run_pretraining(
        windows_dir=windows_dir,
        out_dir=out_dir,
        pretrain_cfg=cfg.pretrain,
        model_cfg=cfg.model,
        device=args.device,
        show_progress=not args.no_progress,
    )
    print(
        f"\nBest val_loss={metrics['best_val_loss']:.5f} at epoch {metrics['best_epoch']} "
        f"-> {metrics['encoder_path']}"
    )
    print(
        "Next: uv run python scripts/train_multimodal_tcn.py "
        f"--config {args.config} --init-encoder {metrics['encoder_path']}"
    )


if __name__ == "__main__":
    main()
