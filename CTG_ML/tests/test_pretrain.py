from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pretrain_fixtures import (
    make_pretrain_cfg,
    make_seq_cfg,
    write_parquet,
    write_splits,
)

from ctg_ml.models import MaskedReconstructionTCN, MultimodalMultitaskTCN, TCNEncoder
from ctg_ml.multimodal_config import MultimodalModelConfig
from ctg_ml.pretrain import (
    ENCODER_FILENAME,
    METRICS_FILENAME,
    apply_mask,
    load_pretrained_encoder,
    make_span_mask,
    masked_reconstruction_loss,
    prepare_batch,
    run_pretraining,
    set_encoder_frozen,
)
from ctg_ml.pretrain_preprocess import build_pretrain_windows

MODEL_CFG = MultimodalModelConfig(
    tcn_channels=[4, 4], kernel_size=3, dropout=0.0, tabular_hidden_dim=6, fusion_hidden_dim=8
)


def test_span_mask_covers_requested_ratio_with_contiguous_spans() -> None:
    gen = torch.Generator().manual_seed(0)
    mask = make_span_mask(64, 3600, mask_ratio=0.3, span=30, generator=gen)
    assert mask.shape == (64, 3600) and mask.dtype == torch.bool
    assert abs(float(mask.float().mean()) - 0.3) < 0.05
    # Every masked run is at least `span` long (spans only ever merge, never shrink).
    row = mask[0].numpy().astype(np.int8)
    edges = np.diff(np.concatenate([[0], row, [0]]))
    run_lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    assert run_lengths.min() >= 30
    assert not make_span_mask(2, 10, 0.0, 3).any()


def test_apply_mask_only_touches_fhr_and_toco_channels() -> None:
    x = torch.rand(3, 5, 40) + 0.5  # strictly positive so zeros are unambiguous
    mask = make_span_mask(3, 40, mask_ratio=0.4, span=5, generator=torch.Generator().manual_seed(1))
    masked = apply_mask(x, mask)
    m = mask.unsqueeze(1)
    assert torch.all(masked[:, :2, :][m.expand(-1, 2, -1)] == 0.0)
    assert torch.equal(masked[:, :2, :][~m.expand(-1, 2, -1)], x[:, :2, :][~m.expand(-1, 2, -1)])
    assert torch.equal(masked[:, 2:, :], x[:, 2:, :])


def test_loss_ignores_missing_positions() -> None:
    raw = torch.full((2, 3, 10), 100.0)
    raw[0, 0, :5] = float("nan")  # missing FHR
    means = torch.tensor([100.0, 100.0])
    stds = torch.tensor([1.0, 1.0])
    x_in, target, valid = prepare_batch(raw, means, stds)
    assert torch.isfinite(x_in).all() and torch.isfinite(target).all()
    assert not valid[0, 0, :5].any() and valid[0, 0, 5:].all() and valid[:, 1, :].all()
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, :5] = True  # only masked positions are missing ones -> no contribution
    pred = torch.ones_like(target)  # error of 1 everywhere
    loss = masked_reconstruction_loss(pred, target, mask, valid)
    assert float(loss) == pytest.approx(1.0)  # from the toco channel at the 5 masked steps
    mask[:] = False
    assert float(masked_reconstruction_loss(pred, target, mask, valid)) == 0.0


def test_encode_sequence_keeps_time_axis_and_forward_is_unchanged() -> None:
    enc = TCNEncoder(in_channels=5, channels=[4, 4], kernel_size=3, dropout=0.0).eval()
    x = torch.randn(2, 5, 32)
    feats = enc.encode_sequence(x)
    assert feats.shape == (2, 4, 32)
    assert torch.allclose(enc(x), feats.mean(dim=-1))
    recon = MaskedReconstructionTCN(5, [4, 4], 3, 0.0, decoder_hidden_dim=3, decoder_kernel_size=5)
    assert recon(x).shape == (2, 2, 32)


def _pretrain_tiny(tmp_path: Path, epochs: int = 1) -> Path:
    source = write_parquet(tmp_path, as_directory=False)
    windows_dir = tmp_path / "pretrain"
    build_pretrain_windows(
        source,
        write_splits(tmp_path),
        windows_dir,
        make_seq_cfg(),
        make_pretrain_cfg(),
        show_progress=False,
    )
    cfg = make_pretrain_cfg(
        epochs=epochs, batch_size=4, decoder_hidden_dim=3, decoder_kernel_size=3
    )
    metrics = run_pretraining(
        windows_dir, windows_dir, cfg, MODEL_CFG, device="cpu", show_progress=False
    )
    assert metrics["epochs_run"] == epochs
    return windows_dir


def test_pretraining_writes_encoder_and_metrics(tmp_path: Path) -> None:
    out_dir = _pretrain_tiny(tmp_path)
    ckpt = torch.load(out_dir / ENCODER_FILENAME, map_location="cpu", weights_only=False)
    assert set(ckpt) >= {"sequence_encoder_state_dict", "normalization", "config", "channel_names"}
    assert ckpt["channel_names"] == [
        "FHR",
        "toco",
        "Hr1_SignalQuality==Y",
        "Hr1_SignalQuality==R",
        "padding_mask",
    ]
    assert ckpt["sequence_encoder_state_dict"]["tcn.0.net.0.weight"].shape == (4, 5, 3)
    metrics = json.loads((out_dir / METRICS_FILENAME).read_text())
    assert np.isfinite(metrics["best_val_loss"]) and len(metrics["history"]) == 1
    assert metrics["n_train_babies"] + metrics["n_val_babies"] == 2


def _supervised_model(seq_channels: int) -> MultimodalMultitaskTCN:
    torch.manual_seed(123)
    return MultimodalMultitaskTCN(
        sequence_in_channels=seq_channels,
        tabular_in_features=3,
        tcn_channels=MODEL_CFG.tcn_channels,
        kernel_size=MODEL_CFG.kernel_size,
        dropout=0.0,
        tabular_hidden_dim=6,
        fusion_hidden_dim=8,
        num_apgar_outputs=1,
        categorical_output_dims=[2],
        num_regression_outputs=1,
        num_binary_outputs=2,
    )


def test_loading_encoder_changes_weights_and_forward_works(tmp_path: Path) -> None:
    encoder_path = _pretrain_tiny(tmp_path) / ENCODER_FILENAME
    model = _supervised_model(seq_channels=5)
    before = {k: v.clone() for k, v in model.sequence_encoder.state_dict().items()}
    head_before = model.apgar_head.weight.clone()

    load_pretrained_encoder(
        model,
        encoder_path,
        expected_channel_names=[
            "FHR",
            "toco",
            "Hr1_SignalQuality==Y",
            "Hr1_SignalQuality==R",
            "padding_mask",
        ],
    )

    after = model.sequence_encoder.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert torch.equal(head_before, model.apgar_head.weight)  # only the encoder is touched
    model.eval()
    with torch.no_grad():
        apgar, cats, reg, binary = model(torch.randn(2, 5, 60), torch.randn(2, 3))
    assert apgar.shape == (2, 1, 11) and reg.shape == (2, 1) and binary.shape == (2, 2)

    set_encoder_frozen(model, True)
    assert not any(p.requires_grad for p in model.sequence_encoder.parameters())
    assert all(p.requires_grad for p in model.tabular_encoder.parameters())
    set_encoder_frozen(model, False)
    assert all(p.requires_grad for p in model.sequence_encoder.parameters())


def test_loading_encoder_with_wrong_channel_count_raises(tmp_path: Path) -> None:
    encoder_path = _pretrain_tiny(tmp_path) / ENCODER_FILENAME
    with pytest.raises(ValueError, match="input channels"):
        load_pretrained_encoder(_supervised_model(seq_channels=3), encoder_path)
    with pytest.raises(ValueError, match="channel names"):
        load_pretrained_encoder(
            _supervised_model(seq_channels=5), encoder_path, expected_channel_names=["a"] * 5
        )
