from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ctg_ml.multimodal_config import MultimodalSequenceConfig
from ctg_ml.multimodal_preprocess import _finalize_sequence, sequence_channel_names


def _make_cfg(pad_short: bool = True) -> MultimodalSequenceConfig:
    # window_minutes * 60 * sample_rate_hz -> 60 steps, the smallest window the config allows.
    return MultimodalSequenceConfig(
        window_minutes=1,
        sample_rate_hz=1,
        pad_short=pad_short,
        treat_fhr_zero_as_missing=True,
        treat_toco_zero_as_missing=False,
        include_padding_mask=True,
        include_signal_quality_channels=True,
        quality_levels=["Good", "Bad"],
        output_dir=Path("unused"),
        chunk_vectors_per_batch=1,
    )


def _make_group(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fhr": 120.0 + np.arange(n_rows, dtype=np.float64),
            "toco": 10.0 + np.arange(n_rows, dtype=np.float64),
            "hr1_signal_quality": ["Good" if i % 2 == 0 else "Bad" for i in range(n_rows)],
        }
    )


def test_short_recording_is_left_padded_with_mask() -> None:
    cfg = _make_cfg()
    channels = sequence_channel_names(cfg)
    n_steps = 60
    group = _make_group(20)
    pad = n_steps - len(group)

    seq, raw_len = _finalize_sequence(group, cfg)

    assert seq is not None
    assert raw_len == 20
    assert seq.shape == (len(channels), n_steps)
    assert seq.dtype == np.float32

    mask = seq[channels.index("padding_mask")]
    assert np.all(mask[:pad] == 1.0)
    assert np.all(mask[pad:] == 0.0)

    fhr = seq[channels.index("FHR")]
    toco = seq[channels.index("toco")]
    assert np.all(np.isnan(fhr[:pad]))
    assert np.all(np.isnan(toco[:pad]))
    np.testing.assert_array_equal(fhr[pad:], group["fhr"].to_numpy(dtype=np.float32))
    np.testing.assert_array_equal(toco[pad:], group["toco"].to_numpy(dtype=np.float32))

    good = seq[channels.index("Hr1_SignalQuality==Good")]
    bad = seq[channels.index("Hr1_SignalQuality==Bad")]
    assert np.all(good[:pad] == 0.0) and np.all(bad[:pad] == 0.0)
    np.testing.assert_array_equal(good[pad:], (group["hr1_signal_quality"] == "Good").to_numpy())
    np.testing.assert_array_equal(bad[pad:], (group["hr1_signal_quality"] == "Bad").to_numpy())


def test_long_recording_keeps_only_the_tail_and_masks_zero_fhr() -> None:
    cfg = _make_cfg()
    channels = sequence_channel_names(cfg)
    group = _make_group(75)
    group.loc[group.index[-1], "fhr"] = 0.0  # zero FHR is treated as missing

    seq, raw_len = _finalize_sequence(group, cfg)

    assert seq is not None
    assert raw_len == 75
    assert seq.shape == (len(channels), 60)
    assert np.all(seq[channels.index("padding_mask")] == 0.0)

    expected_fhr = group["fhr"].to_numpy(dtype=np.float32)[-60:]
    fhr = seq[channels.index("FHR")]
    np.testing.assert_array_equal(fhr[:-1], expected_fhr[:-1])
    assert np.isnan(fhr[-1])
    np.testing.assert_array_equal(
        seq[channels.index("toco")], group["toco"].to_numpy(dtype=np.float32)[-60:]
    )


def test_short_recording_is_dropped_when_padding_disabled() -> None:
    seq, raw_len = _finalize_sequence(_make_group(20), _make_cfg(pad_short=False))

    assert seq is None
    assert raw_len == 20
