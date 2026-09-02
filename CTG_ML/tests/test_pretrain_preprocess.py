from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pretrain_fixtures import (
    T0,
    make_pretrain_cfg,
    make_seq_cfg,
    synthetic_frames,
    write_parquet,
    write_splits,
)

from ctg_ml.multimodal_preprocess import sequence_channel_names
from ctg_ml.pretrain_preprocess import (
    build_pretrain_windows,
    cut_session_windows,
    load_excluded_baby_ids,
    pretrain_sequence_config,
)


def test_cut_session_windows_reindexes_gaps_and_applies_signal_threshold() -> None:
    seq_cfg = pretrain_sequence_config(make_seq_cfg(), make_pretrain_cfg())
    session = synthetic_frames()["C"].rename(
        columns={
            "Timestamp": "ts",
            "FHR": "fhr",
            "toco": "toco",
            "Hr1_SignalQuality": "hr1_signal_quality",
        }
    )
    result = cut_session_windows(
        session, seq_cfg, stride_seconds=30, min_signal_fraction=0.5, exclude_final_window=True
    )

    # Candidate starts 0,30,..,120: valid FHR counts 50, 20, 0, 30, 60 -> keep 3 (>= 30).
    assert len(result.windows) == 3
    assert result.dropped_low_signal == 2
    assert result.window_starts_unix == [int(T0.timestamp()) + s for s in (0, 90, 120)]
    channels = sequence_channel_names(seq_cfg)
    first = result.windows[0]
    assert first.shape == (len(channels), 60)
    fhr = first[channels.index("FHR")]
    assert np.isfinite(fhr[:50]).all()
    assert np.isnan(fhr[50:]).all()  # missing seconds became FHR=0 -> NaN
    assert np.all(first[channels.index("padding_mask")] == 0.0)
    assert first[channels.index("Hr1_SignalQuality==Y")][:50].sum() > 0
    assert np.all(first[channels.index("Hr1_SignalQuality==Y")][50:] == 0.0)


def test_cut_session_windows_drops_windows_overlapping_final_window() -> None:
    seq_cfg = pretrain_sequence_config(make_seq_cfg(), make_pretrain_cfg())
    session = synthetic_frames()["A"]
    session = session[session["session_id"] == 2].rename(
        columns={"Timestamp": "ts", "FHR": "fhr", "Hr1_SignalQuality": "hr1_signal_quality"}
    )
    kept = cut_session_windows(session, seq_cfg, 30, 0.5, exclude_final_window=True)
    assert len(kept.windows) == 1 and kept.dropped_final_overlap == 1
    all_windows = cut_session_windows(session, seq_cfg, 30, 0.5, exclude_final_window=False)
    assert len(all_windows.windows) == 2


def test_missing_splits_fails_loudly_unless_allowed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_excluded_baby_ids(tmp_path / "missing.csv", allow_no_splits=False)
    assert load_excluded_baby_ids(tmp_path / "missing.csv", allow_no_splits=True) == set()
    assert load_excluded_baby_ids(write_splits(tmp_path), allow_no_splits=False) == {"B", "X"}


@pytest.mark.parametrize("as_directory", [False, True])
def test_build_pretrain_windows_excludes_val_test_and_writes_shards(
    tmp_path: Path, as_directory: bool
) -> None:
    source = write_parquet(tmp_path, as_directory)
    splits = write_splits(tmp_path)
    out_dir = tmp_path / "pretrain"
    stats = build_pretrain_windows(
        source, splits, out_dir, make_seq_cfg(), make_pretrain_cfg(), show_progress=False
    )

    # A: 4 (session 1) + 1 (session 2, one dropped for final-window overlap); C: 3; B excluded.
    assert stats.n_windows == 8
    assert stats.n_babies == 2
    assert stats.n_sessions == 3
    assert stats.n_excluded_baby_ids == 2
    assert stats.n_windows_dropped_final_overlap == 1
    assert stats.n_windows_dropped_low_signal == 2
    assert stats.n_steps == 60
    assert len(stats.shard_paths) == 2  # windows_per_shard=4 -> 4 + 4

    meta = json.loads(stats.meta_path.read_text())
    assert meta["channel_names"] == stats.channel_names
    assert meta["shards"] == [p.name for p in stats.shard_paths]
    assert meta["counts"]["windows"] == 8
    assert 129.0 < meta["normalization"]["means"][0] < 137.0

    baby_ids: list[str] = []
    for path in stats.shard_paths:
        data = np.load(path, allow_pickle=False)
        assert data["x"].dtype == np.float16
        assert data["x"].shape[1:] == (len(stats.channel_names), 60)
        assert data["baby_ids"].dtype.kind == "U"
        baby_ids.extend(data["baby_ids"].tolist())
    assert "B" not in baby_ids
    assert set(baby_ids) == {"A", "C"}


def test_build_pretrain_windows_without_splits_requires_explicit_flag(tmp_path: Path) -> None:
    source = write_parquet(tmp_path, as_directory=False)
    with pytest.raises(FileNotFoundError):
        build_pretrain_windows(
            source, tmp_path / "none.csv", tmp_path / "out", make_seq_cfg(), make_pretrain_cfg()
        )
    stats = build_pretrain_windows(
        source,
        tmp_path / "none.csv",
        tmp_path / "out",
        make_seq_cfg(),
        make_pretrain_cfg(),
        allow_no_splits=True,
        show_progress=False,
    )
    assert stats.n_babies == 3 and stats.n_windows == 12
