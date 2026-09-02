"""Cut unlabeled fixed-length CTG windows for self-supervised encoder pretraining.

Input contract (produced by the CTG_preprocess stage 3 side output): a parquet file or a
directory of ``*.parquet`` bucket files with columns ``BabyID`` (str), ``session_id``
(int, 1-based per pregnancy), ``Timestamp`` (1 Hz rows, duplicates already collapsed but
seconds may be missing), ``FHR`` (float, 0 = missing), ``toco`` (float),
``Hr1_SignalQuality`` (str) and ``in_final_window`` (bool, rows overlapping the supervised
last-hour window). BabyIDs match the labeled dataset where the pregnancy is labeled, so
val/test pregnancies can (and must) be excluded here.

Channels are built with the exact same ``_finalize_sequence`` used by the supervised
pipeline so the pretrained encoder weights are transferable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from ctg_ml.multimodal_config import MultimodalPretrainConfig, MultimodalSequenceConfig
from ctg_ml.multimodal_preprocess import _finalize_sequence, sequence_channel_names

EXCLUDED_SPLITS: tuple[str, ...] = ("val", "test")
# Sessions spanning more than this (after reindexing on Timestamp) are skipped: they are
# almost certainly stitched from unrelated recordings and would blow up the 1 Hz grid.
MAX_SESSION_SECONDS = 7 * 24 * 3600
SHARD_PREFIX = "windows"
META_FILENAME = "windows_meta.json"


@dataclass(frozen=True)
class PretrainBuildStats:
    n_windows: int
    n_babies: int
    n_sessions: int
    n_excluded_baby_ids: int
    n_sessions_skipped_too_long: int
    n_windows_dropped_low_signal: int
    n_windows_dropped_final_overlap: int
    n_steps: int
    channel_names: list[str]
    means: list[float]
    stds: list[float]
    shard_paths: list[Path]
    meta_path: Path


@dataclass(frozen=True)
class SessionWindowResult:
    windows: list[np.ndarray]
    window_starts_unix: list[int]
    skipped_too_long: bool
    dropped_low_signal: int
    dropped_final_overlap: int


PRETRAIN_SQL = """
SELECT
    CAST(c.BabyID AS VARCHAR) AS BabyID,
    CAST(c.session_id AS INTEGER) AS session_id,
    c."Timestamp" AS ts,
    CAST(c.FHR AS DOUBLE) AS fhr,
    CAST(c.toco AS DOUBLE) AS toco,
    CAST(c.Hr1_SignalQuality AS VARCHAR) AS hr1_signal_quality,
    {final_window_expr} AS in_final_window
FROM read_parquet(?) c
WHERE NOT EXISTS (SELECT 1 FROM excluded_ids e WHERE e.BabyID = c.BabyID)
ORDER BY BabyID, session_id, ts
"""


def resolve_parquet_source(path: str | Path) -> str:
    """Return a DuckDB ``read_parquet`` argument for a single file or a bucket directory."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No *.parquet files found in pretrain directory: {p}")
        return str(p / "*.parquet")
    if not p.exists():
        raise FileNotFoundError(f"Pretrain parquet not found: {p}")
    return str(p)


def load_excluded_baby_ids(splits_csv: str | Path | None, allow_no_splits: bool) -> set[str]:
    """BabyIDs in the supervised val/test splits; these must never enter pretraining."""
    if splits_csv is None or not Path(splits_csv).exists():
        if allow_no_splits:
            return set()
        raise FileNotFoundError(
            f"Missing splits file: {splits_csv}. Pretraining must exclude the supervised val/test "
            "BabyIDs. Run `scripts/make_splits_multimodal.py` first, or pass --allow-no-splits "
            "if there is deliberately no labeled split yet."
        )
    df = pd.read_csv(splits_csv, usecols=["BabyID", "split"], dtype={"BabyID": str})
    excluded = df.loc[df["split"].isin(EXCLUDED_SPLITS), "BabyID"]
    return {str(x) for x in excluded.tolist()}


def pretrain_sequence_config(
    seq_cfg: MultimodalSequenceConfig, pretrain_cfg: MultimodalPretrainConfig
) -> MultimodalSequenceConfig:
    """Supervised channel layout, pretraining window length, no padding of short windows."""
    if not seq_cfg.treat_fhr_zero_as_missing:
        raise ValueError(
            "Pretraining requires sequence.treat_fhr_zero_as_missing=true: the pretraining "
            "parquet encodes missing FHR (including missing seconds) as 0."
        )
    return replace(seq_cfg, window_minutes=pretrain_cfg.window_minutes, pad_short=False)


def _timestamps_to_unix_seconds(ts: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(ts)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed.to_numpy().astype("datetime64[s]").astype(np.int64)


def cut_session_windows(
    session: pd.DataFrame,
    seq_cfg: MultimodalSequenceConfig,
    stride_seconds: int,
    min_signal_fraction: float,
    exclude_final_window: bool,
) -> SessionWindowResult:
    """Cut fixed windows from one (BabyID, session_id) group.

    ``session`` has columns ``ts``, ``fhr``, ``toco``, ``hr1_signal_quality`` and
    ``in_final_window``. Rows are reindexed onto a contiguous 1 Hz grid: a missing
    second becomes FHR=0/toco=0 (i.e. missing), an empty quality string and
    ``in_final_window=False``.
    """
    n_steps = int(seq_cfg.window_minutes * 60 * seq_cfg.sample_rate_hz)
    if stride_seconds <= 0:
        raise ValueError("stride must be positive")
    empty = SessionWindowResult([], [], False, 0, 0)
    if session.empty:
        return empty

    unix = _timestamps_to_unix_seconds(session["ts"])
    order = np.argsort(unix, kind="stable")
    unix = unix[order]
    offsets = unix - unix[0]
    length = int(offsets[-1]) + 1
    if length > MAX_SESSION_SECONDS:
        return SessionWindowResult([], [], True, 0, 0)
    if length < n_steps:
        return empty

    fhr = np.zeros(length, dtype=np.float32)
    toco = np.zeros(length, dtype=np.float32)
    quality = np.full(length, "", dtype=object)
    final = np.zeros(length, dtype=bool)
    fhr[offsets] = session["fhr"].to_numpy(dtype=np.float32)[order]
    toco[offsets] = session["toco"].to_numpy(dtype=np.float32)[order]
    quality[offsets] = session["hr1_signal_quality"].fillna("").astype(str).to_numpy()[order]
    final[offsets] = session["in_final_window"].fillna(False).to_numpy(dtype=bool)[order]

    fhr[~np.isfinite(fhr)] = 0.0
    signal = np.concatenate([[0], np.cumsum(fhr != 0.0)])
    overlap = np.concatenate([[0], np.cumsum(final)])
    starts = np.arange(0, length - n_steps + 1, stride_seconds)
    signal_counts = signal[starts + n_steps] - signal[starts]
    keep = signal_counts >= min_signal_fraction * n_steps
    dropped_low_signal = int((~keep).sum())
    dropped_final = 0
    if exclude_final_window:
        overlaps = (overlap[starts + n_steps] - overlap[starts]) > 0
        dropped_final = int((keep & overlaps).sum())
        keep &= ~overlaps

    windows: list[np.ndarray] = []
    window_starts: list[int] = []
    for start in starts[keep]:
        s = int(start)
        group = pd.DataFrame(
            {
                "fhr": fhr[s : s + n_steps],
                "toco": toco[s : s + n_steps],
                "hr1_signal_quality": quality[s : s + n_steps],
            }
        )
        seq, _ = _finalize_sequence(group, seq_cfg)
        if seq is None:  # cannot happen: the group has exactly n_steps rows
            continue
        windows.append(seq)
        window_starts.append(int(unix[0]) + s)
    return SessionWindowResult(windows, window_starts, False, dropped_low_signal, dropped_final)


class _RunningStats:
    """Streaming mean/std over finite values of the FHR and toco channels."""

    def __init__(self) -> None:
        self.sum = np.zeros(2, dtype=np.float64)
        self.sumsq = np.zeros(2, dtype=np.float64)
        self.count = np.zeros(2, dtype=np.int64)

    def update(self, windows: np.ndarray) -> None:
        for ch in range(2):
            vals = windows[:, ch, :].astype(np.float64)
            vals = vals[np.isfinite(vals)]
            self.sum[ch] += vals.sum()
            self.sumsq[ch] += np.square(vals).sum()
            self.count[ch] += vals.size

    def finalize(self) -> tuple[list[float], list[float]]:
        means: list[float] = []
        stds: list[float] = []
        for ch in range(2):
            if self.count[ch] == 0:
                means.append(0.0)
                stds.append(1.0)
                continue
            mean = self.sum[ch] / self.count[ch]
            var = max(self.sumsq[ch] / self.count[ch] - mean * mean, 0.0)
            std = float(np.sqrt(var))
            means.append(float(mean))
            stds.append(std if std > 1e-6 else 1.0)
        return means, stds


def _write_shard(
    out_dir: Path,
    shard_index: int,
    windows: list[np.ndarray],
    baby_ids: list[str],
    session_ids: list[int],
    window_starts: list[int],
) -> Path:
    path = out_dir / f"{SHARD_PREFIX}_{shard_index:04d}.npz"
    x = np.stack(windows).astype(np.float16)
    np.savez_compressed(
        path,
        x=x,
        baby_ids=np.array(baby_ids, dtype=str),
        session_ids=np.asarray(session_ids, dtype=np.int32),
        window_start_unix=np.asarray(window_starts, dtype=np.int64),
    )
    return path


def build_pretrain_windows(
    pretrain_parquet: str | Path,
    splits_csv: str | Path | None,
    output_dir: str | Path,
    seq_cfg: MultimodalSequenceConfig,
    pretrain_cfg: MultimodalPretrainConfig,
    allow_no_splits: bool = False,
    show_progress: bool = True,
) -> PretrainBuildStats:
    source = resolve_parquet_source(pretrain_parquet)
    excluded = load_excluded_baby_ids(splits_csv, allow_no_splits)
    win_cfg = pretrain_sequence_config(seq_cfg, pretrain_cfg)
    channel_names = sequence_channel_names(win_cfg)
    n_steps = int(win_cfg.window_minutes * 60 * win_cfg.sample_rate_hz)
    stride_seconds = int(round(pretrain_cfg.stride_minutes * 60 * win_cfg.sample_rate_hz))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{SHARD_PREFIX}_*.npz"):
        stale.unlink()

    con = duckdb.connect(database=":memory:")
    stats = _RunningStats()
    pending: list[np.ndarray] = []
    pending_babies: list[str] = []
    pending_sessions: list[int] = []
    pending_starts: list[int] = []
    shard_paths: list[Path] = []
    babies: set[str] = set()
    sessions: set[tuple[str, int]] = set()
    n_windows = 0
    n_too_long = 0
    n_low_signal = 0
    n_final_overlap = 0
    pbar = tqdm(desc="pretrain windows", unit="baby", disable=not show_progress)

    def flush() -> None:
        if not pending:
            return
        shard_paths.append(
            _write_shard(
                out_dir, len(shard_paths), pending, pending_babies, pending_sessions, pending_starts
            )
        )
        pending.clear()
        pending_babies.clear()
        pending_sessions.clear()
        pending_starts.clear()

    def process_baby(baby_df: pd.DataFrame) -> None:
        nonlocal n_windows, n_too_long, n_low_signal, n_final_overlap
        baby_id = str(baby_df["BabyID"].iloc[0])
        for _, session in baby_df.groupby("session_id", sort=True):
            session_id = int(session["session_id"].iloc[0])
            result = cut_session_windows(
                session,
                win_cfg,
                stride_seconds=stride_seconds,
                min_signal_fraction=pretrain_cfg.min_signal_fraction,
                exclude_final_window=pretrain_cfg.exclude_final_window,
            )
            n_too_long += int(result.skipped_too_long)
            n_low_signal += result.dropped_low_signal
            n_final_overlap += result.dropped_final_overlap
            if not result.windows:
                continue
            batch = np.stack(result.windows)
            stats.update(batch)
            pending.extend(result.windows)
            pending_babies.extend([baby_id] * len(result.windows))
            pending_sessions.extend([session_id] * len(result.windows))
            pending_starts.extend(result.window_starts_unix)
            n_windows += len(result.windows)
            babies.add(baby_id)
            sessions.add((baby_id, session_id))
            if len(pending) >= pretrain_cfg.windows_per_shard:
                flush()
        pbar.update(1)

    try:
        columns = [
            desc[0]
            for desc in con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [source]).description
        ]
        required = {"BabyID", "session_id", "Timestamp", "FHR", "toco", "Hr1_SignalQuality"}
        if missing := (required - set(columns)):
            raise ValueError(f"Pretrain parquet is missing columns: {sorted(missing)}")
        if "in_final_window" in columns:
            final_expr = "CAST(c.in_final_window AS BOOLEAN)"
        elif pretrain_cfg.exclude_final_window:
            raise ValueError(
                "Pretrain parquet has no in_final_window column but exclude_final_window=true."
            )
        else:
            final_expr = "FALSE"
        con.register(
            "excluded_ids", pd.DataFrame({"BabyID": pd.Series(sorted(excluded), dtype=str)})
        )
        res = con.execute(PRETRAIN_SQL.format(final_window_expr=final_expr), [source])
        carry: pd.DataFrame | None = None
        while True:
            chunk = res.fetch_df_chunk(vectors_per_chunk=pretrain_cfg.chunk_vectors_per_batch)
            if chunk is None or chunk.empty:
                break
            if carry is not None and not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
                carry = None
            last_baby = str(chunk["BabyID"].iloc[-1])
            is_last = chunk["BabyID"].astype(str) == last_baby
            carry = chunk.loc[is_last].copy()
            full_chunk = chunk.loc[~is_last]
            if full_chunk.empty:
                continue
            for _, baby_df in full_chunk.groupby("BabyID", sort=False):
                process_baby(baby_df)
        if carry is not None and not carry.empty:
            process_baby(carry)
        flush()
    finally:
        pbar.close()
        con.close()

    means, stds = stats.finalize()
    meta_path = out_dir / META_FILENAME
    meta = {
        "source": source,
        "splits_csv": str(splits_csv) if splits_csv is not None else None,
        "channel_names": channel_names,
        "n_steps": n_steps,
        "window_minutes": win_cfg.window_minutes,
        "stride_minutes": pretrain_cfg.stride_minutes,
        "sample_rate_hz": win_cfg.sample_rate_hz,
        "min_signal_fraction": pretrain_cfg.min_signal_fraction,
        "exclude_final_window": pretrain_cfg.exclude_final_window,
        "x_dtype": "float16",
        "missing_encoding": "NaN in FHR/toco channels (raw, unnormalized values)",
        "counts": {
            "windows": n_windows,
            "babies": len(babies),
            "sessions": len(sessions),
            "excluded_baby_ids": len(excluded),
            "sessions_skipped_too_long": n_too_long,
            "windows_dropped_low_signal": n_low_signal,
            "windows_dropped_final_overlap": n_final_overlap,
        },
        "normalization": {"channels": ["FHR", "toco"], "means": means, "stds": stds},
        "shards": [p.name for p in shard_paths],
        "sequence_config": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(win_cfg).items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return PretrainBuildStats(
        n_windows=n_windows,
        n_babies=len(babies),
        n_sessions=len(sessions),
        n_excluded_baby_ids=len(excluded),
        n_sessions_skipped_too_long=n_too_long,
        n_windows_dropped_low_signal=n_low_signal,
        n_windows_dropped_final_overlap=n_final_overlap,
        n_steps=n_steps,
        channel_names=channel_names,
        means=means,
        stds=stds,
        shard_paths=shard_paths,
        meta_path=meta_path,
    )
