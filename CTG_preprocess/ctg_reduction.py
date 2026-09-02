from __future__ import annotations

import argparse
import base64
import gc
import shutil
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from config import (
    DEFAULT_BABYID_SALT,
    DEFAULT_PARTITION_OUTPUT_DIR,
    DEFAULT_PARTITION_REPORT_EVERY,
    DEFAULT_STAGE0_DIR,
    DEFAULT_STAGE1_CUTOFF_DATE,
    DEFAULT_STAGE1_DIR,
    DEFAULT_STAGE2_DIR,
    DEFAULT_STAGE2_EXTRA_COLUMNS,
    DEFAULT_STAGE3_ALL_SESSIONS_DIR,
    DEFAULT_STAGE3_BUCKETS,
    DEFAULT_STAGE3_DIR,
    DEFAULT_STAGE3_GAP_MINUTES,
    DEFAULT_STAGE3_LAST_HOUR_MINUTES,
    DEFAULT_STAGE3_OUTPUT_FILE,
    DEFAULT_STAGE3_PREG_GAP_DAYS,
    DEFAULT_STAGE3_WINDOW_SCOPE,
    DEFAULT_STAGE4_DIR,
    DEFAULT_STAGE4_DUP_THRESHOLD,
    DEFAULT_STAGE4_OUTPUT_FILE,
    DEFAULT_STAGE5_5_OUTPUT_FILE,
    DEFAULT_STAGE5_MIN_FHR_SECONDS,
    DEFAULT_STAGE5_OUTPUT_FILE,
)


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _field_type_or_default(
    dataset_or_schema: ds.Dataset | pa.Schema,
    name: str,
    default_type: pa.DataType = pa.string(),
) -> pa.DataType:
    schema = (
        dataset_or_schema.schema if isinstance(dataset_or_schema, ds.Dataset) else dataset_or_schema
    )
    if name in schema.names:
        return schema.field(name).type
    return default_type


def _column_or_default(batch: pa.RecordBatch, name: str, value_type: pa.DataType) -> pa.Array:
    idx = batch.schema.get_field_index(name)
    if idx != -1:
        return batch.column(idx)
    # ``value_type`` is only known as a generic ``pa.DataType`` at this point, so build the
    # all-null column via a null -> target cast (supported by Arrow for every data type).
    return pa.nulls(batch.num_rows).cast(value_type)


def _resolve_raw_parquet_files(input_dir: str | Path) -> list[str]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Raw input directory not found: {input_dir}")

    parquet_files = sorted(str(path) for path in input_dir.glob("*.parquet") if path.is_file())
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in raw input directory: {input_dir}")
    return parquet_files


def _unify_parquet_schemas(input_paths: list[str]) -> pa.Schema:
    schemas = []
    for path in input_paths:
        parquet_file = pq.ParquetFile(path)
        try:
            schemas.append(parquet_file.schema_arrow)
        finally:
            parquet_file.close()
    try:
        return pa.unify_schemas(schemas, promote_options="permissive")
    except TypeError:
        # Older PyArrow versions do not support permissive promotion.
        return pa.unify_schemas(schemas)


def _resolve_parquet_input_paths(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix != ".parquet":
            raise FileNotFoundError(f"Input file is not a parquet file: {input_path}")
        return [input_path]

    parquet_files = sorted(path for path in input_path.glob("*.parquet") if path.is_file())
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in input directory: {input_path}")
    return parquet_files


def _sql_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _duckdb_read_parquet_sql(input_path: str | Path, recursive: bool = False) -> str:
    input_path = Path(input_path)
    if input_path.is_dir():
        pattern = input_path / ("**/*.parquet" if recursive else "*.parquet")
    else:
        pattern = input_path
    return f"read_parquet('{_sql_quote(pattern)}')"


def _stage2_output_path(output_dir: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return output_dir / "part-0000.parquet"
    return output_dir / f"part-{shard_index:04d}-of-{shard_count:04d}.parquet"


def _build_stage2_shards(
    input_paths: list[Path], shard_count: int
) -> tuple[list[list[tuple[Path, int, int]]], int]:
    file_row_groups: list[tuple[Path, int]] = []
    total_row_groups = 0
    for path in input_paths:
        parquet_file = pq.ParquetFile(path)
        try:
            num_row_groups = parquet_file.metadata.num_row_groups
        finally:
            parquet_file.close()
        if num_row_groups == 0:
            continue
        file_row_groups.append((path, num_row_groups))
        total_row_groups += num_row_groups

    if total_row_groups == 0:
        raise RuntimeError("Stage2 input does not contain any row groups.")
    if shard_count < 1:
        raise ValueError("Stage2 shard count must be at least 1.")
    if shard_count > total_row_groups:
        raise ValueError(
            f"Stage2 shard count ({shard_count}) exceeds available row groups ({total_row_groups})."
        )

    shard_plans: list[list[tuple[Path, int, int]]] = []
    for shard_index in range(shard_count):
        global_start = total_row_groups * shard_index // shard_count
        global_stop = total_row_groups * (shard_index + 1) // shard_count
        shard_slices: list[tuple[Path, int, int]] = []
        offset = 0
        for path, num_row_groups in file_row_groups:
            file_start = offset
            file_stop = offset + num_row_groups
            local_start = max(global_start, file_start) - file_start
            local_stop = min(global_stop, file_stop) - file_start
            if local_start < local_stop:
                shard_slices.append((path, local_start, local_stop))
            offset = file_stop
        shard_plans.append(shard_slices)

    return shard_plans, total_row_groups


def _clear_parquet_outputs(output_dir: Path, stage_label: str) -> None:
    """Remove parquet files (and emptied partition directories) left by a previous run.

    ``ds.write_dataset(..., existing_data_behavior="overwrite_or_ignore")`` only overwrites
    part files with the same name; a rerun that produces fewer parts would otherwise leave
    stale files that later stages read as duplicated rows.
    """
    stale = [path for path in output_dir.rglob("*.parquet") if path.is_file()]
    for path in stale:
        path.unlink()
    for sub in sorted((p for p in output_dir.rglob("*") if p.is_dir()), reverse=True):
        if not any(sub.iterdir()):
            sub.rmdir()
    if stale:
        print(f"{stage_label}: removed {len(stale)} stale parquet files from {output_dir}")


def stage1_timefilter(
    input_dir: str | Path,
    output_dir: str | Path,
    cutoff_date: str,
    batch_size: int = 65536,
    report_every_batches: int = DEFAULT_PARTITION_REPORT_EVERY,
) -> None:
    input_paths = _resolve_raw_parquet_files(input_dir)
    print(f"Stage1: reading {len(input_paths)} raw parquet files from {input_dir}")
    schema = _unify_parquet_schemas(input_paths)
    dataset = ds.dataset(input_paths, format="parquet", schema=schema)
    cutoff_dt = _parse_date(cutoff_date)
    filter_expr = ds.field("Timestamp") >= cutoff_dt
    scanner = dataset.scanner(filter=filter_expr, batch_size=batch_size)

    def batch_iter():
        # simple progress counter
        batches = 0
        rows = 0
        for batch in scanner.to_batches():
            batches += 1
            rows += batch.num_rows
            if report_every_batches and batches % report_every_batches == 0:
                print(f"Stage1: {batches} batches, {rows} rows")
            yield batch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_parquet_outputs(output_dir, "Stage1")

    ds.write_dataset(
        batch_iter(),
        output_dir,
        format="parquet",
        existing_data_behavior="overwrite_or_ignore",
        schema=schema,
    )


FHR_SOURCE_COLUMNS = ("Hr1_0", "Hr1_1", "Hr1_2", "Hr1_3")


def _compute_fhr(batch: pa.RecordBatch) -> pa.Array:
    """Per-row mean of the strictly positive ``Hr1_*`` values (0.0 when none are positive)."""
    zero = pa.scalar(0)
    values: list[pa.Array] = []
    counts: list[pa.Array] = []
    for name in FHR_SOURCE_COLUMNS:
        raw = batch.column(batch.schema.get_field_index(name))
        col: pa.Array = pc.fill_null(raw, pa.scalar(0, type=raw.type))
        mask: pa.Array = pc.greater(col, zero)
        values.append(pc.if_else(mask, col, zero))
        counts.append(pc.cast(mask, pa.int32()))
    sums: pa.Array = values[0]
    total_counts: pa.Array = counts[0]
    for val, count in zip(values[1:], counts[1:], strict=True):
        sums = pc.add(sums, val)
        total_counts = pc.add(total_counts, count)
    has_vals: pa.Array = pc.greater(total_counts, zero)
    safe_counts: pa.Array = pc.if_else(has_vals, total_counts, pa.scalar(1))
    # The converter stores Hr1_* as int16; cast before dividing so the mean is not
    # truncated by integer division (140,141,142,143 must give 141.5, not 141).
    mean: pa.Array = pc.divide(pc.cast(sums, pa.float64()), safe_counts)
    fhr: pa.Array = pc.if_else(has_vals, mean, pa.scalar(0.0))
    return fhr.cast(pa.float32())


_B64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_B64_LUT = np.full(256, 0xFF, dtype=np.uint8)
for _i, _ch in enumerate(_B64_ALPHABET):
    _B64_LUT[_ch] = _i


def _toco_from_bytes(decoded: bytes) -> float:
    """Mean of the physiologically valid toco bytes (1..99); mean of all bytes if none; 0 if empty."""
    if not decoded:
        return 0.0
    valid = [value for value in decoded if 1 <= value <= 99]
    if valid:
        return sum(valid) / len(valid)
    return sum(decoded) / len(decoded)


def _fixed_width_string_bytes(arr: pa.Array, width: int) -> np.ndarray:
    """Concatenated bytes of a string array whose values are all ``width`` bytes long.

    Reads the Arrow data buffer directly (no Python objects) when the array is compact and
    every value really is ``width`` bytes; otherwise falls back to joining the Python strings
    with non-ASCII characters replaced, which the base64 lookup then rejects.
    """
    n = len(arr)
    buffers = arr.buffers() if n and arr.null_count == 0 and pa.types.is_string(arr.type) else []
    if len(buffers) == 3 and buffers[1] is not None and buffers[2] is not None:
        offsets = np.frombuffer(
            buffers[1], dtype=np.int32, count=n + 1, offset=arr.offset * 4
        ).astype(np.int64)
        if offsets[0] == 0 and offsets[-1] == width * n and bool(np.all(np.diff(offsets) == width)):
            return np.frombuffer(buffers[2], dtype=np.uint8, count=width * n)
    joined = "".join(str(v) for v in arr.to_pylist()).encode("ascii", errors="replace")
    return np.frombuffer(joined, dtype=np.uint8)


def _decode_toco_fast(raw: np.ndarray) -> np.ndarray:
    """Vectorised decode of concatenated 8-character base64 strings ending in ``==``.

    ``raw`` holds ``n * 8`` ASCII bytes. Returns an ``(n, 4)`` uint8 array; rows containing
    characters outside the base64 alphabet get a 0xFF marker in every column so the caller
    can fall back to the exact per-cell path.
    """
    if raw.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    sextets = _B64_LUT[raw].reshape(-1, 8)[:, :6].astype(np.uint16)
    bad = (sextets == 0xFF).any(axis=1)
    s0, s1, s2, s3, s4, s5 = (sextets[:, i] for i in range(6))
    out = np.empty((sextets.shape[0], 4), dtype=np.uint8)
    out[:, 0] = ((s0 << 2) | (s1 >> 4)) & 0xFF
    out[:, 1] = (((s1 & 0xF) << 4) | (s2 >> 2)) & 0xFF
    out[:, 2] = (((s2 & 0x3) << 6) | s3) & 0xFF
    out[:, 3] = ((s4 << 2) | (s5 >> 4)) & 0xFF
    out[bad] = 0xFF
    return out


def _compute_toco(batch: pa.RecordBatch) -> pa.Array:
    """Per-row toco from the base64 ``Toco_Values`` column (4 samples per second).

    The exports encode every second as an 8-character base64 string; those rows are decoded
    with numpy in bulk. Anything else (nulls, other lengths, malformed strings) goes through
    the exact per-cell fallback, so the result is identical to a pure-Python decode.
    """
    idx = batch.schema.get_field_index("Toco_Values")
    if idx == -1:
        return pa.array([0.0] * batch.num_rows, type=pa.float32())
    toco_vals = batch.column(idx)
    n = batch.num_rows
    out = np.zeros(n, dtype=np.float32)

    filled = cast(pa.StringArray, pc.fill_null(toco_vals, pa.scalar("", type=toco_vals.type)))
    lengths: pa.Array = pc.utf8_length(filled)
    is_fast: pa.Array = pc.and_(
        pc.equal(lengths, pa.scalar(8)),
        pc.ends_with(filled, pattern="=="),
    )
    fast_idx = np.flatnonzero(is_fast.to_numpy(zero_copy_only=False))
    if fast_idx.size:
        taken = pc.take(toco_vals, pa.array(fast_idx, type=pa.int64()))
        raw = _fixed_width_string_bytes(taken, 8)
        decoded = _decode_toco_fast(raw)
        bad = (decoded == 0xFF).all(axis=1)
        valid = (decoded >= 1) & (decoded <= 99)
        valid_count = valid.sum(axis=1)
        valid_sum = (decoded * valid).sum(axis=1, dtype=np.float64)
        mean_all = decoded.sum(axis=1, dtype=np.float64) / 4.0
        fast_out = np.where(valid_count > 0, valid_sum / np.maximum(valid_count, 1), mean_all)
        out[fast_idx[~bad]] = fast_out[~bad]
        slow_rows = np.setdiff1d(np.arange(n), fast_idx[~bad])
    else:
        slow_rows = np.arange(n)

    for row_idx in slow_rows:
        encoded = toco_vals[int(row_idx)].as_py()
        if encoded is None:
            continue
        try:
            out[row_idx] = _toco_from_bytes(base64.b64decode(encoded))
        except Exception:
            continue
    return pa.array(out, type=pa.float32())


def stage2_columnfilter(
    input_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 65536,
    report_every_batches: int = DEFAULT_PARTITION_REPORT_EVERY,
    shard_count: int = 1,
    shard_index: int | None = None,
) -> None:
    input_paths = _resolve_parquet_input_paths(input_dir)
    dataset = ds.dataset([str(path) for path in input_paths], format="parquet")
    input_schema = dataset.schema
    requested_batch_size = batch_size
    batch_size = min(batch_size, 8192)
    write_row_group_size = max(batch_size * 16, 131072)
    row_group_chunk_size = 1024
    extra_columns = [name for name in DEFAULT_STAGE2_EXTRA_COLUMNS if name in input_schema.names]
    columns = [
        "Timestamp",
        "PatientID",
        "RegistrationID",
        "Hr1_0",
        "Hr1_1",
        "Hr1_2",
        "Hr1_3",
        "Toco_Values",
        *extra_columns,
    ]
    registration_type = _field_type_or_default(input_schema, "RegistrationID")
    extra_column_types = {
        name: _field_type_or_default(input_schema, name) for name in extra_columns
    }

    schema = pa.schema(
        [
            ("Timestamp", input_schema.field("Timestamp").type),
            ("PatientID", input_schema.field("PatientID").type),
            ("RegistrationID", registration_type),
            ("FHR", pa.float32()),
            ("toco", pa.float32()),
            *[(name, extra_column_types[name]) for name in extra_columns],
        ]
    )

    if shard_index is not None and (shard_index < 0 or shard_index >= shard_count):
        raise ValueError(f"Stage2 shard index must be within 0..{shard_count - 1}.")

    shard_plans, total_row_groups = _build_stage2_shards(input_paths, shard_count)
    selected_shards = [shard_index] if shard_index is not None else list(range(shard_count))
    resume_mode = shard_count > 1 or shard_index is not None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = {
        _stage2_output_path(output_dir, idx, shard_count).name for idx in range(shard_count)
    }
    extra_outputs = [
        path for path in output_dir.glob("*.parquet") if path.name not in expected_outputs
    ]
    if extra_outputs:
        sample = ", ".join(path.name for path in extra_outputs[:3])
        raise RuntimeError(
            f"Stage2 output directory contains existing parquet files ({sample}). "
            f"Use a clean output directory or remove unexpected files first: {output_dir}"
        )
    if requested_batch_size != batch_size:
        print(
            f"Stage2: capping batch size from {requested_batch_size} to {batch_size} "
            "to reduce peak RAM usage"
        )
    print(
        f"Stage2: reading {len(input_paths)} parquet file(s) across {total_row_groups} row groups"
    )
    print(
        f"Stage2: buffering output into row groups of about {write_row_group_size} rows "
        "to avoid excessive Parquet metadata growth"
    )
    if shard_count > 1:
        shard_labels = ", ".join(str(idx) for idx in selected_shards)
        print(
            f"Stage2: sharding output into {shard_count} restartable shard(s); "
            f"this run will process shard index {shard_labels}"
        )

    memory_pool = pa.default_memory_pool()

    def transform_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
        timestamp = batch.column(batch.schema.get_field_index("Timestamp"))
        patient_id = batch.column(batch.schema.get_field_index("PatientID"))
        registration_id = _column_or_default(batch, "RegistrationID", registration_type)
        fhr = _compute_fhr(batch)
        toco = _compute_toco(batch)
        extras = [
            _column_or_default(batch, name, extra_column_types[name]) for name in extra_columns
        ]
        return pa.RecordBatch.from_arrays(
            [timestamp, patient_id, registration_id, fhr, toco, *extras],
            schema=schema,
        )

    def flush_buffer(
        writer: pq.ParquetWriter,
        buffered_batches: list[pa.RecordBatch],
        buffered_rows: int,
        force: bool = False,
    ) -> tuple[list[pa.RecordBatch], int]:
        if not buffered_batches:
            return buffered_batches, buffered_rows

        table = pa.Table.from_batches(buffered_batches, schema=schema)
        rows_to_write = (
            table.num_rows
            if force
            else (table.num_rows // write_row_group_size) * write_row_group_size
        )
        if rows_to_write:
            writer.write_table(table.slice(0, rows_to_write), row_group_size=write_row_group_size)
        remainder = table.slice(rows_to_write)
        next_batches = remainder.to_batches(max_chunksize=batch_size)
        next_rows = remainder.num_rows
        del remainder
        del table
        gc.collect()
        memory_pool.release_unused()
        return next_batches, next_rows

    for selected_shard in selected_shards:
        shard_slices = shard_plans[selected_shard]
        final_path = _stage2_output_path(output_dir, selected_shard, shard_count)
        temp_path = output_dir / f".{final_path.name}.tmp"
        shard_row_groups = sum(stop - start for _, start, stop in shard_slices)

        if final_path.exists():
            if resume_mode:
                print(
                    f"Stage2: skipping completed shard {selected_shard + 1}/{shard_count} "
                    f"({final_path.name})"
                )
                continue
            final_path.unlink()
        if temp_path.exists():
            temp_path.unlink()

        print(
            f"Stage2: shard {selected_shard + 1}/{shard_count} -> {final_path.name} "
            f"({shard_row_groups} row groups)"
        )

        writer = pq.ParquetWriter(temp_path, schema=schema)
        try:
            buffered_batches: list[pa.RecordBatch] = []
            buffered_rows = 0
            batches = 0
            rows = 0

            for input_path, row_group_start, row_group_stop in shard_slices:
                parquet_file = pq.ParquetFile(input_path)
                try:
                    for chunk_start in range(row_group_start, row_group_stop, row_group_chunk_size):
                        row_groups = list(
                            range(
                                chunk_start, min(chunk_start + row_group_chunk_size, row_group_stop)
                            )
                        )
                        for batch in parquet_file.iter_batches(
                            batch_size=batch_size,
                            row_groups=row_groups,
                            columns=columns,
                            use_threads=False,
                        ):
                            batches += 1
                            rows += batch.num_rows
                            if report_every_batches and batches % report_every_batches == 0:
                                print(
                                    f"Stage2 shard {selected_shard + 1}/{shard_count}: "
                                    f"{batches} batches, {rows} rows"
                                )

                            out_batch = transform_batch(batch)
                            buffered_batches.append(out_batch)
                            buffered_rows += out_batch.num_rows
                            if buffered_rows >= write_row_group_size:
                                buffered_batches, buffered_rows = flush_buffer(
                                    writer,
                                    buffered_batches,
                                    buffered_rows,
                                )
                finally:
                    parquet_file.close()

            buffered_batches, buffered_rows = flush_buffer(
                writer,
                buffered_batches,
                buffered_rows,
                force=True,
            )
        except Exception:
            writer.close()
            if temp_path.exists():
                temp_path.unlink()
            raise
        else:
            writer.close()
            temp_path.replace(final_path)
            gc.collect()
            memory_pool.release_unused()


def _stage3_babyid_expr(hash_func: str, babyid_salt: str) -> str:
    """SQL expression hashing ``salt|PatientID|session_end`` into the pseudonymous BabyID."""
    salt = babyid_salt.replace("'", "''")
    return (
        f"{hash_func}(concat('{salt}', '|', CAST(PatientID AS VARCHAR),"
        f" '|', CAST(session_end AS VARCHAR)))"
    )


def _stage3_bucket_expr(bucket_count: int) -> str:
    """SQL expression assigning each PatientID to a bucket in ``0..bucket_count-1``."""
    return (
        "CAST(("
        "COALESCE("
        "try_cast(right(CAST(PatientID AS VARCHAR), 4) AS UBIGINT), "
        "hash(CAST(PatientID AS VARCHAR))"
        f") % {bucket_count}) AS INTEGER)"
    )


def stage3_sessionfilter(
    input_dir: str | Path,
    output_file: str | Path,
    gap_minutes: int = DEFAULT_STAGE3_GAP_MINUTES,
    preg_gap_days: int = DEFAULT_STAGE3_PREG_GAP_DAYS,
    last_hour_minutes: int = DEFAULT_STAGE3_LAST_HOUR_MINUTES,
    babyid_salt: str = DEFAULT_BABYID_SALT,
    show_progress: bool = True,
    bucket_count: int = DEFAULT_STAGE3_BUCKETS,
    bucket_index: int | None = None,
    prebucket: bool = True,
    window_scope: str = DEFAULT_STAGE3_WINDOW_SCOPE,
    all_sessions_output: str | Path | None = None,
) -> None:
    if window_scope not in STAGE3_WINDOW_SCOPES:
        raise ValueError(
            f"window_scope must be one of {STAGE3_WINDOW_SCOPES}, got {window_scope!r}"
        )
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for stage3. Install it with pip/uv.") from exc

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    if show_progress:
        try:
            con.execute("PRAGMA enable_progress_bar")
            con.execute("PRAGMA progress_bar_time=5")
        except Exception:
            pass
    try:
        con.execute("SET preserve_insertion_order=false")
    except Exception:
        pass
    source_sql = _duckdb_read_parquet_sql(input_dir)

    def _pick_hash_func() -> str:
        for func in ("sha256", "md5"):
            try:
                con.execute(f"SELECT {func}('test')").fetchone()
                return func
            except Exception:
                continue
        return "md5"

    hash_func = _pick_hash_func()
    babyid_expr = _stage3_babyid_expr(hash_func, babyid_salt)

    def _bucket_expr() -> str:
        return _stage3_bucket_expr(bucket_count)

    def _build_query(extra_where: str) -> str:
        return _stage3_query(
            babyid_expr,
            gap_minutes=gap_minutes,
            preg_gap_days=preg_gap_days,
            last_hour_minutes=last_hour_minutes,
            extra_where=extra_where,
            window_scope=window_scope,
        )

    def _build_all_sessions_query(extra_where: str) -> str:
        return _stage3_all_sessions_query(
            babyid_expr,
            gap_minutes=gap_minutes,
            preg_gap_days=preg_gap_days,
            last_hour_minutes=last_hour_minutes,
            extra_where=extra_where,
        )

    all_sessions_dir = Path(all_sessions_output) if all_sessions_output else None
    if all_sessions_dir is not None:
        all_sessions_dir.mkdir(parents=True, exist_ok=True)
        if bucket_index is None:
            for stale in all_sessions_dir.glob("all_sessions*.parquet"):
                stale.unlink()
        print(f"Stage3: also writing all sessions for pretraining to {all_sessions_dir}")

    output_path = Path(output_file)
    prebucket_dir: Path | None = None
    try:
        if bucket_count and bucket_count > 1:
            base_dir = output_path if output_path.suffix == "" else output_path.parent
            base_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_path.stem if output_path.suffix else "stage3_sessions"

            if bucket_index is None:
                for stale in base_dir.glob(f"{prefix}_bucket_*.parquet"):
                    stale.unlink()

            if bucket_index is None and prebucket:
                prebucket_dir = base_dir / f".{prefix}_stage3_input_buckets_{bucket_count}"
                if prebucket_dir.exists():
                    shutil.rmtree(prebucket_dir)
                prebucket_dir.mkdir(parents=True, exist_ok=True)
                print(
                    "Stage3: pre-bucketing input by PatientID in one pass "
                    f"to avoid {bucket_count} full parquet scans"
                )
                con.execute(
                    f"""
                    COPY (
                        SELECT *, {_bucket_expr()} AS patient_bucket
                        FROM {source_sql}
                    )
                    TO '{_sql_quote(prebucket_dir)}'
                    (FORMAT PARQUET, PARTITION_BY (patient_bucket))
                    """
                )

            indices = [bucket_index] if bucket_index is not None else range(bucket_count)
            for idx in indices:
                out_path = base_dir / f"{prefix}_bucket_{idx:04d}.parquet"
                print(f"Stage3 bucket {idx + 1}/{bucket_count}: {out_path.name}")
                if bucket_index is None and prebucket_dir is not None:
                    bucket_path = prebucket_dir / f"patient_bucket={idx}"
                    if not bucket_path.exists():
                        print(f"Stage3 bucket {idx + 1}/{bucket_count}: no input rows, skipping")
                        continue
                    con.execute(
                        "CREATE OR REPLACE VIEW ctg AS SELECT * FROM "
                        f"{_duckdb_read_parquet_sql(bucket_path)}"
                    )
                    bucket_where = ""
                else:
                    con.execute(f"CREATE OR REPLACE VIEW ctg AS SELECT * FROM {source_sql}")
                    bucket_where = f"{_bucket_expr()} = {idx}"
                query = _build_query(bucket_where)

                if out_path.exists():
                    out_path.unlink()
                con.execute("COPY (" + query + ") TO ? (FORMAT PARQUET)", [str(out_path)])
                if all_sessions_dir is not None:
                    all_path = all_sessions_dir / f"all_sessions_bucket_{idx:04d}.parquet"
                    if all_path.exists():
                        all_path.unlink()
                    con.execute(
                        "COPY ("
                        + _build_all_sessions_query(bucket_where)
                        + ") TO ? (FORMAT PARQUET)",
                        [str(all_path)],
                    )
        else:
            if output_path.exists():
                output_path.unlink()
            con.execute(f"CREATE OR REPLACE VIEW ctg AS SELECT * FROM {source_sql}")
            query = _build_query("")
            con.execute("COPY (" + query + ") TO ? (FORMAT PARQUET)", [str(output_path)])
            if all_sessions_dir is not None:
                all_path = all_sessions_dir / "all_sessions.parquet"
                if all_path.exists():
                    all_path.unlink()
                con.execute(
                    "COPY (" + _build_all_sessions_query("") + ") TO ? (FORMAT PARQUET)",
                    [str(all_path)],
                )
    finally:
        con.close()
        if prebucket_dir is not None and prebucket_dir.exists():
            shutil.rmtree(prebucket_dir)


STAGE3_WINDOW_SCOPES = ("pregnancy", "final_session")

STAGE3_SIGNAL_COLUMNS = (
    "PatientID",
    "RegistrationID",
    "Timestamp",
    "FHR",
    "toco",
    "Hr1_SignalQuality",
    "Hr1Mode",
    "TocoMode",
)


def _stage3_common_ctes(gap_minutes: int, preg_gap_days: int, extra_where: str) -> str:
    """CTEs shared by the Stage 3 queries over a view ``ctg`` with the Stage 2 columns.

    ``sessioned`` splits each patient's rows into sessions at gaps longer than
    ``gap_minutes``; ``preg_sessions`` groups sessions into pregnancies at gaps longer than
    ``preg_gap_days``; ``final_sessions`` gives each pregnancy's last session end (the key
    used for the BabyID hash); ``pregnancy_anchors`` gives the last non-zero FHR of the
    whole pregnancy (falling back to its final session end).
    """
    where_clause = ("\n    WHERE " + extra_where) if extra_where else ""
    columns = ",\n        ".join(STAGE3_SIGNAL_COLUMNS)
    # DISTINCT: the raw exports overlap in time (the same recording is exported more than
    # once), so identical rows are collapsed here before sessionisation. Rows that differ
    # for the same second are kept and handled by the Stage 4 conflict rule.
    return f"""
WITH deduplicated AS (
    SELECT DISTINCT
        {columns}
    FROM ctg{where_clause}
),
ordered AS (
    SELECT
        *,
        Timestamp - LAG(Timestamp) OVER (PARTITION BY PatientID ORDER BY Timestamp) AS gap
    FROM deduplicated
),
sessioned AS (
    SELECT
        *,
        SUM(CASE WHEN gap IS NULL OR gap > INTERVAL '{gap_minutes} minutes'
            THEN 1 ELSE 0 END
        ) OVER (PARTITION BY PatientID ORDER BY Timestamp) AS session_id
    FROM ordered
),
session_end AS (
    SELECT PatientID, session_id, MAX(Timestamp) AS session_end
    FROM sessioned
    GROUP BY PatientID, session_id
),
preg_sessions AS (
    SELECT
        PatientID,
        session_id,
        session_end,
        SUM(CASE WHEN prev_end IS NULL OR session_end - prev_end > INTERVAL '{preg_gap_days} days'
            THEN 1 ELSE 0 END
        ) OVER (PARTITION BY PatientID ORDER BY session_end) AS pregnancy_id
    FROM (
        SELECT
            *,
            LAG(session_end) OVER (PARTITION BY PatientID ORDER BY session_end) AS prev_end
        FROM session_end
    )
),
final_sessions AS (
    SELECT PatientID, pregnancy_id, MAX(session_end) AS session_end
    FROM preg_sessions
    GROUP BY PatientID, pregnancy_id
),
pregnancy_anchors AS (
    SELECT
        s.PatientID,
        p.pregnancy_id,
        f.session_end,
        COALESCE(MAX(s.Timestamp) FILTER (WHERE s.FHR > 0), f.session_end) AS anchor_ts
    FROM sessioned s
    JOIN preg_sessions p
      ON s.PatientID = p.PatientID AND s.session_id = p.session_id
    JOIN final_sessions f
      ON f.PatientID = p.PatientID AND f.pregnancy_id = p.pregnancy_id
    GROUP BY s.PatientID, p.pregnancy_id, f.session_end
)"""


def _stage3_query(
    babyid_expr: str,
    *,
    gap_minutes: int,
    preg_gap_days: int,
    last_hour_minutes: int,
    extra_where: str = "",
    window_scope: str = DEFAULT_STAGE3_WINDOW_SCOPE,
) -> str:
    """Stage 3 SQL: the last ``last_hour_minutes`` of signal per pregnancy, with BabyID.

    ``window_scope="pregnancy"`` anchors on the last non-zero FHR anywhere in the pregnancy
    and keeps rows from every session inside the window. ``"final_session"`` is the legacy
    behaviour that only looks at the pregnancy's last session, which silently discards the
    whole labour when the recording ends with a short reconnection or a signal-less tail.
    """
    if window_scope not in STAGE3_WINDOW_SCOPES:
        raise ValueError(
            f"window_scope must be one of {STAGE3_WINDOW_SCOPES}, got {window_scope!r}"
        )
    common = _stage3_common_ctes(gap_minutes, preg_gap_days, extra_where)
    signal = ",\n        ".join(f"s.{name}" for name in STAGE3_SIGNAL_COLUMNS)
    window = f"BETWEEN a.anchor_ts - INTERVAL '{last_hour_minutes} minutes' AND a.anchor_ts"
    if window_scope == "pregnancy":
        final_rows = f"""
final_rows AS (
    SELECT
        {signal},
        a.session_end
    FROM sessioned s
    JOIN preg_sessions p
      ON s.PatientID = p.PatientID AND s.session_id = p.session_id
    JOIN pregnancy_anchors a
      ON a.PatientID = p.PatientID AND a.pregnancy_id = p.pregnancy_id
    WHERE s.Timestamp {window}
)"""
    else:
        final_rows = f"""
final_session_anchors AS (
    SELECT
        s.PatientID,
        s.session_id,
        p.session_end,
        COALESCE(MAX(s.Timestamp) FILTER (WHERE s.FHR > 0), p.session_end) AS anchor_ts
    FROM sessioned s
    JOIN preg_sessions p
      ON s.PatientID = p.PatientID AND s.session_id = p.session_id
    JOIN final_sessions f
      ON f.PatientID = p.PatientID
     AND f.pregnancy_id = p.pregnancy_id
     AND f.session_end = p.session_end
    GROUP BY s.PatientID, s.session_id, p.session_end
),
final_rows AS (
    SELECT
        {signal},
        a.session_end
    FROM sessioned s
    JOIN final_session_anchors a
      ON s.PatientID = a.PatientID AND s.session_id = a.session_id
    WHERE s.Timestamp {window}
)"""
    output = ",\n    ".join(STAGE3_SIGNAL_COLUMNS)
    return f"""{common},{final_rows}
SELECT
    {babyid_expr} AS BabyID,
    {output}
FROM final_rows
"""


def _stage3_all_sessions_query(
    babyid_expr: str,
    *,
    gap_minutes: int,
    preg_gap_days: int,
    last_hour_minutes: int,
    extra_where: str = "",
) -> str:
    """Stage 3 side output for pretraining: every session of every pregnancy.

    Rows carry the same BabyID as the supervised window (so labelled val/test pregnancies
    can be excluded downstream), a 1-based ``session_id`` within the pregnancy and an
    ``in_final_window`` flag for rows overlapping the supervised window. Duplicate
    timestamps are collapsed. PatientID is deliberately not included.
    """
    common = _stage3_common_ctes(gap_minutes, preg_gap_days, extra_where)
    return f"""{common},
session_rows AS (
    SELECT
        s.PatientID,
        s.Timestamp,
        s.FHR,
        s.toco,
        s.Hr1_SignalQuality,
        s.Hr1Mode,
        s.TocoMode,
        a.session_end,
        a.anchor_ts,
        DENSE_RANK() OVER (
            PARTITION BY s.PatientID, p.pregnancy_id ORDER BY p.session_id
        ) AS session_idx
    FROM sessioned s
    JOIN preg_sessions p
      ON s.PatientID = p.PatientID AND s.session_id = p.session_id
    JOIN pregnancy_anchors a
      ON a.PatientID = p.PatientID AND a.pregnancy_id = p.pregnancy_id
)
SELECT
    {babyid_expr} AS BabyID,
    CAST(session_idx AS INTEGER) AS session_id,
    Timestamp,
    CAST(COALESCE(
        avg(FHR) FILTER (WHERE FHR > 0 AND FHR < 255),
        avg(FHR) FILTER (WHERE FHR > 0),
        0
    ) AS FLOAT) AS FHR,
    CAST(COALESCE(avg(toco) FILTER (WHERE toco BETWEEN 1 AND 99), avg(toco), 0) AS FLOAT) AS toco,
    COALESCE(mode(Hr1_SignalQuality), MIN(Hr1_SignalQuality)) AS Hr1_SignalQuality,
    COALESCE(mode(Hr1Mode), MIN(Hr1Mode)) AS Hr1Mode,
    COALESCE(mode(TocoMode), MIN(TocoMode)) AS TocoMode,
    bool_or(
        Timestamp BETWEEN anchor_ts - INTERVAL '{last_hour_minutes} minutes' AND anchor_ts
    ) AS in_final_window
FROM session_rows
GROUP BY PatientID, session_end, session_idx, Timestamp
"""


def stage4_duplicatefilter(
    input_dir: str | Path,
    output_file: str | Path,
    dup_threshold: float = DEFAULT_STAGE4_DUP_THRESHOLD,
    show_progress: bool = True,
) -> None:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for stage4. Install it with pip/uv.") from exc

    input_path = Path(input_dir)
    input_files = sorted(input_path.glob("*.parquet")) if input_path.is_dir() else [input_path]
    input_files = [p for p in input_files if p.exists()]
    if not input_files:
        raise FileNotFoundError(f"No parquet files found for stage4 input: {input_dir}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_path.stem if output_path.suffix else "stage4_dedup"
    for stale in output_path.parent.glob(f"{prefix}*.parquet"):
        stale.unlink()

    query = _stage4_query(dup_threshold)

    total = len(input_files)
    for idx, in_file in enumerate(input_files, start=1):
        out_file = output_path.parent / f"{prefix}_{idx - 1:04d}.parquet"
        con = duckdb.connect()
        if show_progress:
            try:
                con.execute("PRAGMA enable_progress_bar")
                con.execute("PRAGMA progress_bar_time=5")
            except Exception:
                pass
        try:
            con.execute("SET preserve_insertion_order=false")
        except Exception:
            pass

        try:
            con.execute(f"CREATE VIEW ctg AS SELECT * FROM {_duckdb_read_parquet_sql(in_file)}")
            con.execute("COPY (" + query + ") TO ? (FORMAT PARQUET)", [str(out_file)])
        finally:
            con.close()
        print(f"Stage4: {idx}/{total} buckets -> {out_file.name}")


def _stage4_query(dup_threshold: float) -> str:
    """Stage 4 SQL over a view ``ctg`` with the Stage 3 columns.

    Duplicate timestamps come from two sources: the same signal exported twice (identical
    FHR/toco, harmless) and genuinely conflicting rows for the same second (two registrations
    or two monitors). Only the *conflicting* share counts towards ``dup_threshold``; exact
    duplicates are simply collapsed. Counting exact duplicates as well previously dropped a
    third of all pregnancies.
    """
    return f"""
    WITH ts_counts AS (
        SELECT
            BabyID,
            Timestamp,
            COUNT(DISTINCT (
                COALESCE(FHR, -1),
                COALESCE(toco, -1),
                COALESCE(CAST(Hr1_SignalQuality AS VARCHAR), '')
            )) AS distinct_cnt
        FROM ctg
        GROUP BY BabyID, Timestamp
    ),
    per_baby AS (
        SELECT
            BabyID,
            SUM(CASE WHEN distinct_cnt > 1 THEN 1 ELSE 0 END) AS conflict_ts,
            COUNT(*) AS total_ts
        FROM ts_counts
        GROUP BY BabyID
    ),
    keep_baby AS (
        SELECT BabyID
        FROM per_baby
        WHERE CASE WHEN total_ts = 0 THEN 0 ELSE conflict_ts * 1.0 / total_ts END
            <= {dup_threshold}
    ),
    filtered AS (
        SELECT c.*
        FROM ctg c
        JOIN keep_baby k USING (BabyID)
    ),
    ranked AS (
        SELECT
            *,
            CASE upper(trim(CAST(Hr1_SignalQuality AS VARCHAR)))
                WHEN 'G' THEN 1
                WHEN 'Y' THEN 2
                WHEN 'R' THEN 3
                ELSE 4
            END AS signal_quality_rank
        FROM filtered
    ),
    best_quality AS (
        SELECT BabyID, Timestamp, MIN(signal_quality_rank) AS signal_quality_rank
        FROM ranked
        GROUP BY BabyID, Timestamp
    ),
    selected AS (
        SELECT r.*
        FROM ranked r
        JOIN best_quality b
          ON r.BabyID = b.BabyID
         AND r.Timestamp = b.Timestamp
         AND r.signal_quality_rank = b.signal_quality_rank
    ),
    agg AS (
        SELECT
            BabyID,
            MIN(PatientID) AS PatientID,
            Timestamp,
            CAST(COALESCE(
                avg(FHR) FILTER (WHERE FHR > 0 AND FHR < 255),
                avg(FHR) FILTER (WHERE FHR > 0),
                0
            ) AS FLOAT) AS FHR,
            CAST(COALESCE(
                avg(toco) FILTER (WHERE toco BETWEEN 1 AND 99),
                avg(toco)
            ) AS FLOAT) AS toco,
            CASE MIN(signal_quality_rank)
                WHEN 1 THEN 'G'
                WHEN 2 THEN 'Y'
                WHEN 3 THEN 'R'
                ELSE COALESCE(mode(Hr1_SignalQuality), MIN(Hr1_SignalQuality))
            END AS Hr1_SignalQuality,
            COALESCE(mode(Hr1Mode), MIN(Hr1Mode)) AS Hr1Mode,
            COALESCE(mode(TocoMode), MIN(TocoMode)) AS TocoMode
        FROM selected
        GROUP BY BabyID, Timestamp
    )
    SELECT * FROM agg
    """


def stage5_qualityfilter(
    input_dir: str | Path,
    output_file: str | Path,
    min_fhr_seconds: int = DEFAULT_STAGE5_MIN_FHR_SECONDS,
    show_progress: bool = True,
) -> None:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for stage5. Install it with pip/uv.") from exc

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    con = duckdb.connect()
    if show_progress:
        try:
            con.execute("PRAGMA enable_progress_bar")
            con.execute("PRAGMA progress_bar_time=5")
        except Exception:
            pass
    try:
        con.execute("SET preserve_insertion_order=false")
    except Exception:
        pass

    con.execute(f"CREATE VIEW ctg AS SELECT * FROM {_duckdb_read_parquet_sql(input_dir)}")

    query = f"""
    WITH per_baby AS (
        SELECT
            BabyID,
            SUM(CASE WHEN FHR > 0 THEN 1 ELSE 0 END) AS fhr_nz
        FROM ctg
        GROUP BY BabyID
    ),
    keep AS (
        SELECT BabyID
        FROM per_baby
        WHERE fhr_nz >= {min_fhr_seconds}
    )
    SELECT c.*
    FROM ctg c
    JOIN keep k USING (BabyID)
    """

    try:
        con.execute("COPY (" + query + ") TO ? (FORMAT PARQUET)", [str(output_file)])
    finally:
        con.close()


def stage6_partitioning(
    input_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 65536,
    report_every_batches: int = DEFAULT_PARTITION_REPORT_EVERY,
) -> None:
    dataset = ds.dataset(str(input_path), format="parquet")
    has_ctg_date = "ctg_date" in dataset.schema.names
    columns = list(dataset.schema.names)

    scanner = dataset.scanner(columns=columns, batch_size=batch_size)

    total_rows = None
    try:
        total_rows = dataset.count_rows()
    except Exception:
        total_rows = None
    total_batches = None
    if total_rows is not None and batch_size:
        total_batches = (total_rows + batch_size - 1) // batch_size

    anchor_babies: pa.Array | None = None
    anchor_dates: pa.Array | None = None
    if not has_ctg_date:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for stage6 when ctg_date is missing.") from exc

        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW ctg AS SELECT BabyID, Timestamp FROM {_duckdb_read_parquet_sql(input_path)}"
        )
        try:
            anchor_rows = con.execute(
                "SELECT BabyID, CAST(MAX(Timestamp) AS DATE) AS anchor_date FROM ctg GROUP BY BabyID"
            ).fetchall()
        finally:
            con.close()
        if not anchor_rows:
            print("Stage6: no rows found in input.")
            return
        baby_type = dataset.schema.field("BabyID").type
        anchor_babies = pa.array([r[0] for r in anchor_rows], type=baby_type)
        anchor_dates = pa.array([r[1] for r in anchor_rows], type=pa.date32())
    else:
        print("Stage6: using ctg_date from input (sorted Stage 5.5 output).")

    base_fields = [dataset.schema.field(name) for name in columns]
    if has_ctg_date:
        schema = pa.schema(base_fields)
    else:
        schema = pa.schema(
            base_fields
            + [
                pa.field("ctg_date", pa.date32()),
            ]
        )

    def batch_iter():
        import time

        start_time = time.perf_counter()
        batches = 0
        rows = 0
        for batch in scanner.to_batches():
            batches += 1
            rows += batch.num_rows
            if report_every_batches and batches % report_every_batches == 0:
                elapsed = time.perf_counter() - start_time
                rate = rows / elapsed if elapsed else 0.0
                if total_batches:
                    pct = batches / total_batches * 100.0
                    print(
                        f"Stage6: {batches}/{total_batches} batches ({pct:.1f}%) "
                        f"{rows} rows ({rate:,.0f} rows/s)"
                    )
                else:
                    print(f"Stage6: {batches} batches, {rows} rows ({rate:,.0f} rows/s)")
            if not has_ctg_date:
                if anchor_babies is None or anchor_dates is None:
                    raise RuntimeError("Stage6: anchor dates were not computed.")
                baby = batch.column(batch.schema.get_field_index("BabyID"))
                idx = pc.index_in(baby, value_set=anchor_babies)
                ctg_date = pc.take(anchor_dates, idx).cast(pa.date32())
                batch = batch.append_column("ctg_date", ctg_date)
            yield batch
        if report_every_batches:
            elapsed = time.perf_counter() - start_time
            rate = rows / elapsed if elapsed else 0.0
            if total_batches:
                print(
                    f"Stage6 done: {batches}/{total_batches} batches "
                    f"{rows} rows ({rate:,.0f} rows/s)"
                )
            else:
                print(f"Stage6 done: {batches} batches, {rows} rows ({rate:,.0f} rows/s)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_parquet_outputs(output_dir, "Stage6")
    ds.write_dataset(
        batch_iter(),
        output_dir,
        format="parquet",
        partitioning=["ctg_date"],
        existing_data_behavior="overwrite_or_ignore",
        max_open_files=64,
        schema=schema,
    )


def stage5_5_sort(
    input_file: str | Path,
    output_file: str | Path,
    show_progress: bool = True,
) -> None:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for stage5.5. Install it with pip/uv.") from exc

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    con = duckdb.connect()
    if show_progress:
        try:
            con.execute("PRAGMA enable_progress_bar")
            con.execute("PRAGMA progress_bar_time=5")
        except Exception:
            pass
    try:
        con.execute("SET preserve_insertion_order=false")
    except Exception:
        pass

    con.execute(f"CREATE VIEW ctg AS SELECT * FROM {_duckdb_read_parquet_sql(input_file)}")

    query = """
    WITH anchors AS (
        SELECT BabyID, CAST(MAX(Timestamp) AS DATE) AS ctg_date
        FROM ctg
        GROUP BY BabyID
    )
    SELECT c.*, a.ctg_date
    FROM ctg c
    JOIN anchors a USING (BabyID)
    ORDER BY a.ctg_date, c.BabyID, c.Timestamp
    """

    try:
        con.execute("COPY (" + query + ") TO ? (FORMAT PARQUET)", [str(output_file)])
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CTG reduction stages.")
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "stage1",
            "stage2",
            "stage3",
            "stage4",
            "stage5",
            "stage5_5",
            "stage6",
            "partition",
        ],
        required=True,
        help="Which stage to run.",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Input path(s) override for the stage.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory override for the stage.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=DEFAULT_STAGE1_CUTOFF_DATE,
        help="Stage1 cutoff date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Scanner batch size.",
    )
    parser.add_argument(
        "--report-every-batches",
        type=int,
        default=DEFAULT_PARTITION_REPORT_EVERY,
        help="Progress report frequency in batches (0 to disable).",
    )
    parser.add_argument(
        "--stage2-shard-count",
        type=int,
        default=1,
        help="Stage2: split processing into restartable row-group shards.",
    )
    parser.add_argument(
        "--stage2-shard-index",
        type=int,
        default=None,
        help="Stage2: process only a single shard index (0..stage2-shard-count-1).",
    )

    parser.add_argument(
        "--gap-minutes",
        type=int,
        default=DEFAULT_STAGE3_GAP_MINUTES,
        help="Stage3 session gap threshold (minutes).",
    )
    parser.add_argument(
        "--preg-gap-days",
        type=int,
        default=DEFAULT_STAGE3_PREG_GAP_DAYS,
        help="Stage3 pregnancy gap threshold (days).",
    )
    parser.add_argument(
        "--last-hour-minutes",
        type=int,
        default=DEFAULT_STAGE3_LAST_HOUR_MINUTES,
        help="Stage3 window length in minutes.",
    )
    parser.add_argument(
        "--babyid-salt",
        type=str,
        default=DEFAULT_BABYID_SALT,
        help="Salt used for BabyID hashing.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable DuckDB progress bar for stage3/stage4.",
    )

    parser.add_argument(
        "--dup-threshold",
        type=float,
        default=DEFAULT_STAGE4_DUP_THRESHOLD,
        help="Stage4: drop BabyIDs with duplicate rate above this threshold.",
    )

    parser.add_argument(
        "--min-fhr-seconds",
        type=int,
        default=DEFAULT_STAGE5_MIN_FHR_SECONDS,
        help="Stage5: minimum number of non-zero FHR seconds to keep a BabyID.",
    )

    parser.add_argument(
        "--bucket-count",
        type=int,
        default=DEFAULT_STAGE3_BUCKETS,
        help="Stage3: process in buckets (set to 1 to disable).",
    )
    parser.add_argument(
        "--bucket-index",
        type=int,
        default=None,
        help="Stage3: process a single bucket index (0..bucket-count-1).",
    )
    parser.add_argument(
        "--stage3-window-scope",
        type=str,
        choices=list(STAGE3_WINDOW_SCOPES),
        default=DEFAULT_STAGE3_WINDOW_SCOPE,
        help=(
            "Stage3: anchor the final window on the last non-zero FHR of the whole pregnancy "
            "('pregnancy', default) or only of its last session ('final_session', legacy)."
        ),
    )
    parser.add_argument(
        "--stage3-all-sessions-out",
        type=str,
        nargs="?",
        const=DEFAULT_STAGE3_ALL_SESSIONS_DIR,
        default=None,
        help=(
            "Stage3: also write every session of every pregnancy (BabyID, no PatientID) "
            "for self-supervised pretraining. Optional directory argument; defaults to "
            f"{DEFAULT_STAGE3_ALL_SESSIONS_DIR} when given without a value."
        ),
    )
    parser.add_argument(
        "--no-stage3-prebucket",
        action="store_true",
        help=(
            "Stage3: disable the one-pass temporary PatientID pre-bucket step. "
            "This uses less temporary disk space but rescans the input once per bucket."
        ),
    )
    args = parser.parse_args()

    if args.stage == "stage1":
        stage1_timefilter(
            input_dir=(args.input[0] if args.input else DEFAULT_STAGE0_DIR),
            output_dir=args.output or DEFAULT_STAGE1_DIR,
            cutoff_date=args.cutoff_date,
            batch_size=args.batch_size,
            report_every_batches=args.report_every_batches,
        )
        return

    if args.stage == "stage2":
        stage2_columnfilter(
            input_dir=(args.input[0] if args.input else DEFAULT_STAGE1_DIR),
            output_dir=args.output or DEFAULT_STAGE2_DIR,
            batch_size=args.batch_size,
            report_every_batches=args.report_every_batches,
            shard_count=args.stage2_shard_count,
            shard_index=args.stage2_shard_index,
        )
        return

    if args.stage == "stage3":
        stage3_sessionfilter(
            input_dir=(args.input[0] if args.input else DEFAULT_STAGE2_DIR),
            output_file=args.output or DEFAULT_STAGE3_OUTPUT_FILE,
            gap_minutes=args.gap_minutes,
            preg_gap_days=args.preg_gap_days,
            last_hour_minutes=args.last_hour_minutes,
            babyid_salt=args.babyid_salt,
            show_progress=not args.no_progress,
            bucket_count=args.bucket_count,
            bucket_index=args.bucket_index,
            prebucket=not args.no_stage3_prebucket,
            window_scope=args.stage3_window_scope,
            all_sessions_output=args.stage3_all_sessions_out,
        )
        return

    if args.stage == "stage4":
        stage4_duplicatefilter(
            input_dir=(args.input[0] if args.input else DEFAULT_STAGE3_DIR),
            output_file=args.output or DEFAULT_STAGE4_OUTPUT_FILE,
            dup_threshold=args.dup_threshold,
            show_progress=not args.no_progress,
        )
        return

    if args.stage == "stage5":
        stage5_qualityfilter(
            input_dir=(args.input[0] if args.input else DEFAULT_STAGE4_DIR),
            output_file=args.output or DEFAULT_STAGE5_OUTPUT_FILE,
            min_fhr_seconds=args.min_fhr_seconds,
            show_progress=not args.no_progress,
        )
        return

    if args.stage == "stage5_5":
        stage5_5_sort(
            input_file=(args.input[0] if args.input else DEFAULT_STAGE5_OUTPUT_FILE),
            output_file=args.output or DEFAULT_STAGE5_5_OUTPUT_FILE,
            show_progress=not args.no_progress,
        )
        return

    if args.stage == "stage6" or args.stage == "partition":
        stage6_partitioning(
            input_path=(args.input[0] if args.input else DEFAULT_STAGE5_5_OUTPUT_FILE),
            output_dir=args.output or DEFAULT_PARTITION_OUTPUT_DIR,
            batch_size=args.batch_size,
            report_every_batches=args.report_every_batches,
        )
        return


if __name__ == "__main__":
    main()
