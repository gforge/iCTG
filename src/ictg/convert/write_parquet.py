import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import questionary

from .file_reader import JSON_SUFFIX, ZIP_SUFFIX
from .json_decoder import records_from_path
from .pydanticModels import normalize_patient_record


def write_parquet_per_input(
    glob_exprs: list[str],
    out_dir: Path,
    member: Optional[str] = None,
    batch_size: int = 100_000,
) -> None:
    for glob_expr in glob_exprs:
        for path_str in sorted(glob.glob(glob_expr)):
            path = Path(path_str)
            out_path = _get_output_path(path, out_dir)
            if out_path is None:
                print(f"[warn] Skipping file: {path}")
                continue

            print(f"Processing {path} -> {out_path}")

            writer: Optional[pq.ParquetWriter] = None
            batch: List[Dict[str, Any]] = []
            try:
                for rec in records_from_path(path, member):
                    batch.append(normalize_patient_record(rec))
                    if len(batch) >= batch_size:
                        _flush_parquet_batch(
                            batch, out_path, writer_container := {"writer": writer}
                        )
                        writer = writer_container["writer"]
                        batch.clear()
                if batch:
                    _flush_parquet_batch(
                        batch, out_path, writer_container := {"writer": writer}
                    )
                    writer = writer_container["writer"]
            finally:
                if writer is not None:
                    writer.close()


def _get_output_path(path: Path, out_dir: Path) -> Path | None:
    if path.suffix.lower() not in {JSON_SUFFIX, ZIP_SUFFIX} or not path.is_file():
        return None

    out_path = out_dir / (path.stem + ".parquet")
    if out_path.exists():
        should_remove = questionary.confirm(
            f"Output file exists: {out_path}\nRemove and continue?"
        ).ask()
        if should_remove:
            out_path.unlink()
            print(f"Removed existing file: {out_path}")
        else:
            return None

    return out_path


def _flush_parquet_batch(
    batch: List[Dict[str, Any]], out_path: Path, writer_container: Dict[str, Any]
) -> None:

    df = pd.DataFrame(batch)
    table = pa.Table.from_pandas(df, preserve_index=False)
    writer = writer_container.get("writer")
    if writer is None:
        writer = pq.ParquetWriter(out_path.as_posix(), table.schema, compression="zstd")
        writer_container["writer"] = writer
    writer.write_table(table)
