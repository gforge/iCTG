"""The vectorised toco decode must be byte-for-byte equivalent to the per-cell reference."""

from __future__ import annotations

import base64
import random

import pyarrow as pa
import pytest
from ctg_reduction import _compute_toco, _decode_toco_fast, _fixed_width_string_bytes


def _reference(values: list[str | None]) -> list[float]:
    out: list[float] = []
    for encoded in values:
        if encoded is None:
            out.append(0.0)
            continue
        try:
            decoded = base64.b64decode(encoded)
        except Exception:
            out.append(0.0)
            continue
        valid = [v for v in decoded if 1 <= v <= 99]
        if valid:
            out.append(sum(valid) / len(valid))
        elif decoded:
            out.append(sum(decoded) / len(decoded))
        else:
            out.append(0.0)
    return out


def _random_values(seed: int, n: int) -> list[str | None]:
    rng = random.Random(seed)
    values: list[str | None] = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.85:  # the real export: 4 bytes -> 8 chars with '=='
            values.append(base64.b64encode(bytes(rng.randrange(256) for _ in range(4))).decode())
        elif roll < 0.90:
            values.append(None)
        elif roll < 0.93:
            values.append(base64.b64encode(bytes(rng.randrange(256) for _ in range(6))).decode())
        elif roll < 0.96:
            values.append("!!!!!!==")  # 8 chars, invalid alphabet -> slow path
        else:
            values.append("")
    return values


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_toco_matches_reference_on_mixed_input(seed: int) -> None:
    values = _random_values(seed, 3000)
    batch = pa.RecordBatch.from_pydict({"Toco_Values": pa.array(values, type=pa.string())})
    got = _compute_toco(batch).to_pylist()
    assert got == pytest.approx(_reference(values), abs=1e-4)


def test_decode_toco_fast_round_trips_and_flags_bad_rows() -> None:
    raw = [bytes([0, 1, 99, 255]), bytes([40, 41, 42, 43])]
    encoded = [base64.b64encode(b).decode() for b in raw] + ["!!!!!!=="]
    decoded = _decode_toco_fast(_fixed_width_string_bytes(pa.array(encoded), 8))
    assert decoded[0].tolist() == [0, 1, 99, 255]
    assert decoded[1].tolist() == [40, 41, 42, 43]
    assert decoded[2].tolist() == [255, 255, 255, 255]


def test_fixed_width_string_bytes_reads_arrow_buffer_and_handles_slices() -> None:
    arr = pa.array(["AAAAAAA=", "BBBBBBB=", "CCCCCCC="])
    assert _fixed_width_string_bytes(arr, 8).tobytes() == b"AAAAAAA=BBBBBBB=CCCCCCC="
    # A sliced array has a non-zero offset; the result must still be the sliced values.
    assert _fixed_width_string_bytes(arr.slice(1), 8).tobytes() == b"BBBBBBB=CCCCCCC="
    # Multi-byte characters break the fixed width -> fallback path with replacement.
    odd = pa.array(["ÅÅÅÅÅÅ=="])
    assert len(_fixed_width_string_bytes(odd, 8)) == 8


def test_compute_toco_without_column_is_zero() -> None:
    batch = pa.RecordBatch.from_pydict({"other": pa.array([1, 2])})
    assert _compute_toco(batch).to_pylist() == [0.0, 0.0]
