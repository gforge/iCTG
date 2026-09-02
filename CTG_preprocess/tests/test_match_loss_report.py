"""End-to-end test of ``match_loss_report`` on tiny synthetic stage outputs and a registry CSV.

Every category (and every ``registry_row_excluded`` sub-reason) is covered at least once.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from match_loss_report import (
    CATEGORIES,
    OFFSET_BUCKETS,
    MatchLossReport,
    compute_match_loss_report,
    main,
    offset_bucket,
)

# Every gravniva column referenced by the shared Stage 7 registry SQL. Values not relevant for
# matching are left empty.
REGISTRY_COLUMNS = [
    "glopnr",
    "personnummer_mor",
    "forlossningsdatum_fv1",
    "forlossningstid_fv1",
    "etablerade_varkar_datum",
    "etablerade_varkar_tid",
    "forlossningsstart_basta_skattning",
    "forlossningsslut_basta_skattning",
    "apgar_1_min",
    "apgar_5_min",
    "apgar_10_min",
    "gl_v_barn",
    "gl_d_barn",
    "fodelseland",
    "utbildningsniva",
    "para_mhv1",
    "langd_inskrivning_cm",
    "bmi_inskrivning",
    "tidigare_sectio",
    "tobak_3_manader_fore_graviditet",
    "tobak_inskrivning",
    "tobak_vecka_30_32",
    "diabetes_mellitus",
    "kon",
    "alkohol_audit_poang",
    "ph_navelartar",
    "ph_navelven",
    "avled_datum",
    "moderns_diagnoser_rad",
    "moderns_atgarder_rad",
    "barnets_diagnoser_rad",
    "barnets_atgarder_rad",
    "ventilation_pa_mask_min",
    "intubation_min",
    "hjartmassage_min",
]

# (personnummer_mor, forlossningsdatum_fv1, apgar_5_min) -- one tuple per registry birth row.
REGISTRY_ROWS: list[tuple[str, str, str]] = [
    # short_personnummer (10 digits). NOTE: the shared Stage 7 SQL derives the mother's birth
    # date with a non-try strptime over the first 8 digits, so the prefix must still parse as
    # a date ("80010112" here); e.g. "800101-0001" ("80010100", day 00) would raise.
    ("800101-1234", "2020-01-01", "9"),
    ("", "2021-01-02", "9"),  # short_personnummer (no personnummer at all)
    ("19800101-0002", "2020-03-01", ""),  # missing_apgar5
    ("19800101-0003", "", "9"),  # missing_birth_day
    ("19800101-0004", "2020-04-01", "9"),  # no_ctg_for_patient
    ("19800101-0005", "2020-05-10", "9"),  # ctg_only_outside_window, nearest -5 days
    ("19800101-0006", "2021-06-10", "9"),  # ctg_only_outside_window, nearest +10 (vs -40)
    ("19800101-0007", "2020-07-15", "9"),  # dropped_stage4_duplicates
    ("19800101-0008", "2021-08-20", "9"),  # dropped_stage5_short_signal
    ("19800101-0009", "2020-09-09", "9"),  # multiple_ctg_matches
    ("19800101-0010", "2021-10-10", "9"),  # ctg_shared_by_multiple_registry_rows (twin A)
    ("19800101-0010", "2021-10-10", "7"),  # ctg_shared_by_multiple_registry_rows (twin B)
    ("19800101-0011", "2021-11-11", "9"),  # matched
]

# (BabyID, PatientID, window end) -- the Stage 3 pregnancies. ctg_date = date(window end).
STAGE3_BABIES: list[tuple[str, str, datetime]] = [
    ("bid_out1", "19800101-0005", datetime(2020, 5, 5, 12, 0)),
    ("bid_out2_after", "19800101-0006", datetime(2021, 6, 20, 12, 0)),
    ("bid_out2_before", "19800101-0006", datetime(2021, 5, 1, 12, 0)),
    ("bid_s4drop", "19800101-0007", datetime(2020, 7, 15, 12, 0)),
    ("bid_s5drop", "19800101-0008", datetime(2021, 8, 19, 12, 0)),
    ("bid_multi_a", "19800101-0009", datetime(2020, 9, 9, 12, 0)),
    ("bid_multi_b", "19800101-0009", datetime(2020, 9, 8, 12, 0)),
    ("bid_twins", "19800101-0010", datetime(2021, 10, 10, 12, 0)),
    ("bid_ok", "19800101-0011", datetime(2021, 11, 11, 12, 0)),
    ("bid_ok_old", "19800101-0011", datetime(2021, 1, 1, 12, 0)),
]
STAGE4_DROPPED = {"bid_s4drop"}
STAGE5_DROPPED = {"bid_s5drop"}


def _write_registry_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, delimiter=";")
        writer.writeheader()
        for index, (pnr, birth_day, apgar5) in enumerate(REGISTRY_ROWS):
            row = dict.fromkeys(REGISTRY_COLUMNS, "")
            row.update(
                glopnr=str(index + 1),
                personnummer_mor=pnr,
                forlossningsdatum_fv1=birth_day,
                apgar_5_min=apgar5,
                kon="Pojke",
            )
            writer.writerow(row)


def _ctg_table(
    babies: list[tuple[str, str, datetime]], with_ctg_date: bool, seconds: int = 3
) -> pa.Table:
    rows = [
        (baby, patient, end - timedelta(seconds=seconds - 1 - i))
        for baby, patient, end in babies
        for i in range(seconds)
    ]
    columns: dict[str, pa.Array] = {
        "BabyID": pa.array([r[0] for r in rows], type=pa.string()),
        "PatientID": pa.array([r[1] for r in rows], type=pa.string()),
        "Timestamp": pa.array([r[2] for r in rows], type=pa.timestamp("us")),
        "FHR": pa.array([140.0] * len(rows), type=pa.float32()),
    }
    if with_ctg_date:
        columns["ctg_date"] = pa.array([r[2].date() for r in rows], type=pa.date32())
    return pa.table(columns)


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    registry = tmp_path / "gravniva.csv"
    _write_registry_csv(registry)

    # Stage 3 is a directory of bucket files; split the babies across two files.
    stage3_dir = tmp_path / "stage_3_sessionfilter"
    stage3_dir.mkdir()
    half = len(STAGE3_BABIES) // 2
    pq.write_table(
        _ctg_table(STAGE3_BABIES[:half], with_ctg_date=False),
        stage3_dir / "stage3_sessions_bucket_0000.parquet",
    )
    pq.write_table(
        _ctg_table(STAGE3_BABIES[half:], with_ctg_date=False),
        stage3_dir / "stage3_sessions_bucket_0001.parquet",
    )

    stage4_dir = tmp_path / "stage_4_duplicatefilter"
    stage4_dir.mkdir()
    stage4_babies = [b for b in STAGE3_BABIES if b[0] not in STAGE4_DROPPED]
    pq.write_table(
        _ctg_table(stage4_babies, with_ctg_date=False), stage4_dir / "stage4_dedup_0000.parquet"
    )

    stage5_5 = tmp_path / "stage5_5_sorted.parquet"
    stage5_5_babies = [b for b in stage4_babies if b[0] not in STAGE5_DROPPED]
    pq.write_table(_ctg_table(stage5_5_babies, with_ctg_date=True), stage5_5)

    return {"registry": registry, "stage3": stage3_dir, "stage4": stage4_dir, "stage5_5": stage5_5}


@pytest.fixture
def report(synthetic_inputs: dict[str, Path]) -> MatchLossReport:
    return compute_match_loss_report(
        registry_csv=synthetic_inputs["registry"],
        stage3_path=synthetic_inputs["stage3"],
        stage4_path=synthetic_inputs["stage4"],
        stage5_5_path=synthetic_inputs["stage5_5"],
        show_progress=False,
    )


def test_every_registry_row_gets_exactly_one_category(report: MatchLossReport) -> None:
    assert report.registry_rows == len(REGISTRY_ROWS)
    assert set(report.category_counts) <= set(CATEGORIES)
    assert sum(report.category_counts.values()) == report.registry_rows
    assert report.category_counts == {
        "registry_row_excluded": 4,
        "no_ctg_for_patient": 1,
        "ctg_only_outside_window": 2,
        "dropped_stage4_duplicates": 1,
        "dropped_stage5_short_signal": 1,
        "multiple_ctg_matches": 1,
        "ctg_shared_by_multiple_registry_rows": 2,
        "matched": 1,
    }
    # ``matched`` reconciles with the Stage 7 uniqueness SQL replayed on Stage 5.5.
    assert report.stage7_unique_matches == report.category_counts["matched"] == 1


def test_excluded_sub_reasons_partition_the_excluded_rows(report: MatchLossReport) -> None:
    assert report.sub_reason_counts == {
        "short_personnummer": 2,
        "missing_apgar5": 1,
        "missing_birth_day": 1,
    }
    assert sum(report.sub_reason_counts.values()) == report.category_counts["registry_row_excluded"]


def test_nearest_offset_is_signed_and_closest_pregnancy_wins(report: MatchLossReport) -> None:
    # -5: CTG five days before birth; +10: the +10 day pregnancy beats the -40 day one.
    assert report.offset_counts == {-5: 1, 10: 1}
    buckets = report.offset_bucket_counts()
    assert buckets["-7..-2"] == 1
    assert buckets["+8..+30"] == 1
    assert sum(buckets.values()) == report.category_counts["ctg_only_outside_window"]
    assert list(buckets) == [label for label, _, _ in OFFSET_BUCKETS]


def test_year_table_covers_all_rows_including_unknown_birth_day(report: MatchLossReport) -> None:
    assert set(report.year_counts) == {"2020", "2021", "unknown"}
    assert sum(sum(per_cat.values()) for per_cat in report.year_counts.values()) == 13
    assert sum(report.year_counts["2020"].values()) == 6
    assert sum(report.year_counts["2021"].values()) == 6
    assert report.year_counts["unknown"] == {"registry_row_excluded": 1}
    assert report.year_counts["2021"]["matched"] == 1
    assert report.year_counts["2021"]["ctg_shared_by_multiple_registry_rows"] == 2


def test_markdown_has_counts_but_no_identifiers(report: MatchLossReport) -> None:
    markdown = report.to_markdown()
    for category in CATEGORIES:
        assert f"| {category} | " in markdown
    assert "| Total | | 13 | 100.00 |" in markdown
    assert "| -7..-2 | 1 |" in markdown
    assert "19800101" not in markdown
    assert "800101-1234" not in markdown
    for baby, _, _ in STAGE3_BABIES:
        assert baby not in markdown


def test_cli_writes_markdown_to_out(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "reports" / "match_loss.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "match_loss_report.py",
            "--registry-csv",
            str(synthetic_inputs["registry"]),
            "--stage3",
            str(synthetic_inputs["stage3"]),
            "--stage4",
            str(synthetic_inputs["stage4"]),
            "--stage5-5",
            str(synthetic_inputs["stage5_5"]),
            "--out",
            str(out),
            "--no-progress",
        ],
    )
    main()
    captured = capsys.readouterr()
    assert captured.out.startswith("# Registry match loss report")
    assert out.read_text(encoding="utf-8").strip() == captured.out.strip()


@pytest.mark.parametrize(
    ("offset_days", "label"),
    [
        (-1000, "< -365"),
        (-366, "< -365"),
        (-365, "-365..-31"),
        (-31, "-365..-31"),
        (-30, "-30..-8"),
        (-8, "-30..-8"),
        (-7, "-7..-2"),
        (-2, "-7..-2"),
        (1, "+1"),
        (2, "+2..+7"),
        (7, "+2..+7"),
        (8, "+8..+30"),
        (30, "+8..+30"),
        (31, "> +30"),
        (400, "> +30"),
        # Inside the Stage 7 window; can never be "nearest outside" but must not crash.
        (0, "in_window (unexpected)"),
        (-1, "in_window (unexpected)"),
    ],
)
def test_offset_bucket_boundaries(offset_days: int, label: str) -> None:
    assert offset_bucket(offset_days) == label


def test_offset_bucket_counts_aggregates_per_bucket() -> None:
    report = MatchLossReport(
        registry_rows=0,
        category_counts={},
        sub_reason_counts={},
        offset_counts={-400: 2, -100: 1, -3: 4, 1: 1, 5: 2, 20: 3, 90: 1},
    )
    assert report.offset_bucket_counts() == {
        "< -365": 2,
        "-365..-31": 1,
        "-30..-8": 0,
        "-7..-2": 4,
        "+1": 1,
        "+2..+7": 2,
        "+8..+30": 3,
        "> +30": 1,
    }
