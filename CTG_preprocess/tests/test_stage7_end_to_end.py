"""End-to-end Stage 7 on synthetic inputs: gravniva + long tables + SNQ + stage 5.5/6 parquet.

Checks the anonymized outputs (no identifiers), the derived outcome variables and the
anonymized long-table exports. All data here is invented.
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from registry_matching import (
    IDENTIFYING_COLUMNS,
    SNQ_VARIABLES,
    _leading_int_expr,
    _snq_expr,
    registry_match,
)

# Synthetic mothers: 12-digit "personnummer" (YYYYMMDDNNNN) -> PatientID YYYYMMDD-NNNN.
MOTHER_A = "198001010001"
MOTHER_B = "199002020002"
MOTHER_C = "198503030003"
BIRTH_A = "2019-03-10"
BIRTH_B = "2020-07-01"
BIRTH_C = "2021-01-15"


def _gravniva_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "glopnr": "",
        "personnummer_mor": "",
        "forlossningsdatum_fv1": "",
        "forlossningstid_fv1": "43200",  # 12:00
        "etablerade_varkar_datum": "",
        "etablerade_varkar_tid": "",
        "varkar_borjade_datum": "",
        "varkar_borjade_tid": "",
        "vattenavgang_datum": "",
        "vattenavgang_tid": "",
        "amniotomi_datum": "",
        "amniotomi_tid": "",
        "krystvarkar_datum": "",
        "krystvarkar_tid": "",
        "sectio_start_datum": "",
        "sectio_start_tid": "",
        "sectio_slut_datum": "",
        "sectio_slut_tid": "",
        "forlossningsstart_basta_skattning": "Spontan start",
        "forlossningsslut_basta_skattning": "Vaginalt, ej instrumentellt",
        "indikation": "",
        "oxytocin_under_forlossning": "",
        "smartlindring_epidural": "",
        "presentation": "Framstupa kronbjudning",
        "robsongrupp": "1",
        "total_blodning_ml": "300",
        "ctg_intagningstest": "Normal",
        "ivf_graviditet": "Nej",
        "kronisk_hypertoni": "Nej",
        "diagnosen_graviditetsdiabetes_stalld": "Nej",
        "apgar_1_min": "9",
        "apgar_5_min": "9",
        "apgar_10_min": "10",
        "gl_v_barn": "40",
        "gl_d_barn": "0",
        "fodelseland": "Sverige",
        "utbildningsniva": "Universitet",
        "para_mhv1": "0",
        "langd_inskrivning_cm": "165",
        "bmi_inskrivning": "22.5",
        "tidigare_sectio": "Nej",
        "tobak_3_manader_fore_graviditet": "Nej",
        "tobak_inskrivning": "Nej",
        "tobak_vecka_30_32": "Nej",
        "diabetes_mellitus": "Nej",
        "kon": "Pojke",
        "alkohol_audit_poang": "0",
        "ph_navelartar": "7.25",
        "ph_navelven": "7.30",
        "be_navelartar_mmol_l": "-4",
        "be_navelven_mmol_l": "-3",
        "pco2_navelartar_kpa": "7.0",
        "po2_navelartar_kpa": "2.5",
        "fodelsevikt_g": "3500",
        "vikt_avvikelse_perc": "1.0",
        "ventilation_pa_mask_min": "",
        "intubation_min": "",
        "hjartmassage_min": "",
        "acidoskorrektion": "Nej",
        "utskriven_till_hemmet": "Ja",
        "utskrivning_datum": "",
        "avled_datum": "",
        "moderns_diagnoser_rad": "",
        "moderns_atgarder_rad": "",
        "barnets_diagnoser_rad": "",
        "barnets_atgarder_rad": "",
    }
    row.update(overrides)
    return row


def _write_semicolon_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _snq_row(glopnr: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "BarnID": f"b{glopnr}",
        "glopnr": glopnr,
        "ICD_kod": "",
        "KVÅ_kod": "",
        "Högst HIE": "",
        "HIE": "Nej",
    }
    for _name, col, kind, _agg in SNQ_VARIABLES:
        if col not in row:
            row[col] = "Nej" if kind in {"ja_nej"} else ""
    row.update(overrides)
    return row


@pytest.fixture
def stage7_outputs(tmp_path: Path) -> dict[str, Path]:
    # --- gravniva: A = severe outcome, B = uncomplicated, C = no CTG on the birth day ----
    grav = [
        _gravniva_row(
            glopnr="1001",
            personnummer_mor=MOTHER_A,
            forlossningsdatum_fv1=BIRTH_A,
            forlossningstid_fv1="43200",
            etablerade_varkar_datum=BIRTH_A,
            etablerade_varkar_tid="21600",  # 06:00 -> 6 h before birth
            krystvarkar_datum=BIRTH_A,
            krystvarkar_tid="39600",  # 11:00 -> 1 h before birth
            sectio_start_datum=BIRTH_A,
            sectio_start_tid="41400",  # 11:30
            sectio_slut_datum=BIRTH_A,
            sectio_slut_tid="45000",  # 12:30 -> 1 h duration
            forlossningsslut_basta_skattning="Akut kejsarsnitt",
            indikation="Urakut",
            ctg_intagningstest="Ej normal",
            apgar_5_min="4",
            apgar_10_min="6",
            ph_navelartar="7.01",
            be_navelartar_mmol_l="-14",
            intubation_min="3",
            utskrivning_datum="2019-03-15",
            moderns_diagnoser_rad="O680, O14.1",
            barnets_diagnoser_rad="P21.0, P916",
            barnets_atgarder_rad="DV034",
        ),
        _gravniva_row(
            glopnr="1002",
            personnummer_mor=MOTHER_B,
            forlossningsdatum_fv1=BIRTH_B,
            diagnosen_graviditetsdiabetes_stalld="Ja, läkemedelsbehandlad",
            moderns_diagnoser_rad="O45.9",
        ),
        _gravniva_row(
            glopnr="1003",
            personnummer_mor=MOTHER_C,
            forlossningsdatum_fv1=BIRTH_C,
        ),
    ]
    grav_path = tmp_path / "gravniva.csv"
    _write_semicolon_csv(grav_path, grav)

    # --- long tables: dated codes; B gets a child diagnosis only here (not in *_rad) ------
    mother_long: list[dict[str, object]] = [
        {
            "glopnr": "1001",
            "startdatum_moderns_diagnoser": BIRTH_A,
            "diagnoskod_moderns_diagnoser": "O680",
        },
        {
            "glopnr": "1001",
            "startdatum_moderns_diagnoser": "2019-01-10",
            "diagnoskod_moderns_diagnoser": "O24.4",
        },
    ]
    child_long: list[dict[str, object]] = [
        {
            "glopnr": "1002",
            "bordnr": "1",
            "startdatum_barnets_diagnoser": "2020-07-03",
            "diagnoskod_barnets_diagnoser": "P22.1",
        },
    ]
    proc_long: list[dict[str, object]] = [
        {
            "glopnr": "1001",
            "bordnr": "1",
            "startdatum_barnets_atgarder": BIRTH_A,
            "atgardskod_barnets_atgarder": "DV034",
        },
    ]
    _write_semicolon_csv(tmp_path / "mdiag.csv", mother_long)
    _write_semicolon_csv(tmp_path / "cdiag.csv", child_long)
    _write_semicolon_csv(tmp_path / "cproc.csv", proc_long)

    # --- SNQ: only A admitted --------------------------------------------------------------
    snq = [
        _snq_row(
            "1001",
            **{
                "HIE": "Ja",
                "Högst HIE": "2",
                "ICD_kod": "P916;P90.0",
                "KVÅ_kod": "DG021",
                "Behandlad med hypotermi": "Ja",
                "Kramper": "Ja",
                "HLR-åtgärder": "2 Ja (>= 10 min)",
                "Högst IVH": "88 Ej undersökt",
                "Tidig bakt. sepsis, odlingsverif. (antal)": "1",
                "Vårdtid, neonatologi": "9",
                "Artär pH": "7,01",
                "Avliden enl. SNQ": "Nej",
                "Avliden enl. DOR": "",
            },
        )
    ]
    pd.DataFrame(snq).to_csv(tmp_path / "snq.csv", index=False)

    # --- CTG side: stage 5.5 map and stage 6 signal ------------------------------------------
    s55 = pa.table(
        {
            "BabyID": ["babyA", "babyB", "babyC"],
            "PatientID": [
                f"{MOTHER_A[:8]}-{MOTHER_A[8:]}",
                f"{MOTHER_B[:8]}-{MOTHER_B[8:]}",
                f"{MOTHER_C[:8]}-{MOTHER_C[8:]}",
            ],
            "ctg_date": pa.array(
                [
                    pd.Timestamp(BIRTH_A).date(),
                    pd.Timestamp("2020-06-30").date(),
                    pd.Timestamp("2021-01-01").date(),
                ],
                pa.date32(),
            ),
        }
    )
    pq.write_table(s55, tmp_path / "stage5_5.parquet")
    stage6 = tmp_path / "stage6"
    stage6.mkdir()
    pq.write_table(
        pa.table(
            {
                "BabyID": ["babyA"] * 3 + ["babyB"] * 2 + ["babyC"],
                "Timestamp": pa.array(
                    pd.to_datetime(
                        ["2019-03-10 10:00"] * 3 + ["2020-06-30 23:00"] * 2 + ["2021-01-01 01:00"]
                    )
                ),
                "FHR": [140.0, 141.0, 139.0, 150.0, 151.0, 130.0],
                "toco": [10.0, 12.0, 11.0, 20.0, 21.0, 5.0],
                "Hr1_SignalQuality": [3, 3, 3, 3, 3, 3],
            }
        ),
        stage6 / "part.parquet",
    )

    out = tmp_path / "out"
    registry_match(
        registry_csv=grav_path,
        snq_file=tmp_path / "snq.csv",
        stage5_5_file=tmp_path / "stage5_5.parquet",
        stage6_dir=stage6,
        registry_out=out / "registry.csv",
        ctg_out=out / "ctg_final.parquet",
        mother_diag_csv=tmp_path / "mdiag.csv",
        child_diag_csv=tmp_path / "cdiag.csv",
        child_proc_csv=tmp_path / "cproc.csv",
        mother_diag_out=out / "mother_diagnoses.csv",
        child_diag_out=out / "child_diagnoses.csv",
        child_proc_out=out / "child_procedures.csv",
        show_progress=False,
    )
    return {
        "registry": out / "registry.csv",
        "ctg": out / "ctg_final.parquet",
        "mother": out / "mother_diagnoses.csv",
        "child": out / "child_diagnoses.csv",
        "proc": out / "child_procedures.csv",
    }


def test_outputs_are_anonymized_and_matched_by_birth_day(stage7_outputs: dict[str, Path]) -> None:
    reg = pd.read_csv(stage7_outputs["registry"])
    assert list(reg["BabyID"]) == ["babyA", "babyB"]  # C's CTG is two weeks before birth
    forbidden = set(IDENTIFYING_COLUMNS) | {"personnummer_mor", "reg_digits"}
    assert not forbidden & set(reg.columns)
    assert reg.columns[0] == "BabyID"
    ctg = pq.read_table(stage7_outputs["ctg"]).to_pandas()
    assert set(ctg["BabyID"]) == {"babyA", "babyB"}
    assert "PatientID" not in ctg.columns


def test_severe_outcome_and_derived_variables(stage7_outputs: dict[str, Path]) -> None:
    reg = pd.read_csv(stage7_outputs["registry"]).set_index("BabyID")
    a = reg.loc["babyA"]
    b = reg.loc["babyB"]

    # gravniva-derived
    assert bool(a["emergency_c_section"]) and not bool(b["emergency_c_section"])
    assert a["c_section_urgency"] == "Urakut"
    assert bool(a["ctg_admission_test_abnormal"]) and not bool(b["ctg_admission_test_abnormal"])
    assert a["etablerade_varkar_seconds"] == 6 * 3600
    assert a["second_stage_seconds_before_birth"] == 3600
    assert a["c_section_start_seconds_before_birth"] == 1800
    assert a["c_section_duration_seconds"] == 3600
    assert a["days_to_discharge"] == 5
    assert bool(a["apgar5_below7"]) and bool(a["apgar10_below7"])
    assert bool(a["metabolic_acidosis"]) and not bool(b["metabolic_acidosis"])
    assert bool(a["ph_navel_below705"]) and not bool(a["ph_navel_below7"])
    assert a["intubation_min"] == 3
    assert bool(b["graviditetsdiabetes_diagnos"]) and not bool(a["graviditetsdiabetes_diagnos"])

    # ICD flags from the collapsed columns and the long tables combined
    assert bool(a["fetal_distress_in_labour"]) and bool(a["preeclampsia"])
    assert bool(a["gestational_or_pregestational_diabetes"])  # O24.4 only in the long table
    assert bool(b["placental_abruption"]) and not bool(a["placental_abruption"])  # O45, not O711
    assert bool(a["severe_birth_asphyxia"]) and bool(a["birth_asphyxia_any"])
    assert bool(b["respiratory_distress_newborn"])  # P22.1 only in the long table
    assert not bool(b["birth_asphyxia_any"])

    # SNQ
    assert bool(a["neonatal_care_admission"]) and not bool(b["neonatal_care_admission"])
    assert a["highest_hie"] == 2 and bool(a["hie"])
    assert bool(a["hie_icd"]) and bool(a["neonatal_convulsions"])
    assert bool(a["snq_hypothermia_treatment"]) and bool(a["snq_seizures"])
    assert bool(a["snq_resuscitation"]) and bool(a["snq_resuscitation_over_10min"])
    assert pd.isna(a["snq_highest_ivh"])  # "88 Ej undersökt" -> NULL
    assert bool(a["snq_early_culture_verified_sepsis"])
    assert a["snq_care_days_neonatal"] == 9
    assert a["snq_ph_navelartar"] == 7.01  # decimal comma parsed
    assert bool(a["respiratorbehandling"])
    assert pd.isna(b["snq_hypothermia_treatment"])  # not in SNQ -> NULL, never False

    # composite
    assert bool(a["severe_neonatal_outcome"]) and not bool(b["severe_neonatal_outcome"])


def test_long_tables_are_exported_with_day_offsets(stage7_outputs: dict[str, Path]) -> None:
    mother = pd.read_csv(stage7_outputs["mother"])
    assert list(mother.columns) == ["BabyID", "day_offset", "code"]
    assert set(mother["BabyID"]) == {"babyA"}
    assert sorted(mother["day_offset"]) == [-59, 0]
    child = pd.read_csv(stage7_outputs["child"])
    assert child.to_dict("records") == [{"BabyID": "babyB", "day_offset": 2, "code": "P22.1"}]
    proc = pd.read_csv(stage7_outputs["proc"])
    assert proc.to_dict("records") == [{"BabyID": "babyA", "day_offset": 0, "code": "DV034"}]


def test_snq_category_expressions() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE t (i INTEGER, v VARCHAR)")
    values = ["0 Ingen IVH", "3 Grad 3", "88 Ej undersökt", "us", None]
    con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(values)))
    assert [
        r[0] for r in con.execute(f"SELECT {_leading_int_expr('v')} FROM t ORDER BY i").fetchall()
    ] == [
        0,
        3,
        88,
        None,
        None,
    ]
    assert [
        r[0]
        for r in con.execute(f"SELECT {_snq_expr('ivh_grade', 'v')} FROM t ORDER BY i").fetchall()
    ] == [
        0,
        3,
        None,
        None,
        None,
    ]
    con.execute("DELETE FROM t")
    values = [
        "0 Nej (vitalt barn)",
        "1 Ja (< 10 min)",
        "2 Ja (>= 10 min)",
        "3 Nej (palliation)",
        "us",
    ]
    con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(values)))
    assert [
        r[0] for r in con.execute(f"SELECT {_snq_expr('hlr', 'v')} FROM t ORDER BY i").fetchall()
    ] == [
        False,
        True,
        True,
        None,
        None,
    ]
    assert [
        r[0]
        for r in con.execute(f"SELECT {_snq_expr('hlr_over_10', 'v')} FROM t ORDER BY i").fetchall()
    ] == [
        False,
        False,
        True,
        None,
        None,
    ]
