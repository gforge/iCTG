"""Stage 7: match the reduced CTG cohort to the registry export and write anonymized outputs.

Inputs (see ``config.py``):

* ``gravniva.csv``            one row per live-born singleton (Swedish Pregnancy Register, SPR);
* the SPR long tables         mother diagnoses, child diagnoses and child procedures during the
                              first 28 days, one dated row per code (optional, enrich the flags
                              and are exported anonymized);
* the SNQ export              one row per child admitted to neonatal care (optional columns are
                              tolerated: a missing SNQ column yields NULL and a warning).

Outputs: ``registry.csv`` (one row per matched BabyID; no PatientID/glopnr), ``ctg_final.parquet``,
and the anonymized long tables ``mother_diagnoses.csv`` / ``child_diagnoses.csv`` /
``child_procedures.csv`` (BabyID, day_offset relative to birth, code).

All variables are documented in ``stage7_registry_data_dictionary.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from config import (
    DEFAULT_CHILD_DIAG_CSV,
    DEFAULT_CHILD_PROC_CSV,
    DEFAULT_MOTHER_DIAG_CSV,
    DEFAULT_PATIENT_CSV,
    DEFAULT_SNQ_FILE,
    DEFAULT_STAGE2_EXTRA_COLUMNS,
    DEFAULT_STAGE5_5_OUTPUT_FILE,
    DEFAULT_STAGE6_DIR,
    DEFAULT_STAGE7_CHILD_DIAG_CSV,
    DEFAULT_STAGE7_CHILD_PROC_CSV,
    DEFAULT_STAGE7_CTG_PARQUET,
    DEFAULT_STAGE7_MOTHER_DIAG_CSV,
    DEFAULT_STAGE7_REGISTRY_CSV,
)
from pseudonyms import mother_id_sql
from secrets_store import get_secret


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Stage 7 match uniqueness. A registry birth row must map to exactly one BabyID and a
# BabyID must map to exactly one registry row; twins/multiples share the mother's CTG and
# are therefore excluded rather than duplicated (they need their own handling).
MULTI_BABY_REGISTRY_ROWS_SQL = """
SELECT COUNT(*) FROM (
    SELECT reg_row
    FROM matches
    GROUP BY reg_row
    HAVING COUNT(*) > 1
)
"""

MULTI_REGISTRY_BABIES_SQL = """
SELECT COUNT(*) FROM (
    SELECT BabyID
    FROM matches
    GROUP BY BabyID
    HAVING COUNT(*) > 1
)
"""

UNIQUE_MATCHES_SQL = """
WITH per_row AS (
    SELECT reg_row, COUNT(*) AS cnt
    FROM matches
    GROUP BY reg_row
),
per_baby AS (
    SELECT BabyID, COUNT(*) AS cnt
    FROM matches
    GROUP BY BabyID
)
SELECT m.*
FROM matches m
JOIN per_row r USING (reg_row)
JOIN per_baby b USING (BabyID)
WHERE r.cnt = 1 AND b.cnt = 1
"""

# Columns that must never leave Stage 7 (identifiers / linkage keys).
IDENTIFYING_COLUMNS = ("reg_row", "PatientID", "glopnr", "reg_digits", "ctg_date")


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Run a single-value aggregate query (e.g. ``COUNT(*)``) and return it as an int."""
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no rows: {sql.strip()[:200]}")
    return int(row[0] or 0)


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return (
        _count(
            con,
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'",
        )
        > 0
    )


def _columns(con: duckdb.DuckDBPyConnection, relation: str) -> set[str]:
    return {row[0] for row in con.execute(f"DESCRIBE {relation}").fetchall()}


def _safe_path(path: str | Path) -> str:
    return str(path).replace("'", "''")


# ---------------------------------------------------------------------------------------------
# SQL expression builders (each takes a column name or expression and returns SQL text)
# ---------------------------------------------------------------------------------------------


def _clean_text_expr(col: str) -> str:
    return f"NULLIF(trim(CAST({col} AS VARCHAR)), '')"


def _int_expr(col: str) -> str:
    return f"TRY_CAST(REPLACE({_clean_text_expr(col)}, ',', '.') AS INTEGER)"


def _float_expr(col: str) -> str:
    return f"TRY_CAST(REPLACE({_clean_text_expr(col)}, ',', '.') AS DOUBLE)"


def _date_expr(col: str) -> str:
    return f"TRY_CAST({_clean_text_expr(col)} AS DATE)"


def _timestamp_from_date_and_seconds(date_expr: str, seconds_expr: str) -> str:
    return (
        "CASE "
        f"WHEN {date_expr} IS NOT NULL AND {seconds_expr} IS NOT NULL "
        f"THEN CAST({date_expr} AS TIMESTAMP) + ({seconds_expr} * INTERVAL 1 SECOND) "
        "ELSE NULL END"
    )


def _bool_ja_nej_expr(col: str) -> str:
    """``Ja``/``Nej`` (case-insensitive, surrounding whitespace ignored) to boolean, else NULL.

    Values such as ``us`` (uppgift saknas) or ``Vet ej`` therefore become NULL.
    """
    clean = _clean_text_expr(col)
    return (
        "CASE "
        f"WHEN lower({clean}) = 'ja' THEN TRUE "
        f"WHEN lower({clean}) = 'nej' THEN FALSE "
        "ELSE NULL END"
    )


def _bool_ja_prefix_expr(col: str) -> str:
    """Like ``_bool_ja_nej_expr`` but any value starting with ``Ja`` counts as TRUE
    (``Ja, läkemedelsbehandlad``)."""
    clean = _clean_text_expr(col)
    return (
        "CASE "
        f"WHEN lower({clean}) LIKE 'ja%' THEN TRUE "
        f"WHEN lower({clean}) = 'nej' THEN FALSE "
        "ELSE NULL END"
    )


def _text_equals_expr(col: str, value: str) -> str:
    """TRUE/FALSE when the cleaned text equals/does not equal ``value``; NULL when missing."""
    clean = _clean_text_expr(col)
    safe = value.replace("'", "''")
    return f"CASE WHEN {clean} IS NULL THEN NULL ELSE ({clean} = '{safe}') END"


def _leading_int_expr(col: str) -> str:
    """Integer code at the start of a coded SNQ category (``'2 Grad 2'`` -> 2); NULL otherwise."""
    clean = _clean_text_expr(col)
    return f"TRY_CAST(regexp_extract({clean}, '^(\\d+)', 1) AS INTEGER)"


def _count_positive_expr(col: str) -> str:
    """``(antal)`` columns: TRUE when > 0, FALSE when 0, NULL when not numeric."""
    n = _int_expr(col)
    return f"CASE WHEN {n} IS NULL THEN NULL ELSE ({n} > 0) END"


def _smoke_detect_expr(col: str) -> str:
    clean = _clean_text_expr(col)
    normalized = f"lower(replace(replace(coalesce({clean}, ''), '–', '-'), '−', '-'))"
    return f"({normalized} LIKE '%cigg%')"


def _normalized_codes_expr(col: str) -> str:
    """Upper-case the code list and drop whitespace and dots (``O14.1`` -> ``O141``)."""
    clean = _clean_text_expr(col)
    return f"regexp_replace(upper(coalesce({clean}, '')), '[\\s.]+', '', 'g')"


def _code_prefix_expr(col: str, prefixes: list[str], delimiter: str = ",") -> str:
    normalized = _normalized_codes_expr(col)
    return (
        "("
        + " OR ".join(
            f"regexp_matches({normalized}, '(^|{delimiter}){prefix}[A-Z0-9]*($|{delimiter})')"
            for prefix in prefixes
        )
        + ")"
    )


def _code_exact_expr(col: str, codes: list[str], delimiter: str = ",") -> str:
    normalized = _normalized_codes_expr(col)
    return (
        "("
        + " OR ".join(
            f"regexp_matches({normalized}, '(^|{delimiter}){code}($|{delimiter})')"
            for code in codes
        )
        + ")"
    )


def _normalized_glopnr_expr(col: str) -> str:
    return f"NULLIF(regexp_replace(regexp_replace(trim(CAST({col} AS VARCHAR)), '\\.0+$', ''), '\\s+', '', 'g'), '')"


def _has_value_expr(col: str) -> str:
    return f"({_clean_text_expr(col)} IS NOT NULL)"


def _seconds_before_expr(event_ts: str, reference_ts: str) -> str:
    """Seconds from ``event_ts`` to ``reference_ts`` (positive when the event precedes it)."""
    return (
        f"CASE WHEN {event_ts} IS NOT NULL AND {reference_ts} IS NOT NULL "
        f"THEN date_diff('second', {event_ts}, {reference_ts}) ELSE NULL END"
    )


# ---------------------------------------------------------------------------------------------
# Input views
# ---------------------------------------------------------------------------------------------


def _load_snq_view(con: duckdb.DuckDBPyConnection, snq_file: Path) -> None:
    if not snq_file.exists():
        raise FileNotFoundError(f"SNQ file not found: {snq_file}")

    suffix = snq_file.suffix.lower()
    if suffix == ".csv":
        con.execute(
            f"""
            CREATE VIEW snq_raw AS
            SELECT * FROM read_csv_auto('{_safe_path(snq_file)}', header=true, all_varchar=true)
            """
        )
        return

    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        snq_df = pd.read_excel(snq_file, dtype=str)
        con.register("snq_raw_df", snq_df)
        con.execute(
            """
            CREATE VIEW snq_raw AS
            SELECT * FROM snq_raw_df
            """
        )
        return

    raise ValueError(f"Unsupported SNQ file type: {snq_file.suffix}")


REG_DEFAULT_ROW_FILTER_SQL = "personnummer_mor IS NOT NULL"


def _create_reg_raw_view(con: duckdb.DuckDBPyConnection, registry_csv: Path) -> None:
    """Expose the semicolon-delimited registry CSV as the ``reg_raw`` view."""
    con.execute(
        f"""
        CREATE VIEW reg_raw AS
        SELECT * FROM read_csv_auto('{_safe_path(registry_csv)}', delim=';', header=true)
        """
    )


def _create_long_code_tables(
    con: duckdb.DuckDBPyConnection,
    mother_diag_csv: str | Path | None,
    child_diag_csv: str | Path | None,
    child_proc_csv: str | Path | None,
) -> bool:
    """Load the dated SPR long tables that exist.

    Creates ``mother_diag_long`` / ``child_diag_long`` / ``child_proc_long`` (glopnr,
    event_date, code) for the files present and the per-pregnancy ``long_codes`` table
    (glopnr plus comma-joined code lists) used to enrich the registry flags. Returns False
    when none of the files exist, in which case only the collapsed ``*_rad`` columns of
    ``gravniva.csv`` feed the flags.
    """
    specs = [
        (
            "mother_diag_long",
            mother_diag_csv,
            "startdatum_moderns_diagnoser",
            "diagnoskod_moderns_diagnoser",
        ),
        (
            "child_diag_long",
            child_diag_csv,
            "startdatum_barnets_diagnoser",
            "diagnoskod_barnets_diagnoser",
        ),
        (
            "child_proc_long",
            child_proc_csv,
            "startdatum_barnets_atgarder",
            "atgardskod_barnets_atgarder",
        ),
    ]
    loaded: list[str] = []
    for table, path, date_col, code_col in specs:
        if path is None or not Path(path).exists():
            print(f"Long table {table}: file not found, skipped ({path})")
            continue
        con.execute(
            f"""
            CREATE TEMP TABLE {table} AS
            SELECT
                {_normalized_glopnr_expr("glopnr")} AS glopnr,
                {_date_expr(date_col)} AS event_date,
                {_clean_text_expr(code_col)} AS code
            FROM read_csv_auto('{_safe_path(path)}', delim=';', header=true, all_varchar=true)
            WHERE {_clean_text_expr(code_col)} IS NOT NULL
            """
        )
        loaded.append(table)
    if not loaded:
        return False

    def agg(table: str) -> str:
        if table not in loaded:
            return "NULL::VARCHAR"
        return f"(SELECT string_agg(code, ',') FROM {table} t WHERE t.glopnr = g.glopnr)"

    con.execute(
        f"""
        CREATE TEMP TABLE long_codes AS
        SELECT
            g.glopnr AS lc_glopnr,
            {agg("mother_diag_long")} AS lc_mother_diag_codes,
            {agg("child_diag_long")} AS lc_child_diag_codes,
            {agg("child_proc_long")} AS lc_child_proc_codes
        FROM (
            {" UNION ".join(f"SELECT DISTINCT glopnr FROM {t}" for t in loaded)}
        ) g
        WHERE g.glopnr IS NOT NULL
        """
    )
    return True


# ---------------------------------------------------------------------------------------------
# Registry (gravniva) table
# ---------------------------------------------------------------------------------------------


def _create_reg_table(
    con: duckdb.DuckDBPyConnection,
    row_filter_sql: str = REG_DEFAULT_ROW_FILTER_SQL,
) -> None:
    """Build the typed ``reg`` temp table (one row per registry birth row) from ``reg_raw``.

    ``row_filter_sql`` is the WHERE clause applied to ``reg_raw``; the default drops rows
    without a maternal personnummer, exactly as Stage 7 always has. When the ``long_codes``
    table exists (see ``_create_long_code_tables``) its dated codes are unioned with the
    collapsed ``*_rad`` code strings before any code flag is derived.
    """
    use_long = _table_exists(con, "long_codes")
    available = _columns(con, "reg_raw")
    missing: list[str] = []

    def col(name: str) -> str:
        """Column reference, or NULL when the export lacks the column (reported once)."""
        if name in available:
            return name
        if name not in missing:
            missing.append(name)
        return "NULL::VARCHAR"

    smoke_pre = _clean_text_expr(col("tobak_3_manader_fore_graviditet"))
    smoke_inskrivning = _clean_text_expr(col("tobak_inskrivning"))
    smoke_w30 = _clean_text_expr(col("tobak_vecka_30_32"))
    sex_raw = _clean_text_expr(col("kon"))
    ph_art = _float_expr(col("ph_navelartar"))
    ph_ven = _float_expr(col("ph_navelven"))
    be_art = _float_expr(col("be_navelartar_mmol_l"))
    be_ven = _float_expr(col("be_navelven_mmol_l"))
    apgar5 = _int_expr(col("apgar_5_min"))
    apgar10 = _int_expr(col("apgar_10_min"))
    birth_day = _date_expr(col("forlossningsdatum_fv1"))
    birth_time_seconds = _int_expr(col("forlossningstid_fv1"))
    birth_timestamp = _timestamp_from_date_and_seconds(birth_day, birth_time_seconds)
    labour_day = _date_expr(col("etablerade_varkar_datum"))
    labour_time_seconds = _int_expr(col("etablerade_varkar_tid"))
    labour_timestamp = _timestamp_from_date_and_seconds(labour_day, labour_time_seconds)
    mother_birth_date = "TRY_CAST(try_strptime(substr(regexp_replace(CAST(personnummer_mor AS VARCHAR), '[^0-9]', '', 'g'), 1, 8), '%Y%m%d') AS DATE)"
    death_day = _date_expr(col("avled_datum"))
    discharge_day = _date_expr(col("utskrivning_datum"))

    def event_ts(date_col: str, time_col: str) -> str:
        return _timestamp_from_date_and_seconds(_date_expr(col(date_col)), _int_expr(col(time_col)))

    labour_onset_ts = event_ts("varkar_borjade_datum", "varkar_borjade_tid")
    rom_ts = event_ts("vattenavgang_datum", "vattenavgang_tid")
    amniotomy_ts = event_ts("amniotomi_datum", "amniotomi_tid")
    second_stage_ts = event_ts("krystvarkar_datum", "krystvarkar_tid")
    cs_start_ts = event_ts("sectio_start_datum", "sectio_start_tid")
    cs_end_ts = event_ts("sectio_slut_datum", "sectio_slut_tid")

    if use_long:
        mother_diag_col = f"concat_ws(',', {col('moderns_diagnoser_rad')}, lc_mother_diag_codes)"
        child_diag_col = f"concat_ws(',', {col('barnets_diagnoser_rad')}, lc_child_diag_codes)"
        child_proc_col = f"concat_ws(',', {col('barnets_atgarder_rad')}, lc_child_proc_codes)"
        long_join = f"LEFT JOIN long_codes ON long_codes.lc_glopnr = {_normalized_glopnr_expr('reg_raw.glopnr')}"
    else:
        mother_diag_col = col("moderns_diagnoser_rad")
        child_diag_col = col("barnets_diagnoser_rad")
        child_proc_col = col("barnets_atgarder_rad")
        long_join = ""
    mother_proc_col = col("moderns_atgarder_rad")

    def mprefix(codes: list[str]) -> str:
        return _code_prefix_expr(mother_diag_col, codes)

    def mexact(codes: list[str]) -> str:
        return _code_exact_expr(mother_diag_col, codes)

    def cprefix(codes: list[str]) -> str:
        return _code_prefix_expr(child_diag_col, codes)

    def cexact(codes: list[str]) -> str:
        return _code_exact_expr(child_diag_col, codes)

    # Maternal diagnoses (ICD-10-SE) -------------------------------------------------------
    gest_htn = mprefix(["O13", "O16"])
    preeclampsia = mprefix(["O14", "O15"])
    diabetes = mprefix(["O24"])
    uterine_rupture_diag = mexact(["O710", "O711"])
    sepsis = mprefix(["A41"])
    placental_abruption = mprefix(["O45"])
    heavy_bleeding = mprefix(["O46", "O67"])
    cord_prolapse = mexact(["O690"])
    shoulder_dystocia_diag = mprefix(["O66"])
    labor_dystocia = mexact(["O620", "O621", "O628", "O629"])
    fetal_distress_labour = mprefix(["O68"])
    maternal_care_fetal_problems = mprefix(["O36"])
    fetal_hypoxia_signs = mprefix(["O363"])
    fetal_growth_restriction_care = mprefix(["O365"])
    chorioamnionitis = mprefix(["O411"])
    oligohydramnios = mprefix(["O410"])
    polyhydramnios = mprefix(["O40"])
    prom = mprefix(["O42"])
    preterm_labour = mprefix(["O60"])
    prolonged_pregnancy = mprefix(["O48"])
    failed_induction = mprefix(["O61"])
    prolonged_labour = mprefix(["O63"])
    obstructed_labour = mprefix(["O64", "O65", "O66"])
    cord_complications = mprefix(["O69"])
    hypertensive_any = mprefix(["O10", "O11", "O12", "O13", "O14", "O15", "O16"])
    placenta_previa = mprefix(["O44"])
    postpartum_haemorrhage = mprefix(["O72"])
    intrapartum_fever = mprefix(["O752"])

    oxytocin = _code_exact_expr(mother_proc_col, ["DT036", "DT037"])
    uterine_rupture_proc = _code_exact_expr(mother_proc_col, ["MCC00"])

    # Child diagnoses / procedures ---------------------------------------------------------
    severe_asphyxia_diag = cexact(["P210", "P808", "P809"])
    meconium = cexact(["P240"])
    shoulder_dystocia_child = cexact(["P140", "P141", "P143", "P148", "P149"])
    hypoglycemia_treatment = cexact(["P703", "P704A", "P704B", "P708", "P709"])
    neonatal_anemia = cexact(["P612", "P613", "P614"])
    severe_asphyxia_proc = _code_exact_expr(child_proc_col, ["DV034"])
    intrauterine_hypoxia = cprefix(["P20"])
    birth_asphyxia_any = cprefix(["P21"])
    mild_moderate_asphyxia = cprefix(["P211"])
    respiratory_distress = cprefix(["P22"])
    aspiration_any = cprefix(["P24"])
    convulsions_icd = cprefix(["P90"])
    hie_icd = cprefix(["P916"])
    cerebral_disturbance = cprefix(["P91"])
    ich_icd = cprefix(["P10", "P52"])
    neonatal_infection_icd = cprefix(["P23", "P36", "P39"])
    hypoglycaemia_icd = cprefix(["P70"])
    birth_injury = cprefix(["P10", "P11", "P12", "P13", "P14", "P15"])
    malformation = cprefix(["Q"])
    respirator_grav = (
        f"({_has_value_expr(col('ventilation_pa_mask_min'))} OR "
        f"{_has_value_expr(col('intubation_min'))} OR "
        f"{_has_value_expr(col('hjartmassage_min'))})"
    )
    metabolic_acidosis = (
        f"CASE WHEN {ph_art} IS NULL OR {be_art} IS NULL THEN NULL "
        f"ELSE ({ph_art} < 7.05 AND {be_art} <= -12) END"
    )

    con.execute(
        f"""
        CREATE TEMP TABLE reg AS
        SELECT
            row_number() OVER () AS reg_row,
            {_normalized_glopnr_expr("reg_raw.glopnr")} AS glopnr,
            regexp_replace(CAST(personnummer_mor AS VARCHAR), '[^0-9]', '', 'g') AS reg_digits,
            {birth_day} AS birth_day,
            {birth_time_seconds} AS birth_time_seconds,
            {birth_timestamp} AS birth_timestamp,
            CASE
                WHEN {mother_birth_date} IS NOT NULL AND {birth_day} IS NOT NULL THEN
                    date_diff('year', {mother_birth_date}, {birth_day})
                    - CASE
                        WHEN strftime({birth_day}, '%m-%d') < strftime({mother_birth_date}, '%m-%d') THEN 1
                        ELSE 0
                    END
                ELSE NULL
            END AS maternal_age,
            {labour_day} AS etablerade_varkar_datum,
            {labour_time_seconds} AS etablerade_varkar_tid,
            {labour_timestamp} AS etablerade_varkar_timestamp,
            {_seconds_before_expr(labour_timestamp, birth_timestamp)} AS etablerade_varkar_seconds,
            {_clean_text_expr(col("forlossningsstart_basta_skattning"))} AS forlossningsstart,
            {_clean_text_expr(col("forlossningsslut_basta_skattning"))} AS forlossningsslut,
            {_int_expr(col("apgar_1_min"))} AS apgar1,
            {apgar5} AS apgar5,
            {apgar10} AS apgar10,
            ({_int_expr(col("gl_v_barn"))} * 7 + {_int_expr(col("gl_d_barn"))}) AS gestational_days,
            {_clean_text_expr(col("fodelseland"))} AS fodelseland,
            {_clean_text_expr(col("utbildningsniva"))} AS utbildningsniva,
            {_int_expr(col("para_mhv1"))} AS para_mhv1,
            {_float_expr(col("langd_inskrivning_cm"))} AS langd_inskrivning_cm,
            {_float_expr(col("bmi_inskrivning"))} AS bmi_inskrivning,
            CASE WHEN lower(coalesce({_clean_text_expr(col("tidigare_sectio"))}, '')) = 'ja' THEN TRUE ELSE FALSE END AS previous_c_section,
            {smoke_pre} AS tobak_3_manader_fore_graviditet,
            {smoke_inskrivning} AS tobak_inskrivning,
            {smoke_w30} AS tobak_vecka_30_32,
            CASE
                WHEN {_smoke_detect_expr(col("tobak_3_manader_fore_graviditet"))}
                  OR {_smoke_detect_expr(col("tobak_inskrivning"))}
                  OR {_smoke_detect_expr(col("tobak_vecka_30_32"))}
                THEN TRUE
                ELSE FALSE
            END AS is_smoker,
            {_bool_ja_nej_expr(col("diabetes_mellitus"))} AS diabetes_mellitus,
            CASE
                WHEN {sex_raw} = 'Flicka' THEN 'Flicka'
                WHEN {sex_raw} IS NULL THEN NULL
                ELSE 'Pojke'
            END AS child_sex,
            CASE
                WHEN {sex_raw} = 'Flicka' THEN TRUE
                WHEN {sex_raw} IS NULL THEN NULL
                ELSE FALSE
            END AS is_girl,
            {_int_expr(col("alkohol_audit_poang"))} AS alkohol_audit_poang,
            {ph_art} AS ph_navelartar,
            {ph_ven} AS ph_navelven,
            CASE
                WHEN {ph_art} IS NOT NULL THEN ({ph_art} < 7)
                WHEN {ph_ven} IS NOT NULL THEN ({ph_ven} < 7)
                ELSE NULL
            END AS ph_navel_below7,
            {death_day} AS avled_datum,
            CASE
                WHEN {death_day} IS NOT NULL
                 AND {birth_day} IS NOT NULL
                 AND date_diff('day', {birth_day}, {death_day}) BETWEEN 0 AND 28
                THEN date_diff('day', {birth_day}, {death_day})
                ELSE NULL
            END AS died_after_days,
            {gest_htn} AS gestational_hypertension_without_significant_proteinuria,
            {preeclampsia} AS preeclampsia,
            {diabetes} AS gestational_or_pregestational_diabetes,
            ({uterine_rupture_diag} OR {uterine_rupture_proc}) AS uterine_rupture,
            {sepsis} AS sepsis,
            {placental_abruption} AS placental_abruption,
            {heavy_bleeding} AS heavy_vaginal_bleeding_before_or_during_delivery,
            {cord_prolapse} AS umbilical_cord_prolapse,
            ({shoulder_dystocia_diag} OR {shoulder_dystocia_child}) AS shoulder_dystocia,
            {labor_dystocia} AS labor_dystocia,
            {oxytocin} AS use_of_oxytocin,
            ({severe_asphyxia_diag} OR {severe_asphyxia_proc}) AS severe_birth_asphyxia,
            {meconium} AS meconium_aspiration_syndrome,
            {hypoglycemia_treatment} AS treatment_for_hypoglycemia,
            {neonatal_anemia} AS neonatal_anemia,
            {respirator_grav} AS respiratorbehandling_gravniva,

            -- Delivery mode and labour course (gravniva) ---------------------------------
            {_text_equals_expr(col("forlossningsslut_basta_skattning"), "Akut kejsarsnitt")} AS emergency_c_section,
            {_text_equals_expr(col("forlossningsslut_basta_skattning"), "Planerat kejsarsnitt")} AS planned_c_section,
            {_text_equals_expr(col("forlossningsslut_basta_skattning"), "Instrumentell vaginal förlossning")} AS instrumental_vaginal_delivery,
            {_clean_text_expr(col("indikation"))} AS c_section_urgency,
            {_text_equals_expr(col("forlossningsstart_basta_skattning"), "Induktion")} AS induced_labour,
            {_bool_ja_nej_expr(col("oxytocin_under_forlossning"))} AS oxytocin_under_forlossning,
            {_bool_ja_nej_expr(col("smartlindring_epidural"))} AS epidural,
            {_clean_text_expr(col("presentation"))} AS presentation,
            {_text_equals_expr(col("presentation"), "Sätes- eller fotbjudning")} AS breech_presentation,
            {_clean_text_expr(col("robsongrupp"))} AS robsongrupp,
            {_seconds_before_expr(labour_onset_ts, birth_timestamp)} AS labour_onset_seconds_before_birth,
            {_seconds_before_expr(rom_ts, birth_timestamp)} AS membrane_rupture_seconds_before_birth,
            {_seconds_before_expr(amniotomy_ts, birth_timestamp)} AS amniotomy_seconds_before_birth,
            {_seconds_before_expr(second_stage_ts, birth_timestamp)} AS second_stage_seconds_before_birth,
            {_seconds_before_expr(cs_start_ts, birth_timestamp)} AS c_section_start_seconds_before_birth,
            {_seconds_before_expr(cs_start_ts, cs_end_ts)} AS c_section_duration_seconds,
            {_int_expr(col("total_blodning_ml"))} AS total_blodning_ml,
            {_clean_text_expr(col("ctg_intagningstest"))} AS ctg_intagningstest,
            CASE
                WHEN {_clean_text_expr(col("ctg_intagningstest"))} = 'Ej normal' THEN TRUE
                WHEN {_clean_text_expr(col("ctg_intagningstest"))} = 'Normal' THEN FALSE
                ELSE NULL
            END AS ctg_admission_test_abnormal,

            -- Maternal background (gravniva) --------------------------------------------
            {_bool_ja_nej_expr(col("ivf_graviditet"))} AS ivf_graviditet,
            {_bool_ja_nej_expr(col("kronisk_hypertoni"))} AS kronisk_hypertoni,
            {_bool_ja_prefix_expr(col("diagnosen_graviditetsdiabetes_stalld"))} AS graviditetsdiabetes_diagnos,

            -- Neonatal condition at birth (gravniva) --------------------------------------
            {_int_expr(col("fodelsevikt_g"))} AS birth_weight_g,
            {_float_expr(col("vikt_avvikelse_perc"))} AS birth_weight_deviation_perc,
            {be_art} AS be_navelartar,
            {be_ven} AS be_navelven,
            {_float_expr(col("pco2_navelartar_kpa"))} AS pco2_navelartar,
            {_float_expr(col("po2_navelartar_kpa"))} AS po2_navelartar,
            CASE
                WHEN {ph_art} IS NOT NULL THEN ({ph_art} < 7.05)
                WHEN {ph_ven} IS NOT NULL THEN ({ph_ven} < 7.05)
                ELSE NULL
            END AS ph_navel_below705,
            {metabolic_acidosis} AS metabolic_acidosis,
            CASE WHEN {apgar5} IS NULL THEN NULL ELSE ({apgar5} < 7) END AS apgar5_below7,
            CASE WHEN {apgar10} IS NULL THEN NULL ELSE ({apgar10} < 7) END AS apgar10_below7,
            {_int_expr(col("ventilation_pa_mask_min"))} AS ventilation_pa_mask_min,
            {_int_expr(col("intubation_min"))} AS intubation_min,
            {_int_expr(col("hjartmassage_min"))} AS hjartmassage_min,
            {_bool_ja_nej_expr(col("acidoskorrektion"))} AS acidoskorrektion,
            {_bool_ja_nej_expr(col("utskriven_till_hemmet"))} AS discharged_home,
            CASE
                WHEN {discharge_day} IS NOT NULL AND {birth_day} IS NOT NULL
                THEN date_diff('day', {birth_day}, {discharge_day})
                ELSE NULL
            END AS days_to_discharge,

            -- Additional maternal diagnosis flags (ICD-10 codes, gravniva + long table) ---
            {fetal_distress_labour} AS fetal_distress_in_labour,
            {maternal_care_fetal_problems} AS maternal_care_for_fetal_problems,
            {fetal_hypoxia_signs} AS signs_of_fetal_hypoxia_antenatal,
            {fetal_growth_restriction_care} AS fetal_growth_restriction,
            {chorioamnionitis} AS chorioamnionitis,
            {oligohydramnios} AS oligohydramnios,
            {polyhydramnios} AS polyhydramnios,
            {prom} AS prelabour_rupture_of_membranes,
            {preterm_labour} AS preterm_labour,
            {prolonged_pregnancy} AS prolonged_pregnancy,
            {failed_induction} AS failed_induction,
            {prolonged_labour} AS prolonged_labour,
            {obstructed_labour} AS obstructed_labour,
            {cord_complications} AS umbilical_cord_complications,
            {hypertensive_any} AS hypertensive_disorder_any,
            {placenta_previa} AS placenta_previa,
            {postpartum_haemorrhage} AS postpartum_haemorrhage,
            {intrapartum_fever} AS intrapartum_fever,

            -- Additional child diagnosis flags (ICD-10 codes, first 28 days) --------------
            {intrauterine_hypoxia} AS intrauterine_hypoxia,
            {birth_asphyxia_any} AS birth_asphyxia_any,
            {mild_moderate_asphyxia} AS mild_or_moderate_birth_asphyxia,
            {respiratory_distress} AS respiratory_distress_newborn,
            {aspiration_any} AS neonatal_aspiration_syndromes,
            {convulsions_icd} AS neonatal_convulsions_icd,
            {hie_icd} AS hie_icd,
            {cerebral_disturbance} AS cerebral_disturbance_newborn,
            {ich_icd} AS intracranial_haemorrhage_icd,
            {neonatal_infection_icd} AS neonatal_infection_icd,
            {hypoglycaemia_icd} AS neonatal_hypoglycaemia_icd,
            {birth_injury} AS birth_injury,
            {malformation} AS congenital_malformation
        FROM reg_raw
        {long_join}
        WHERE {row_filter_sql}
        """
    )
    if missing:
        print(
            f"WARNING: gravniva columns not in export, derived variables will be NULL: {missing}",
            file=sys.stderr,
        )


def _create_reg_clean_table(con: duckdb.DuckDBPyConnection, salt: str | None = None) -> None:
    """Keep the ``reg`` rows usable for matching and derive their CTG-style ``PatientID``.

    This is the single definition of which registry rows enter Stage 7 matching; the
    match-loss report reuses it so the two cannot drift. ``MotherID`` is the pseudonymous
    mother key (see ``pseudonyms.py``), hashed with the BabyID salt.
    """
    if salt is None:
        salt = get_secret("babyid_salt")
    patient_sql = "substr(reg_digits, 1, 8) || '-' || substr(reg_digits, 9, 4)"
    con.execute(
        f"""
        CREATE TEMP TABLE reg_clean AS
        SELECT
            * EXCLUDE (reg_digits),
            {patient_sql} AS PatientID,
            {mother_id_sql(salt, patient_sql)} AS MotherID
        FROM reg
        WHERE reg_digits IS NOT NULL
          AND length(reg_digits) >= 12
          AND apgar5 IS NOT NULL
          AND birth_day IS NOT NULL
        """
    )


def _ctg_day_match_predicate(ctg_date: str, birth_day: str) -> str:
    """Stage 7 day rule: the CTG anchor date is the birth date or the day before it."""
    return f"({ctg_date} = {birth_day} OR {ctg_date} = {birth_day} - INTERVAL 1 DAY)"


# ---------------------------------------------------------------------------------------------
# SNQ table
# ---------------------------------------------------------------------------------------------

# (output column, SNQ source column, expression builder, aggregate). The aggregate matters
# only if an SNQ export ever carries several rows per pregnancy.
SNQ_VARIABLES: list[tuple[str, str, str, str]] = [
    ("snq_hypothermia_treatment", "Behandlad med hypotermi", "ja_nej", "bool"),
    ("snq_seizures", "Kramper", "ja_nej", "bool"),
    ("snq_antiepileptic_treatment", "AntiEp_beh under vtf", "ja_nej", "bool"),
    ("snq_eeg_monitoring", "EEG/aEEG övervakning", "ja_nej", "bool"),
    ("snq_cns_haemorrhage", "CNS - blödning", "ja_nej", "bool"),
    ("snq_cns_infarct", "Fokal/multifokal CNS infarkt", "ja_nej", "bool"),
    ("snq_pvl", "PVL (med cystor)", "ja_nej", "bool"),
    ("snq_highest_ivh", "Högst IVH", "ivh_grade", "max"),
    ("snq_resuscitation", "HLR-åtgärder", "hlr", "bool"),
    ("snq_resuscitation_over_10min", "HLR-åtgärder", "hlr_over_10", "bool"),
    ("snq_hlr_extra_oxygen", "HLR_Extra O 2", "ja_nej", "bool"),
    ("snq_hlr_ventilation_mask", "HLR_Ventilation via mask", "ja_nej", "bool"),
    ("snq_hlr_cpap", "HLR_CPAP", "ja_nej", "bool"),
    ("snq_hlr_intubation", "HLR_Intubation", "ja_nej", "bool"),
    ("snq_hlr_chest_compressions", "HLR_Hjärtmassage", "ja_nej", "bool"),
    ("snq_hlr_adrenaline", "HLR_Adrenalin", "ja_nej", "bool"),
    ("snq_cpap", "CPAP", "ja_nej", "bool"),
    ("snq_high_flow", "Högflödesgrimma", "ja_nej", "bool"),
    ("snq_ventilator_conventional", "Resp konv", "ja_nej", "bool"),
    ("snq_ventilator_hfv", "Resp HFV", "ja_nej", "bool"),
    ("snq_ventilator_nava", "Resp NAVA", "ja_nej", "bool"),
    ("snq_nas", "NAS", "ja_nej", "bool"),
    ("snq_pas", "PAS", "ja_nej", "bool"),
    ("snq_mas", "MAS", "ja_nej", "bool"),
    ("snq_rds", "RDS", "ja_nej", "bool"),
    ("snq_pphn", "PPHN", "ja_nej", "bool"),
    ("snq_pneumothorax", "Pneumothorax", "ja_nej", "bool"),
    ("snq_bpd", "BPD", "ja_nej", "bool"),
    ("snq_infection", "Barn med infektion", "ja_nej", "bool"),
    (
        "snq_early_culture_verified_sepsis",
        "Tidig bakt. sepsis, odlingsverif. (antal)",
        "count_positive",
        "bool",
    ),
    ("snq_hypoglycaemia", "Hypoglukemi (<2,6 efter 3 tim)", "ja_nej", "bool"),
    ("snq_inotropic_support", "Inotroptstöd", "ja_nej", "bool"),
    ("snq_erythrocyte_transfusion", "EryTransf", "ja_nej", "bool"),
    ("snq_malformation_or_chromosomal", "Missb./kromosom avv.", "ja_nej", "bool"),
    ("snq_died", "Avliden enl. SNQ", "ja_nej", "bool"),
    ("snq_age_at_death_days", "Ålder vid dödsfall SNQ (dagar)", "int", "max"),
    ("snq_death_cause_perinatal_asphyxia", "Dödors_Perinatal asfyxi", "ja_nej", "bool"),
    ("snq_died_death_register", "Avliden enl. DOR", "ja_nej", "bool"),
    ("snq_admission_age_days", "1:a inskrivning, ålder (dagar)", "int", "min"),
    ("snq_admissions", "Antal Vtf", "int", "max"),
    ("snq_care_days_inpatient", "Vårdtid, inneliggande", "int", "max"),
    ("snq_care_days_neonatal", "Vårdtid, neonatologi", "int", "max"),
    ("snq_apgar1", "Apgar1m", "int", "max"),
    ("snq_apgar5", "Apgar5m", "int", "max"),
    ("snq_apgar10", "Apgar10m", "int", "max"),
    ("snq_ph_navelartar", "Artär pH", "float", "max"),
    ("snq_be_navelartar", "Artär BE", "float", "max"),
    ("snq_ph_navelven", "Ven pH", "float", "max"),
    ("snq_be_navelven", "Ven BE", "float", "max"),
    ("snq_postnatal_ph", "Post pH", "float", "min"),
    ("snq_postnatal_be", "Post BE", "float", "min"),
    ("snq_gestational_weeks", "Grav.längd (v)", "int", "max"),
    ("snq_birth_weight_g", "Födelsevikt", "float", "max"),
    ("snq_birth_weight_zscore", "ZScore", "float", "max"),
    ("snq_iugr", "Intrauterin tillväxthämning", "ja_nej", "bool"),
    ("snq_preeclampsia", "Preeklampsi/Eklampsi", "ja_nej", "bool"),
    ("snq_chorioamnionitis", "Amnionit", "ja_nej", "bool"),
    ("snq_abruption_or_bleeding", "Ablatio/Blödning", "ja_nej", "bool"),
]


def _snq_expr(kind: str, col: str) -> str:
    if kind == "ja_nej":
        return _bool_ja_nej_expr(col)
    if kind == "int":
        return _int_expr(col)
    if kind == "float":
        return _float_expr(col)
    if kind == "count_positive":
        return _count_positive_expr(col)
    if kind == "ivh_grade":
        n = _leading_int_expr(col)
        return f"CASE WHEN {n} BETWEEN 0 AND 4 THEN {n} ELSE NULL END"
    if kind == "hlr":
        # 0 Nej (vitalt barn), 1 Ja (<10 min), 2 Ja (>=10 min), 3 Nej (palliation), us
        n = _leading_int_expr(col)
        return f"CASE WHEN {n} IN (1, 2) THEN TRUE WHEN {n} = 0 THEN FALSE ELSE NULL END"
    if kind == "hlr_over_10":
        n = _leading_int_expr(col)
        return f"CASE WHEN {n} = 2 THEN TRUE WHEN {n} IN (0, 1) THEN FALSE ELSE NULL END"
    raise ValueError(f"Unknown SNQ expression kind: {kind}")


def _snq_agg(agg: str, name: str) -> str:
    if agg == "bool":
        return f"BOOL_OR({name}) FILTER (WHERE {name} IS NOT NULL) AS {name}"
    if agg == "max":
        return f"MAX({name}) AS {name}"
    if agg == "min":
        return f"MIN({name}) AS {name}"
    raise ValueError(f"Unknown SNQ aggregate: {agg}")


def _create_snq_table(con: duckdb.DuckDBPyConnection) -> None:
    """Build the per-pregnancy ``snq`` temp table from ``snq_raw``.

    SNQ columns absent from the export are reported and yield NULL columns so that an older
    or trimmed export still runs.
    """
    available = _columns(con, "snq_raw")

    def src(col: str) -> str:
        if col in available:
            return '"' + col.replace('"', '""') + '"'
        print(f"WARNING: SNQ column not in export, output will be NULL: {col}", file=sys.stderr)
        return "NULL::VARCHAR"

    pre_cols = [
        f"{_normalized_glopnr_expr(src('glopnr'))} AS glopnr",
        f"{_int_expr(src('Högst HIE'))} AS highest_hie",
        f"{_bool_ja_nej_expr(src('HIE'))} AS hie",
        f"{_code_prefix_expr(src('ICD_kod'), ['P10', 'P52'], delimiter=';')} AS intracranial_haemorrhage",
        f"{_code_prefix_expr(src('ICD_kod'), ['P90'], delimiter=';')} AS neonatal_convulsions",
        f"{_code_prefix_expr(src('ICD_kod'), ['P23', 'P36', 'P392'], delimiter=';')} AS neonatal_sepsis_or_pneumonia",
        f"{_code_prefix_expr(src('KVÅ_kod'), ['DG021', 'DG022', 'DG0002'], delimiter=';')} AS respiratorbehandling_snq",
    ]
    agg_cols = [
        "MAX(highest_hie) AS highest_hie",
        _snq_agg("bool", "hie"),
        _snq_agg("bool", "intracranial_haemorrhage"),
        _snq_agg("bool", "neonatal_convulsions"),
        _snq_agg("bool", "neonatal_sepsis_or_pneumonia"),
        _snq_agg("bool", "respiratorbehandling_snq"),
    ]
    for name, col, kind, agg in SNQ_VARIABLES:
        pre_cols.append(f"{_snq_expr(kind, src(col))} AS {name}")
        agg_cols.append(_snq_agg(agg, name))

    con.execute(
        f"""
        CREATE TEMP TABLE snq AS
        WITH snq_pre AS (
            SELECT
                {", ".join(pre_cols)}
            FROM snq_raw
        )
        SELECT
            glopnr,
            {", ".join(agg_cols)}
        FROM snq_pre
        WHERE glopnr IS NOT NULL
        GROUP BY glopnr
        """
    )


# Composite label: any severe neonatal outcome plausibly related to intrapartum hypoxia.
# Components missing because the child is not in SNQ count as FALSE (not admitted to
# neonatal care); gravniva components are present for every matched row.
SEVERE_NEONATAL_OUTCOME_SQL = """
(
    COALESCE(r.apgar5 < 7, FALSE)
    OR COALESCE(r.ph_navelartar < 7.0, FALSE)
    OR COALESCE(r.metabolic_acidosis, FALSE)
    OR COALESCE(r.hie_icd, FALSE)
    OR COALESCE(r.severe_birth_asphyxia, FALSE)
    OR r.died_after_days IS NOT NULL
    OR r.intubation_min IS NOT NULL
    OR COALESCE(s.hie, FALSE)
    OR COALESCE(s.snq_hypothermia_treatment, FALSE)
    OR COALESCE(s.snq_seizures, FALSE)
    OR COALESCE(s.neonatal_convulsions, FALSE)
    OR COALESCE(s.snq_died, FALSE)
    OR COALESCE(s.snq_resuscitation_over_10min, FALSE)
    OR COALESCE(s.snq_hlr_intubation, FALSE)
)
"""


def _create_reg_enriched_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE TEMP TABLE reg_enriched AS
        SELECT
            r.*,
            s.* EXCLUDE (glopnr, respiratorbehandling_snq),
            CASE
                WHEN r.respiratorbehandling_gravniva THEN TRUE
                WHEN s.respiratorbehandling_snq IS NOT NULL THEN s.respiratorbehandling_snq
                ELSE NULL
            END AS respiratorbehandling,
            (s.glopnr IS NOT NULL) AS neonatal_care_admission,
            {SEVERE_NEONATAL_OUTCOME_SQL} AS severe_neonatal_outcome
        FROM reg_clean r
        LEFT JOIN snq s USING (glopnr)
        """
    )


def _export_long_table(con: duckdb.DuckDBPyConnection, table: str, out: Path, label: str) -> None:
    """Write ``table`` (glopnr, event_date, code) restricted to matched babies, keyed by
    BabyID with the date replaced by the day offset from birth."""
    if not _table_exists(con, table):
        print(f"Skipped {label}: source table not loaded")
        return
    _ensure_parent(out)
    con.execute(
        f"""
        COPY (
            SELECT
                u.BabyID,
                date_diff('day', u.birth_day, l.event_date) AS day_offset,
                l.code
            FROM {table} l
            JOIN unique_matches u USING (glopnr)
            ORDER BY u.BabyID, day_offset, l.code
        ) TO '{_safe_path(out)}' (HEADER, DELIMITER ',')
        """
    )
    n = _count(con, f"SELECT COUNT(*) FROM {table} l JOIN unique_matches u USING (glopnr)")
    print(f"Wrote {label}: {out} ({n} rows)")


def registry_match(
    registry_csv: str | Path,
    snq_file: str | Path,
    stage5_5_file: str | Path,
    stage6_dir: str | Path,
    registry_out: str | Path,
    ctg_out: str | Path,
    mother_diag_csv: str | Path | None = DEFAULT_MOTHER_DIAG_CSV,
    child_diag_csv: str | Path | None = DEFAULT_CHILD_DIAG_CSV,
    child_proc_csv: str | Path | None = DEFAULT_CHILD_PROC_CSV,
    mother_diag_out: str | Path = DEFAULT_STAGE7_MOTHER_DIAG_CSV,
    child_diag_out: str | Path = DEFAULT_STAGE7_CHILD_DIAG_CSV,
    child_proc_out: str | Path = DEFAULT_STAGE7_CHILD_PROC_CSV,
    show_progress: bool = True,
) -> None:
    registry_csv = Path(registry_csv)
    snq_file = Path(snq_file)
    stage5_5_file = Path(stage5_5_file)
    stage6_dir = Path(stage6_dir)
    registry_out = Path(registry_out)
    ctg_out = Path(ctg_out)

    _ensure_parent(registry_out)
    _ensure_parent(ctg_out)

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

    _create_reg_raw_view(con, registry_csv)
    _load_snq_view(con, snq_file)
    _create_long_code_tables(con, mother_diag_csv, child_diag_csv, child_proc_csv)
    _create_reg_table(con)
    _create_snq_table(con)
    _create_reg_clean_table(con)
    _create_reg_enriched_table(con)

    con.execute(
        f"""
        CREATE VIEW s55 AS
        SELECT BabyID, PatientID, ctg_date
        FROM read_parquet('{_safe_path(stage5_5_file)}')
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE map AS
        SELECT DISTINCT BabyID, PatientID, ctg_date
        FROM s55
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE matches AS
        SELECT
            r.*,
            m.BabyID,
            m.ctg_date
        FROM reg_enriched r
        JOIN map m
          ON r.PatientID = m.PatientID
         AND {_ctg_day_match_predicate("m.ctg_date", "r.birth_day")}
        """
    )

    multi_rows = _count(con, MULTI_BABY_REGISTRY_ROWS_SQL)
    multi_babies = _count(con, MULTI_REGISTRY_BABIES_SQL)

    con.execute(f"CREATE TEMP TABLE unique_matches AS {UNIQUE_MATCHES_SQL}")

    total_rows = _count(con, "SELECT COUNT(*) FROM reg_raw")
    clean_rows = _count(con, "SELECT COUNT(*) FROM reg_clean")
    match_rows = _count(con, "SELECT COUNT(*) FROM unique_matches")
    snq_rows = _count(con, "SELECT COUNT(*) FROM unique_matches WHERE neonatal_care_admission")
    severe_rows = _count(con, "SELECT COUNT(*) FROM unique_matches WHERE severe_neonatal_outcome")

    print(f"Registry rows total: {total_rows}")
    print(f"Registry rows with valid apgar/birth_day: {clean_rows}")
    print(f"Matched rows: {match_rows}")
    print(f"Matched rows in SNQ (neonatal care admission): {snq_rows}")
    print(f"Matched rows with severe_neonatal_outcome: {severe_rows}")
    if multi_rows:
        print(f"WARNING: {multi_rows} registry rows matched multiple BabyIDs and were dropped.")
    if multi_babies:
        print(
            f"WARNING: {multi_babies} BabyIDs matched multiple registry rows "
            "(twins/multiples or duplicate registry rows) and were dropped."
        )

    ordered_cols = [row[0] for row in con.execute("DESCRIBE unique_matches").fetchall()]
    output_cols = ["BabyID", "MotherID"] + [
        c for c in ordered_cols if c not in IDENTIFYING_COLUMNS and c not in ("BabyID", "MotherID")
    ]
    leaked = [c for c in output_cols if "personnummer" in c.lower() or c in IDENTIFYING_COLUMNS]
    if leaked:
        raise RuntimeError(f"Identifying columns would be exported: {leaked}")

    con.execute(
        f"""
        COPY (
            SELECT {", ".join(output_cols)}
            FROM unique_matches
            ORDER BY BabyID
        ) TO '{_safe_path(registry_out)}'
        (HEADER, DELIMITER ',')
        """
    )

    _export_long_table(con, "mother_diag_long", Path(mother_diag_out), "mother diagnoses")
    _export_long_table(con, "child_diag_long", Path(child_diag_out), "child diagnoses")
    _export_long_table(con, "child_proc_long", Path(child_proc_out), "child procedures")

    con.execute(
        """
        CREATE TEMP TABLE matched_babies AS
        SELECT DISTINCT BabyID FROM unique_matches
        """
    )

    con.execute(
        f"""
        CREATE VIEW s6 AS
        SELECT * FROM read_parquet('{_safe_path(stage6_dir)}/**/*.parquet')
        """
    )

    s6_cols = _columns(con, "s6")
    keep_cols = ["BabyID", "Timestamp", "FHR", "toco"] + [
        name for name in DEFAULT_STAGE2_EXTRA_COLUMNS if name in s6_cols
    ]
    ctg_select = ", ".join(f"s6.{name}" for name in keep_cols)

    con.execute(
        f"""
        COPY (
            SELECT {ctg_select}
            FROM s6
            JOIN matched_babies mb USING (BabyID)
        ) TO '{_safe_path(ctg_out)}'
        (FORMAT PARQUET)
        """
    )

    print(f"Wrote registry CSV: {registry_out} ({len(output_cols)} columns)")
    print(f"Wrote CTG parquet: {ctg_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 registry matching and anonymized output.")
    parser.add_argument("--registry-csv", type=str, default=DEFAULT_PATIENT_CSV)
    parser.add_argument("--snq-file", type=str, default=DEFAULT_SNQ_FILE)
    parser.add_argument("--mother-diag", type=str, default=DEFAULT_MOTHER_DIAG_CSV)
    parser.add_argument("--child-diag", type=str, default=DEFAULT_CHILD_DIAG_CSV)
    parser.add_argument("--child-proc", type=str, default=DEFAULT_CHILD_PROC_CSV)
    parser.add_argument("--stage5-5", type=str, default=DEFAULT_STAGE5_5_OUTPUT_FILE)
    parser.add_argument("--stage6", type=str, default=DEFAULT_STAGE6_DIR)
    parser.add_argument("--registry-out", type=str, default=DEFAULT_STAGE7_REGISTRY_CSV)
    parser.add_argument("--ctg-out", type=str, default=DEFAULT_STAGE7_CTG_PARQUET)
    parser.add_argument("--mother-diag-out", type=str, default=DEFAULT_STAGE7_MOTHER_DIAG_CSV)
    parser.add_argument("--child-diag-out", type=str, default=DEFAULT_STAGE7_CHILD_DIAG_CSV)
    parser.add_argument("--child-proc-out", type=str, default=DEFAULT_STAGE7_CHILD_PROC_CSV)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    registry_match(
        registry_csv=args.registry_csv,
        snq_file=args.snq_file,
        stage5_5_file=args.stage5_5,
        stage6_dir=args.stage6,
        registry_out=args.registry_out,
        ctg_out=args.ctg_out,
        mother_diag_csv=args.mother_diag,
        child_diag_csv=args.child_diag,
        child_proc_csv=args.child_proc,
        mother_diag_out=args.mother_diag_out,
        child_diag_out=args.child_diag_out,
        child_proc_out=args.child_proc_out,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
