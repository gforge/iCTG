from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from config import (
    DEFAULT_PATIENT_CSV,
    DEFAULT_SNQ_FILE,
    DEFAULT_STAGE2_EXTRA_COLUMNS,
    DEFAULT_STAGE5_5_OUTPUT_FILE,
    DEFAULT_STAGE6_DIR,
    DEFAULT_STAGE7_CTG_PARQUET,
    DEFAULT_STAGE7_REGISTRY_CSV,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    clean = _clean_text_expr(col)
    return (
        "CASE "
        f"WHEN lower({clean}) = 'ja' THEN TRUE "
        f"WHEN lower({clean}) = 'nej' THEN FALSE "
        "ELSE NULL END"
    )


def _smoke_detect_expr(col: str) -> str:
    clean = _clean_text_expr(col)
    normalized = f"lower(replace(replace(coalesce({clean}, ''), '–', '-'), '−', '-'))"
    return f"({normalized} LIKE '%cigg%')"


def _normalized_codes_expr(col: str) -> str:
    clean = _clean_text_expr(col)
    return f"regexp_replace(upper(coalesce({clean}, '')), '\\s+', '', 'g')"


def _code_prefix_expr(col: str, prefixes: list[str], delimiter: str = ',') -> str:
    normalized = _normalized_codes_expr(col)
    return '(' + ' OR '.join(
        f"regexp_matches({normalized}, '(^|{delimiter}){prefix}[A-Z0-9]*($|{delimiter})')"
        for prefix in prefixes
    ) + ')'


def _code_exact_expr(col: str, codes: list[str], delimiter: str = ',') -> str:
    normalized = _normalized_codes_expr(col)
    return '(' + ' OR '.join(
        f"regexp_matches({normalized}, '(^|{delimiter}){code}($|{delimiter})')"
        for code in codes
    ) + ')'


def _normalized_glopnr_expr(col: str) -> str:
    return (
        f"NULLIF(regexp_replace(regexp_replace(trim(CAST({col} AS VARCHAR)), '\\.0+$', ''), '\\s+', '', 'g'), '')"
    )


def _has_value_expr(col: str) -> str:
    return f"({_clean_text_expr(col)} IS NOT NULL)"


def _load_snq_view(con: duckdb.DuckDBPyConnection, snq_file: Path) -> None:
    if not snq_file.exists():
        raise FileNotFoundError(f'SNQ file not found: {snq_file}')

    suffix = snq_file.suffix.lower()
    if suffix == '.csv':
        safe_snq = str(snq_file).replace("'", "''")
        con.execute(
            f"""
            CREATE VIEW snq_raw AS
            SELECT * FROM read_csv_auto('{safe_snq}', header=true)
            """
        )
        return

    if suffix in {'.xlsx', '.xls'}:
        import pandas as pd

        snq_df = pd.read_excel(snq_file, dtype=str)
        con.register('snq_raw_df', snq_df)
        con.execute(
            """
            CREATE VIEW snq_raw AS
            SELECT * FROM snq_raw_df
            """
        )
        return

    raise ValueError(f'Unsupported SNQ file type: {snq_file.suffix}')


def registry_match(
    registry_csv: str | Path,
    snq_file: str | Path,
    stage5_5_file: str | Path,
    stage6_dir: str | Path,
    registry_out: str | Path,
    ctg_out: str | Path,
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
            con.execute('PRAGMA enable_progress_bar')
            con.execute('PRAGMA progress_bar_time=5')
        except Exception:
            pass
    try:
        con.execute('SET preserve_insertion_order=false')
    except Exception:
        pass

    safe_registry = str(registry_csv).replace("'", "''")
    safe_stage5_5 = str(stage5_5_file).replace("'", "''")
    safe_stage6 = str(stage6_dir).replace("'", "''")

    con.execute(
        f"""
        CREATE VIEW reg_raw AS
        SELECT * FROM read_csv_auto('{safe_registry}', delim=';', header=true)
        """
    )
    _load_snq_view(con, snq_file)

    smoke_pre = _clean_text_expr('tobak_3_manader_fore_graviditet')
    smoke_inskrivning = _clean_text_expr('tobak_inskrivning')
    smoke_w30 = _clean_text_expr('tobak_vecka_30_32')
    sex_raw = _clean_text_expr('kon')
    ph_art = _float_expr('ph_navelartar')
    ph_ven = _float_expr('ph_navelven')
    birth_day = _date_expr('forlossningsdatum_fv1')
    birth_time_seconds = _int_expr('forlossningstid_fv1')
    birth_timestamp = _timestamp_from_date_and_seconds(birth_day, birth_time_seconds)
    labour_day = _date_expr('etablerade_varkar_datum')
    labour_time_seconds = _int_expr('etablerade_varkar_tid')
    labour_timestamp = _timestamp_from_date_and_seconds(labour_day, labour_time_seconds)
    mother_birth_date = "TRY_CAST(strptime(substr(regexp_replace(CAST(personnummer_mor AS VARCHAR), '[^0-9]', '', 'g'), 1, 8), '%Y%m%d') AS DATE)"
    death_day = _date_expr('avled_datum')

    mother_diag_col = 'moderns_diagnoser_rad'
    mother_proc_col = 'moderns_atgarder_rad'
    child_diag_col = 'barnets_diagnoser_rad'
    child_proc_col = 'barnets_atgarder_rad'

    gest_htn = _code_prefix_expr(mother_diag_col, ['O13', 'O16'])
    preeclampsia = _code_prefix_expr(mother_diag_col, ['O14', 'O15'])
    diabetes = _code_prefix_expr(mother_diag_col, ['O24'])
    uterine_rupture_diag = _code_exact_expr(mother_diag_col, ['O710', 'O711'])
    sepsis = _code_prefix_expr(mother_diag_col, ['A41'])
    placental_abruption = _code_exact_expr(mother_diag_col, ['O711'])
    heavy_bleeding = _code_prefix_expr(mother_diag_col, ['O46', 'O67'])
    cord_prolapse = _code_exact_expr(mother_diag_col, ['O690'])
    shoulder_dystocia_diag = _code_prefix_expr(mother_diag_col, ['O66'])
    labor_dystocia = _code_exact_expr(mother_diag_col, ['O620', 'O621', 'O628', 'O629'])

    oxytocin = _code_exact_expr(mother_proc_col, ['DT036', 'DT037'])
    uterine_rupture_proc = _code_exact_expr(mother_proc_col, ['MCC00'])

    severe_asphyxia_diag = _code_exact_expr(child_diag_col, ['P210', 'P808', 'P809'])
    meconium = _code_exact_expr(child_diag_col, ['P240'])
    shoulder_dystocia_child = _code_exact_expr(child_diag_col, ['P140', 'P141', 'P143', 'P148', 'P149'])
    hypoglycemia_treatment = _code_exact_expr(child_diag_col, ['P703', 'P704A', 'P704B', 'P708', 'P709'])
    severe_asphyxia_proc = _code_exact_expr(child_proc_col, ['DV034'])
    respirator_grav = (
        f"({_has_value_expr('ventilation_pa_mask_min')} OR "
        f"{_has_value_expr('intubation_min')} OR "
        f"{_has_value_expr('hjartmassage_min')})"
    )

    snq_glopnr = _normalized_glopnr_expr('glopnr')
    snq_highest_hie = _int_expr('"Högst HIE"')
    snq_hie = _bool_ja_nej_expr('"HIE"')
    snq_icd_col = '"ICD_kod"'
    snq_kva_col = '"KVÅ_kod"'

    con.execute(
        f"""
        CREATE TEMP TABLE reg AS
        SELECT
            row_number() OVER () AS reg_row,
            {_normalized_glopnr_expr('glopnr')} AS glopnr,
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
            CASE
                WHEN {labour_timestamp} IS NOT NULL AND {birth_timestamp} IS NOT NULL THEN
                    date_diff('second', {labour_timestamp}, {birth_timestamp})
                ELSE NULL
            END AS etablerade_varkar_seconds,
            {_clean_text_expr('forlossningsstart_basta_skattning')} AS forlossningsstart,
            {_clean_text_expr('forlossningsslut_basta_skattning')} AS forlossningsslut,
            {_int_expr('apgar_1_min')} AS apgar1,
            {_int_expr('apgar_5_min')} AS apgar5,
            {_int_expr('apgar_10_min')} AS apgar10,
            {_clean_text_expr('fodelseland')} AS fodelseland,
            {_clean_text_expr('utbildningsniva')} AS utbildningsniva,
            {_int_expr('para_mhv1')} AS para_mhv1,
            {_float_expr('langd_inskrivning_cm')} AS langd_inskrivning_cm,
            {_float_expr('bmi_inskrivning')} AS bmi_inskrivning,
            {smoke_pre} AS tobak_3_manader_fore_graviditet,
            {smoke_inskrivning} AS tobak_inskrivning,
            {smoke_w30} AS tobak_vecka_30_32,
            CASE
                WHEN {_smoke_detect_expr('tobak_3_manader_fore_graviditet')}
                  OR {_smoke_detect_expr('tobak_inskrivning')}
                  OR {_smoke_detect_expr('tobak_vecka_30_32')}
                THEN TRUE
                ELSE FALSE
            END AS is_smoker,
            {_bool_ja_nej_expr('diabetes_mellitus')} AS diabetes_mellitus,
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
            {_int_expr('alkohol_audit_poang')} AS alkohol_audit_poang,
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
            {respirator_grav} AS respiratorbehandling_gravniva
        FROM reg_raw
        WHERE personnummer_mor IS NOT NULL
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE snq AS
        WITH snq_pre AS (
            SELECT
                {snq_glopnr} AS glopnr,
                {snq_highest_hie} AS highest_hie,
                {snq_hie} AS hie,
                {_code_prefix_expr(snq_icd_col, ['P10', 'P52'], delimiter=';')} AS intracranial_haemorrhage,
                {_code_prefix_expr(snq_icd_col, ['P90'], delimiter=';')} AS neonatal_convulsions,
                {_code_prefix_expr(snq_icd_col, ['P23', 'P36', 'P392'], delimiter=';')} AS neonatal_sepsis_or_pneumonia,
                {_code_prefix_expr(snq_kva_col, ['DG021', 'DG022', 'DG0002'], delimiter=';')} AS respiratorbehandling
            FROM snq_raw
        )
        SELECT
            glopnr,
            MAX(highest_hie) AS highest_hie,
            BOOL_OR(hie) FILTER (WHERE hie IS NOT NULL) AS hie,
            BOOL_OR(intracranial_haemorrhage) FILTER (WHERE intracranial_haemorrhage IS NOT NULL) AS intracranial_haemorrhage,
            BOOL_OR(neonatal_convulsions) FILTER (WHERE neonatal_convulsions IS NOT NULL) AS neonatal_convulsions,
            BOOL_OR(neonatal_sepsis_or_pneumonia) FILTER (WHERE neonatal_sepsis_or_pneumonia IS NOT NULL) AS neonatal_sepsis_or_pneumonia,
            BOOL_OR(respiratorbehandling) FILTER (WHERE respiratorbehandling IS NOT NULL) AS respiratorbehandling
        FROM snq_pre
        WHERE glopnr IS NOT NULL
        GROUP BY glopnr
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE reg_clean AS
        SELECT
            reg_row,
            glopnr,
            birth_day,
            birth_time_seconds,
            birth_timestamp,
            maternal_age,
            etablerade_varkar_datum,
            etablerade_varkar_tid,
            etablerade_varkar_timestamp,
            etablerade_varkar_seconds,
            forlossningsstart,
            forlossningsslut,
            apgar1,
            apgar5,
            apgar10,
            fodelseland,
            utbildningsniva,
            para_mhv1,
            langd_inskrivning_cm,
            bmi_inskrivning,
            tobak_3_manader_fore_graviditet,
            tobak_inskrivning,
            tobak_vecka_30_32,
            is_smoker,
            diabetes_mellitus,
            child_sex,
            is_girl,
            alkohol_audit_poang,
            ph_navelartar,
            ph_navelven,
            ph_navel_below7,
            avled_datum,
            died_after_days,
            gestational_hypertension_without_significant_proteinuria,
            preeclampsia,
            gestational_or_pregestational_diabetes,
            uterine_rupture,
            sepsis,
            placental_abruption,
            heavy_vaginal_bleeding_before_or_during_delivery,
            umbilical_cord_prolapse,
            shoulder_dystocia,
            labor_dystocia,
            use_of_oxytocin,
            severe_birth_asphyxia,
            meconium_aspiration_syndrome,
            treatment_for_hypoglycemia,
            respiratorbehandling_gravniva,
            substr(reg_digits, 1, 8) || '-' || substr(reg_digits, 9, 4) AS PatientID
        FROM reg
        WHERE reg_digits IS NOT NULL
          AND length(reg_digits) >= 12
          AND apgar5 IS NOT NULL
          AND birth_day IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE reg_enriched AS
        SELECT
            r.reg_row,
            r.PatientID,
            r.birth_day,
            r.birth_time_seconds,
            r.birth_timestamp,
            r.maternal_age,
            r.etablerade_varkar_datum,
            r.etablerade_varkar_tid,
            r.etablerade_varkar_timestamp,
            r.etablerade_varkar_seconds,
            r.forlossningsstart,
            r.forlossningsslut,
            r.apgar1,
            r.apgar5,
            r.apgar10,
            r.fodelseland,
            r.utbildningsniva,
            r.para_mhv1,
            r.langd_inskrivning_cm,
            r.bmi_inskrivning,
            r.tobak_3_manader_fore_graviditet,
            r.tobak_inskrivning,
            r.tobak_vecka_30_32,
            r.is_smoker,
            r.diabetes_mellitus,
            r.child_sex,
            r.is_girl,
            r.alkohol_audit_poang,
            r.ph_navelartar,
            r.ph_navelven,
            r.ph_navel_below7,
            r.avled_datum,
            r.died_after_days,
            r.gestational_hypertension_without_significant_proteinuria,
            r.preeclampsia,
            r.gestational_or_pregestational_diabetes,
            r.uterine_rupture,
            r.sepsis,
            r.placental_abruption,
            r.heavy_vaginal_bleeding_before_or_during_delivery,
            r.umbilical_cord_prolapse,
            r.shoulder_dystocia,
            r.labor_dystocia,
            r.use_of_oxytocin,
            r.severe_birth_asphyxia,
            r.meconium_aspiration_syndrome,
            r.treatment_for_hypoglycemia,
            s.highest_hie,
            s.hie,
            s.intracranial_haemorrhage,
            s.neonatal_convulsions,
            s.neonatal_sepsis_or_pneumonia,
            CASE
                WHEN r.respiratorbehandling_gravniva THEN TRUE
                WHEN s.respiratorbehandling IS NOT NULL THEN s.respiratorbehandling
                ELSE NULL
            END AS respiratorbehandling
        FROM reg_clean r
        LEFT JOIN snq s USING (glopnr)
        """
    )

    con.execute(
        f"""
        CREATE VIEW s55 AS
        SELECT BabyID, PatientID, ctg_date
        FROM read_parquet('{safe_stage5_5}')
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
        """
        CREATE TEMP TABLE matches AS
        SELECT
            r.reg_row,
            r.PatientID,
            r.birth_day,
            r.birth_time_seconds,
            r.birth_timestamp,
            r.maternal_age,
            r.etablerade_varkar_datum,
            r.etablerade_varkar_tid,
            r.etablerade_varkar_timestamp,
            r.etablerade_varkar_seconds,
            r.forlossningsstart,
            r.forlossningsslut,
            r.apgar1,
            r.apgar5,
            r.apgar10,
            r.fodelseland,
            r.utbildningsniva,
            r.para_mhv1,
            r.langd_inskrivning_cm,
            r.bmi_inskrivning,
            r.tobak_3_manader_fore_graviditet,
            r.tobak_inskrivning,
            r.tobak_vecka_30_32,
            r.is_smoker,
            r.diabetes_mellitus,
            r.child_sex,
            r.is_girl,
            r.alkohol_audit_poang,
            r.ph_navelartar,
            r.ph_navelven,
            r.ph_navel_below7,
            r.avled_datum,
            r.died_after_days,
            r.gestational_hypertension_without_significant_proteinuria,
            r.preeclampsia,
            r.gestational_or_pregestational_diabetes,
            r.uterine_rupture,
            r.sepsis,
            r.placental_abruption,
            r.heavy_vaginal_bleeding_before_or_during_delivery,
            r.umbilical_cord_prolapse,
            r.shoulder_dystocia,
            r.labor_dystocia,
            r.use_of_oxytocin,
            r.severe_birth_asphyxia,
            r.meconium_aspiration_syndrome,
            r.treatment_for_hypoglycemia,
            r.highest_hie,
            r.hie,
            r.intracranial_haemorrhage,
            r.neonatal_convulsions,
            r.neonatal_sepsis_or_pneumonia,
            r.respiratorbehandling,
            m.BabyID,
            m.ctg_date
        FROM reg_enriched r
        JOIN map m
          ON r.PatientID = m.PatientID
         AND (m.ctg_date = r.birth_day OR m.ctg_date = r.birth_day - INTERVAL 1 DAY)
        """
    )

    multi_rows = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT reg_row, COUNT(*) AS cnt
            FROM matches
            GROUP BY reg_row
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    con.execute(
        """
        CREATE TEMP TABLE unique_matches AS
        WITH counted AS (
            SELECT reg_row, COUNT(*) AS cnt
            FROM matches
            GROUP BY reg_row
        )
        SELECT m.*
        FROM matches m
        JOIN counted c USING (reg_row)
        WHERE c.cnt = 1
        """
    )

    total_rows = con.execute('SELECT COUNT(*) FROM reg_raw').fetchone()[0]
    clean_rows = con.execute('SELECT COUNT(*) FROM reg_clean').fetchone()[0]
    match_rows = con.execute('SELECT COUNT(*) FROM unique_matches').fetchone()[0]

    print(f'Registry rows total: {total_rows}')
    print(f'Registry rows with valid apgar/birth_day: {clean_rows}')
    print(f'Matched rows: {match_rows}')
    if multi_rows:
        print(f'WARNING: {multi_rows} registry rows matched multiple BabyIDs and were dropped.')

    con.execute(
        f"""
        COPY (
            SELECT
                BabyID,
                birth_day,
                birth_time_seconds,
                birth_timestamp,
                maternal_age,
                etablerade_varkar_datum,
                etablerade_varkar_tid,
                etablerade_varkar_timestamp,
                etablerade_varkar_seconds,
                forlossningsstart,
                forlossningsslut,
                apgar1,
                apgar5,
                apgar10,
                fodelseland,
                utbildningsniva,
                para_mhv1,
                langd_inskrivning_cm,
                bmi_inskrivning,
                tobak_3_manader_fore_graviditet,
                tobak_inskrivning,
                tobak_vecka_30_32,
                is_smoker,
                diabetes_mellitus,
                child_sex,
                is_girl,
                alkohol_audit_poang,
                ph_navelartar,
                ph_navelven,
                ph_navel_below7,
                avled_datum,
                died_after_days,
                gestational_hypertension_without_significant_proteinuria,
                preeclampsia,
                gestational_or_pregestational_diabetes,
                uterine_rupture,
                sepsis,
                placental_abruption,
                heavy_vaginal_bleeding_before_or_during_delivery,
                umbilical_cord_prolapse,
                shoulder_dystocia,
                labor_dystocia,
                use_of_oxytocin,
                severe_birth_asphyxia,
                meconium_aspiration_syndrome,
                treatment_for_hypoglycemia,
                highest_hie,
                hie,
                intracranial_haemorrhage,
                neonatal_convulsions,
                neonatal_sepsis_or_pneumonia,
                respiratorbehandling
            FROM unique_matches
            ORDER BY BabyID
        ) TO '{str(registry_out).replace("'", "''")}'
        (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE matched_babies AS
        SELECT DISTINCT BabyID FROM unique_matches
        """
    )

    con.execute(
        f"""
        CREATE VIEW s6 AS
        SELECT * FROM read_parquet('{safe_stage6}/**/*.parquet')
        """
    )

    s6_cols = {row[0] for row in con.execute('DESCRIBE SELECT * FROM s6').fetchall()}
    keep_cols = ['BabyID', 'Timestamp', 'FHR', 'toco'] + [
        name for name in DEFAULT_STAGE2_EXTRA_COLUMNS if name in s6_cols
    ]
    ctg_select = ', '.join(f's6.{name}' for name in keep_cols)

    con.execute(
        f"""
        COPY (
            SELECT {ctg_select}
            FROM s6
            JOIN matched_babies mb USING (BabyID)
        ) TO '{str(ctg_out).replace("'", "''")}'
        (FORMAT PARQUET)
        """
    )

    print(f'Wrote registry CSV: {registry_out}')
    print(f'Wrote CTG parquet: {ctg_out}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage 7 registry matching and anonymized output.')
    parser.add_argument('--registry-csv', type=str, default=DEFAULT_PATIENT_CSV)
    parser.add_argument('--snq-file', type=str, default=DEFAULT_SNQ_FILE)
    parser.add_argument('--stage5-5', type=str, default=DEFAULT_STAGE5_5_OUTPUT_FILE)
    parser.add_argument('--stage6', type=str, default=DEFAULT_STAGE6_DIR)
    parser.add_argument('--registry-out', type=str, default=DEFAULT_STAGE7_REGISTRY_CSV)
    parser.add_argument('--ctg-out', type=str, default=DEFAULT_STAGE7_CTG_PARQUET)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()

    registry_match(
        registry_csv=args.registry_csv,
        snq_file=args.snq_file,
        stage5_5_file=args.stage5_5,
        stage6_dir=args.stage6,
        registry_out=args.registry_out,
        ctg_out=args.ctg_out,
        show_progress=not args.no_progress,
    )


if __name__ == '__main__':
    main()
