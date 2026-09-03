# Stage 7 Registry Data Dictionary

## Overview

This document describes the variables currently written to the final (Stage 7) registry file, `registry.csv`, in the CTG preprocessing pipeline. The purpose of this file is to document the origin, meaning, and derivation of each registry variable that is linked to the anonymized CTG data through `BabyID`.

The Stage 7 output consists of one row per matched pregnancy/child episode. Matching is performed between the CTG dataset and registry data using the maternal personal number and delivery date information before anonymization. The resulting `registry.csv` is intended to serve as the metadata table accompanying the final anonymized CTG parquet file.

In the descriptions below:

- `gravniva.csv` refers to the main obstetric registry source (Swedish Pregnancy Register, SPR export; one row per live-born singleton, births 2015-2022).
- `fv1_moderns_diagnoser.csv`, `barn_barnets_diagnoser_forsta_28_dagarna.csv` and `barn_barnets_atgarder_forsta_28_dagarna.csv` are the dated SPR long tables (one row per ICD-10/KVÅ code). Their codes are unioned with the collapsed `*_rad` code strings of `gravniva.csv` before any code flag below is derived, and they are exported anonymized (see "Anonymized long tables").
- `SNQ data.xlsx` refers to the supplementary neonatal intensive care registry source (Swedish Neonatal Quality Register; one row per child admitted to neonatal care).
- Variables marked as derived are calculated from one or more raw source variables rather than copied directly.
- Prevalences quoted below come from the first (2025) cohort; new variables say "see cohort report". `cohort_report.py` recomputes them after every pipeline run.

## Identifier

### `BabyID`
- Type: string
- Source: CTG preprocessing pipeline
- Description: An anonymized identifier representing one pregnancy/child episode.
- Derivation: Created earlier in the CTG pipeline and used as the key linking `registry.csv` to the final anonymized CTG parquet file.

### `MotherID`
- Type: string (16 hex characters)
- Source: CTG preprocessing pipeline (stage 7)
- Description: Pseudonymous identifier of the mother, identical for all her pregnancies in the dataset.
- Derivation: `sha256(salt | "mother" | PatientID)` truncated to 16 characters, with the same secret salt as `BabyID`. Use it for mother-level train/validation/test splits and leakage exclusions. Stage 8 also writes `mothers.csv` (`BabyID`, `MotherID`) for every pregnancy in the CTG data, including the pretraining-only ones that have no registry row.

## Birth timing and maternal age

### `birth_day`
- Type: date
- Source: `gravniva.csv`
- Raw variable: `forlossningsdatum_fv1`
- Description: Date of birth/delivery.

### `birth_time_seconds`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `forlossningstid_fv1`
- Description: Time of birth expressed as seconds since midnight.

### `birth_timestamp`
- Type: timestamp
- Source: derived from `gravniva.csv`
- Description: Full birth date-time.
- Derivation: Calculated as `birth_day + birth_time_seconds`.

### `maternal_age`
- Type: integer
- Source: derived from `gravniva.csv`
- Raw variables: `personnummer_mor`, `forlossningsdatum_fv1`
- Description: Maternal age at delivery, expressed in completed years. Ranges from 15 to 54.
- Derivation: Calculated from the mother’s birth date encoded in `personnummer_mor` and the delivery date. Only year precision is retained.
- ML-use: Used as input
- Prevalance: 100%

## Labour timing

### `etablerade_varkar_datum`
- Type: date
- Source: `gravniva.csv`
- Raw variable: `etablerade_varkar_datum`
- Description: Date of established labour.

### `etablerade_varkar_tid`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `etablerade_varkar_tid`
- Description: Time of established labour expressed as seconds since midnight.

### `etablerade_varkar_timestamp`
- Type: timestamp
- Source: derived from `gravniva.csv`
- Description: Full date-time of established labour.
- Derivation: Calculated as `etablerade_varkar_datum + etablerade_varkar_tid`.

### `etablerade_varkar_seconds`
- Type: integer
- Source: derived from `gravniva.csv`
- Description: Duration in seconds from established labour to birth. Ranges from 120 to 880260.
- Derivation: Calculated as the difference between `etablerade_varkar_timestamp` and `birth_timestamp`.
- ML-use: Used as input, potentially refined through CTG-timestamps
- Prevalance: 66%

## Labour start and end classification

### `forlossningsstart`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `forlossningsstart_basta_skattning`
- Description: Best-estimate classification of how labour started.
- Handling: Saved as raw text. Options: Spontan Start / Induktion / Kejsarsnitt före värkdebut
- ML-use: Used as input, could be used as optional for intervention->outcome prediction
- Prevalance: Spontan start 71.52%, Induktion 26.71%, Kejsarsnitt före värkdebut 1.77%

### `forlossningsslut`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `forlossningsslut_basta_skattning`
- Description: Best-estimate classification of how labour ended.
- Handling: Saved as raw text.
- ML-use: Not used now, could be used as optional for intervention->outcome prediction
- Prevalance: Vaginalt, ej instrumentellt 77.82%, Instrumentell vaginal förlossning 6.61%, Akut kejsarsnitt 6.61%


## Apgar scores

### `apgar1`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `apgar_1_min`
- Description: Apgar score at 1 minute. (Ranges from 0 to 10)
- ML-use: Used as output
- Prevalance: 99.95%

### `apgar5`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `apgar_5_min`
- Description: Apgar score at 5 minutes. (Ranges from 0 to 10)
- Note: Rows without `apgar5` are excluded before Stage 7 matching.
- ML-use: Used as output
- Prevalance: 100%

### `apgar10`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `apgar_10_min`
- Description: Apgar score at 10 minutes. (Ranges from 0 to 10)
- ML-use: Used as output
- Prevalance: 99.98%

### `gestational_days`
- Type: integer
- Source: derived from `gravniva.csv`
- Raw variables: `gl_v_barn`, `gl_d_barn`
- Description: Gestational age at birth expressed as completed pregnancy days.
- Derivation: Calculated as gestational weeks multiplied by 7 plus gestational days: `gl_v_barn * 7 + gl_d_barn`.
- ML-use: Used as input

## Maternal background variables

### `fodelseland`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `fodelseland`
- Description: Mother’s country of birth. 63.3% 'Sverige', after is 2.08% 'Irak', 1.62 'Pålen' etc
- Handling: Saved as raw text.
- ML-use: Used as input, in the future could be refined to include category of country.
- Prevalance: 92.7%

### `utbildningsniva`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `utbildningsniva`
- Description: Maternal education level.
- Handling: Saved as raw text.
- ML-use: Used as input
- Prevalance: Universitet eller högskola (eller motsvarande) 58.92%, Upp till och med gymnasium (eller motsvarande) 21.52%, Vet ej 7.66%, Grundskola (eller motsvarande) 3.75%, Ingen eller skolgång kortare än 9 år 1.02%, missing (no data) 7.1%

### `para_mhv1`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `para_mhv1`
- Description: Number of previous children. Ranges from 0 to 11. Most are 0 (or 1-2).
- ML-use: Used as input
- Prevalance: 99.95%

### `langd_inskrivning_cm`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `langd_inskrivning_cm`
- Description: Maternal height in centimetres at registration. Ranges from 0 (obvious error, should be cleaned) to 192.
- ML-use: Used as input
- Prevalance: 98.3%

### `bmi_inskrivning`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `bmi_inskrivning`
- Description: Maternal body mass index at registration. Ranges from 14.88 to 61.73.
- ML-use: Used as input
- Prevalance: 95.6%

### `previous_c_section`
- Type: boolean
- Source: derived from `gravniva.csv`
- Raw variable: `tidigare_sectio`
- Description: Indicator of whether the mother has had a previous caesarean section before the current delivery.
- Derivation: Set to `True` when `tidigare_sectio` is `Ja`; all other values, including `Nej` and missing/blank values, are set to `False`.
- ML-use: Used as input

## Tobacco use

### `tobak_3_manader_fore_graviditet`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `tobak_3_manader_fore_graviditet`
- Description: Smoking status three months before pregnancy.
- Handling: Saved as raw text.
- ML-use: Used as input
- Prevalance: 99.35% non-missing. Nej 86.54%, 1-9cigg/dag 5.59%, 10 eller fler cigg/dag 3.92%, Ej angivet 3.3%

### `tobak_inskrivning`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `tobak_inskrivning`
- Description: Smoking status at registration.
- Handling: Saved as raw text.
- ML-use: Used as input
- Prevalance: 99.35% non-missing. Nej 94.31%, Ej angivet 2.28%, 1-9 cigg/dag 2.24%, 10 eller fler cigg/dag 0.53%

### `tobak_vecka_30_32`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `tobak_vecka_30_32`
- Description: Smoking status during gestational week 30 to 32.
- Handling: Saved as raw text.
- ML-use: Used as input
- Prevalance: 84.8% non-missing. Nej 83:07% Ej angivet 2.28%, 1-9 cigg/dag 2.24%, 10 eller fler cigg/dag 0.53%

### `is_smoker`
- Type: boolean
- Source: derived from `gravniva.csv`
- Description: Indicator of maternal smoking.
- Derivation: Set to `True` if any of the three tobacco variables contains a smoking quantity entry such as `1–9 cigg/dag` or `10 eller fler cigg/dag`. Otherwise set to `False`.
- ML-use: Used as input
- Prevalance: True 9.68%, False 90.32%

## Maternal clinical variables

### `diabetes_mellitus`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `diabetes_mellitus`
- Description: Maternal diabetes mellitus indicator.
- Derivation: `Ja` is mapped to `True`, `Nej` to `False`, other values remain missing.
- ML-use: Used as input, but risky due to very few True samples.
- Prevalance: 97.6% non-missing. False 96.96%, True 0.65%.

### `alkohol_audit_poang`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `alkohol_audit_poang`
- Description: Maternal alcohol use score. Ranges from 0 to 36. Most are very low (0-3).
- ML-use: Used as input
- Prevalance: 88.5%

## Child sex

### `child_sex`
- Type: string
- Source: derived from `gravniva.csv`
- Raw variable: `kon`
- Description: Child sex.
- Derivation: `Flicka` is retained as `Flicka`. All other non-missing values are stored as `Pojke`. This includes the rare `Okänt` category, which is intentionally reassigned to `Pojke` according to the project rule.

### `is_girl`
- Type: boolean
- Source: derived from `gravniva.csv`
- Description: Boolean representation of child sex.
- Derivation: `True` if `child_sex = Flicka`, otherwise `False` for non-missing values.
- ML-use: Used as input
- Prevalance: False 52.39%, True 47.61%

## Cord blood pH and neonatal death

### `ph_navelartar`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `ph_navelartar`
- Description: Umbilical artery pH. Ranges from 6.58 to 8.23. 
- ML-use: Used as input
- Prevalance: 60.1%

### `ph_navelven`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `ph_navelven`
- Description: Umbilical vein pH. Ranges from 6.19 to 7.83. 95% of cases are above 7.17.
- ML-use: Used as input
- Prevalance: 69.5%

### `ph_navel_below7`
- Type: boolean
- Source: derived from `gravniva.csv`
- Description: Indicator for low umbilical pH.
- Derivation: Evaluated first on `ph_navelartar`; if arterial pH is missing, `ph_navelven` is used instead. Set to `True` when the available pH is below 7, otherwise `False`. Missing if neither pH is available.
- ML-use: Used as input, but risky due to very few true samples.
- Prevalance: 73.2% non-missing. False 72.59%, True 0.65%

### `avled_datum`
- Type: date
- Source: `gravniva.csv`
- Raw variable: `avled_datum`
- Description: Recorded date of neonatal death, when present.
- Prevalance: 0.07%

### `died_after_days`
- Type: integer
- Source: derived from `gravniva.csv`
- Description: Number of days from birth to death within the neonatal period.
- Derivation: Calculated from `avled_datum - birth_day`, but only retained if the interval is between 0 and 28 days inclusive.
- Prevalance: 0.06%

## Diagnosis and intervention indicators from maternal registry code fields

The following variables are derived from comma-separated code lists in `gravniva.csv`. Prefix matching means that a requested code such as `O13` will match more specific codes beginning with `O13`.

### `gestational_hypertension_without_significant_proteinuria`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Maternal gestational hypertension without significant proteinuria.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O13` or `O16`.
- ML-use: Used as input
- Prevalance: 3.74% True

### `preeclampsia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Maternal preeclampsia/eclampsia indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O14` or `O15`.
- ML-use: Used as input
- Prevalance: 3.35% True

### `gestational_or_pregestational_diabetes`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Maternal gestational or pregestational diabetes indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O24`.
- ML-use: Used as input
- Prevalance: 3.75% True

### `uterine_rupture`
- Type: boolean
- Source: `gravniva.csv`
- Raw variables: `moderns_diagnoser_rad`, `moderns_atgarder_rad`
- Description: Uterine rupture indicator.
- Derivation: Set to `True` if maternal diagnosis contains `O710` or `O711`, or if maternal intervention contains `MCC00`.
- ML-use: Not used, due to too few True samples
- Prevalance: 0.15% True

### `sepsis`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Maternal sepsis indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `A41`.
- ML-use: Would be used as output, but not used due to too few True samples
- Prevalance: 0.01% (2 cases total)

### `placental_abruption`
- Type: boolean
- Source: `gravniva.csv` (+ `fv1_moderns_diagnoser.csv`)
- Raw variable: `moderns_diagnoser_rad`
- Description: Placental abruption (abruptio placentae) indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O45`. (Before September 2026 this used the exact code `O711`, which is uterine rupture during labour, so the old column measured uterine rupture rather than abruption; `uterine_rupture` still covers `O711`.)
- ML-use: Not used, due to too few True samples
- Prevalance: see cohort report

### `heavy_vaginal_bleeding_before_or_during_delivery`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Heavy vaginal bleeding before or during delivery.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O46` or `O67`.
- ML-use: Used as input
- Prevalance: 3.95%

### `umbilical_cord_prolapse`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Umbilical cord prolapse indicator.
- Derivation: Set to `True` if exact code `O690` is present.
- ML-use: Not used, due to too few True samples
- Prevalance: 0.1%

### `shoulder_dystocia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variables: `moderns_diagnoser_rad`, `barnets_diagnoser_rad`
- Description: Shoulder dystocia indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O66`, or if child diagnosis contains any of `P140`, `P141`, `P143`, `P148`, or `P149`.
- ML-use: Used as output
- Prevalance: 1.13%

### `labor_dystocia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Labour dystocia indicator.
- Derivation: Set to `True` if any of the exact codes `O620`, `O621`, `O628`, or `O629` are present.
- ML-use: Used as input, potentially irrelevant
- Prevalance: 17.1%

### `use_of_oxytocin`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_atgarder_rad`
- Description: Use of oxytocin during labour/delivery.
- Derivation: Set to `True` if exact intervention code `DT036` or `DT037` is present.
- ML-use: Used as input
- Prevalance: 52.57%

## Diagnosis and intervention indicators from child registry code fields

### `severe_birth_asphyxia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variables: `barnets_diagnoser_rad`, `barnets_atgarder_rad`
- Description: Severe birth asphyxia indicator.
- Derivation: Set to `True` if child diagnosis contains `P210`, `P808`, or `P809`, or if child intervention contains `DV034`.
- ML-use: Would be used as output, but not used due to too few True samples
- Prevalance: 0.1%

### `meconium_aspiration_syndrome`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `barnets_diagnoser_rad`
- Description: Meconium aspiration syndrome indicator.
- Derivation: Set to `True` if exact code `P240` is present.
- ML-use: Would be used as output, but not used due to too few True samples
- Prevalance: 0.14%

### `treatment_for_hypoglycemia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `barnets_diagnoser_rad`
- Description: Treatment for neonatal hypoglycaemia indicator.
- Derivation: Set to `True` if any of `P703`, `P704A`, `P704B`, `P708`, or `P709` is present.
- ML-use: Used as output
- Prevalance: 2.23%

### `neonatal_anemia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `barnets_diagnoser_rad`
- Description: Neonatal anaemia diagnosis indicator.
- Derivation: Set to `True` if the child diagnosis code list contains any exact code `P612`, `P613`, or `P614`; otherwise set to `False`.
- ML-use: Used as output

## SNQ-derived variables

SNQ variables are obtained by linking `gravniva.csv` to `SNQ data.xlsx` using `glopnr`. Because many children are not represented in SNQ, missing values are common for these variables and should not automatically be interpreted as negative findings.

### `highest_hie`
- Type: integer
- Source: `SNQ data.xlsx`
- Raw variable: `Högst HIE`
- Description: Highest recorded grade of hypoxic-ischaemic encephalopathy. Ranges from 1-3.
- ML-use: Would be used as output, but is not used due to too few True samples
- Prevalance: 0.22%

### `hie`
- Type: boolean
- Source: `SNQ data.xlsx`
- Raw variable: `HIE`
- Description: Indicator of hypoxic-ischaemic encephalopathy.
- Derivation: `Ja` is mapped to `True`, `Nej` to `False`.
- ML-use: Would be used as output, but is not used due to too few True samples
- Prevalance: 8.43 False, 0.22% True (rest is no data)

### `intracranial_haemorrhage`
- Type: boolean
- Source: `SNQ data.xlsx`
- Raw variable: `ICD_kod`
- Description: Indicator of intracranial haemorrhage.
- Derivation: Set to `True` if any semicolon-separated SNQ ICD code starts with `P10` or `P52`.
- ML-use: Would be used as output, but is not used due to too few True samples
- Prevalance: 0.14%

### `neonatal_convulsions`
- Type: boolean
- Source: `SNQ data.xlsx`
- Raw variable: `ICD_kod`
- Description: Indicator of neonatal convulsions.
- Derivation: Set to `True` if any SNQ ICD code starts with `P90`.
- ML-use: Would be used as output, but is not used due to too few True samples
- Prevalance: 0.21%

### `neonatal_sepsis_or_pneumonia`
- Type: boolean
- Source: `SNQ data.xlsx`
- Raw variable: `ICD_kod`
- Description: Indicator of neonatal sepsis or pneumonia.
- Derivation: Set to `True` if any SNQ ICD code starts with `P23`, `P36`, or `P392`.
- ML-use: Used as output, but risky due to very few true samples
- Prevalance: 0.48%

### `respiratorbehandling`
- Type: boolean
- Source: `SNQ data.xlsx` and `gravniva.csv`
- Raw variables: `KVÅ_kod`, `ventilation_pa_mask_min`, `intubation_min`, `hjartmassage_min`
- Description: Indicator of ventilator treatment or advanced neonatal resuscitation support.
- Derivation: Set to `True` if SNQ `KVÅ_kod` contains any code starting with `DG021`, `DG022`, or `DG0002`, or if any of the gravniva variables `ventilation_pa_mask_min`, `intubation_min`, or `hjartmassage_min` is non-empty.
- ML-use: Used as output
- Prevalance: False 7.28%, True 2.62% (rest is missing and can be assumed to be False)


## Time shifting (stage 8)

The delivered files in `stage_8_timeshift/` are time-shifted copies of the stage 7 outputs: `birth_day`, `birth_timestamp`, `etablerade_varkar_datum`, `etablerade_varkar_timestamp`, `avled_datum` and every CTG `Timestamp` are moved by the same per-pregnancy number of whole days (up to about a year either way, with sibling intervals jittered by 10-20 %). Time-of-day fields (`birth_time_seconds`, `etablerade_varkar_tid`), all `*_seconds*` variables, `days_to_discharge`, `died_after_days`, `maternal_age` and the `day_offset` of the long tables are unaffected. Do not use the shifted dates for seasonality or calendar-time analyses; day of week is not preserved either.

## Notes on missing values

Missing values in `registry.csv` generally reflect one of the following:

- the source registry variable was not recorded,
- the information was not applicable,
- the child was not represented in the SNQ registry,
- or a derived field could not be calculated because one or more required source variables were missing.

These missing values should be handled explicitly in downstream statistical analysis or machine learning workflows rather than being automatically converted to zero or `False`.

## Notes on code parsing

- In `gravniva.csv`, diagnosis and intervention code fields are treated as comma-separated lists; the long-table codes are appended to the same list.
- In `SNQ data.xlsx`, code fields are treated as semicolon-separated lists.
- Codes are upper-cased and whitespace and dots are removed before matching (`O14.1` and `O141` are the same code).
- Prefix matching is used where specified, meaning that broader codes such as `O13` match more specific codes beginning with `O13`.
- Exact matching is used where a precise code was explicitly requested.
- If a source column is missing from an export, the derived variable is written as missing and a warning is printed; Stage 7 does not fail.

## Delivery mode, labour course and neonatal condition (added September 2026)

All from `gravniva.csv` unless stated. Booleans derived from `Ja`/`Nej` fields are missing when the field is empty or `us`/`Vet ej`. "Seconds before birth" variables are `birth_timestamp - event timestamp` (positive when the event precedes the birth, negative after), computed from the date and time-of-day fields of the event.

| Variable | Type | Raw variable(s) | Description / derivation |
|---|---|---|---|
| `emergency_c_section` | boolean | `forlossningsslut_basta_skattning` | `Akut kejsarsnitt` |
| `planned_c_section` | boolean | `forlossningsslut_basta_skattning` | `Planerat kejsarsnitt` |
| `instrumental_vaginal_delivery` | boolean | `forlossningsslut_basta_skattning` | `Instrumentell vaginal förlossning` (vacuum/forceps) |
| `c_section_urgency` | text | `indikation` | `Urakut` / `Akut` / `Elektiv`; missing for vaginal births |
| `induced_labour` | boolean | `forlossningsstart_basta_skattning` | `Induktion` |
| `oxytocin_under_forlossning` | boolean | `oxytocin_under_forlossning` | registry field, only recorded for induced labours; see `use_of_oxytocin` for the procedure-code version |
| `epidural` | boolean | `smartlindring_epidural` | recorded for a subset of births only |
| `presentation` | text | `presentation` | fetal presentation |
| `breech_presentation` | boolean | `presentation` | `Sätes- eller fotbjudning` |
| `robsongrupp` | text | `robsongrupp` | Robson ten-group classification |
| `labour_onset_seconds_before_birth` | integer | `varkar_borjade_datum/tid` | onset of contractions |
| `membrane_rupture_seconds_before_birth` | integer | `vattenavgang_datum/tid` | rupture of membranes |
| `amniotomy_seconds_before_birth` | integer | `amniotomi_datum/tid` | amniotomy |
| `second_stage_seconds_before_birth` | integer | `krystvarkar_datum/tid` | start of pushing (second stage) |
| `c_section_start_seconds_before_birth` | integer | `sectio_start_datum/tid` | start of caesarean section |
| `c_section_duration_seconds` | integer | `sectio_start/slut` | caesarean duration |
| `total_blodning_ml` | integer | `total_blodning_ml` | total maternal blood loss (ml) |
| `ctg_intagningstest` | text | `ctg_intagningstest` | admission CTG classification as recorded in the registry: `Normal` / `Ej normal` / `Ej utförd` |
| `ctg_admission_test_abnormal` | boolean | `ctg_intagningstest` | `Ej normal` = True, `Normal` = False, otherwise missing |
| `ivf_graviditet` | boolean | `ivf_graviditet` | IVF pregnancy |
| `kronisk_hypertoni` | boolean | `kronisk_hypertoni` | chronic hypertension |
| `graviditetsdiabetes_diagnos` | boolean | `diagnosen_graviditetsdiabetes_stalld` | any `Ja...` = True, `Nej` = False, `Vet ej` missing |
| `birth_weight_g` | integer | `fodelsevikt_g` | birth weight |
| `birth_weight_deviation_perc` | float | `vikt_avvikelse_perc` | deviation from expected weight (%) |
| `be_navelartar`, `be_navelven` | float | `be_navel*_mmol_l` | cord base excess (mmol/l) |
| `pco2_navelartar`, `po2_navelartar` | float | `pco2/po2_navelartar_kpa` | cord arterial gases (kPa) |
| `ph_navel_below705` | boolean | `ph_navelartar`, `ph_navelven` | arterial pH < 7.05 when available, else venous |
| `metabolic_acidosis` | boolean | `ph_navelartar`, `be_navelartar_mmol_l` | arterial pH < 7.05 and BE <= -12; missing unless both are recorded |
| `apgar5_below7`, `apgar10_below7` | boolean | `apgar_5_min`, `apgar_10_min` | Apgar < 7 |
| `ventilation_pa_mask_min`, `intubation_min`, `hjartmassage_min` | integer | same | minutes of mask ventilation / intubation / chest compressions at birth (missing = none recorded) |
| `acidoskorrektion` | boolean | `acidoskorrektion` | acidosis correction given |
| `respiratorbehandling_gravniva` | boolean | the three `*_min` fields | any resuscitation minutes recorded |
| `discharged_home` | boolean | `utskriven_till_hemmet` | |
| `days_to_discharge` | integer | `utskrivning_datum` | days from birth to maternal discharge |

## Additional maternal diagnosis flags (ICD-10, gravniva + long table)

Prefix matches on the maternal diagnosis codes unless noted.

| Variable | Codes | Meaning |
|---|---|---|
| `fetal_distress_in_labour` | O68 | labour and delivery complicated by fetal stress (distress) |
| `maternal_care_for_fetal_problems` | O36 | maternal care for known or suspected fetal problems |
| `signs_of_fetal_hypoxia_antenatal` | O363 | maternal care for signs of fetal hypoxia |
| `fetal_growth_restriction` | O365 | maternal care for poor fetal growth |
| `chorioamnionitis` | O411 | infection of the amniotic sac |
| `oligohydramnios` | O410 | |
| `polyhydramnios` | O40 | |
| `prelabour_rupture_of_membranes` | O42 | |
| `preterm_labour` | O60 | |
| `prolonged_pregnancy` | O48 | |
| `failed_induction` | O61 | |
| `prolonged_labour` | O63 | |
| `obstructed_labour` | O64, O65, O66 | malposition, pelvic abnormality, other obstruction (O66 includes shoulder dystocia) |
| `umbilical_cord_complications` | O69 | |
| `hypertensive_disorder_any` | O10-O16 | any hypertensive disorder of pregnancy |
| `placenta_previa` | O44 | |
| `postpartum_haemorrhage` | O72 | |
| `intrapartum_fever` | O752 | pyrexia during labour |

## Additional child diagnosis flags (ICD-10, first 28 days)

| Variable | Codes | Meaning |
|---|---|---|
| `intrauterine_hypoxia` | P20 | |
| `birth_asphyxia_any` | P21 | any birth asphyxia (P210 severe, P211 mild/moderate, P219 unspecified) |
| `mild_or_moderate_birth_asphyxia` | P211 | |
| `respiratory_distress_newborn` | P22 | |
| `neonatal_aspiration_syndromes` | P24 | includes meconium aspiration P240 |
| `neonatal_convulsions_icd` | P90 | from SPR codes (compare `neonatal_convulsions`, from SNQ codes) |
| `hie_icd` | P916 | hypoxic-ischaemic encephalopathy (compare `hie` from SNQ) |
| `cerebral_disturbance_newborn` | P91 | other disturbances of cerebral status of newborn |
| `intracranial_haemorrhage_icd` | P10, P52 | compare `intracranial_haemorrhage` from SNQ codes |
| `neonatal_infection_icd` | P23, P36, P39 | congenital pneumonia, bacterial sepsis, other infections |
| `neonatal_hypoglycaemia_icd` | P70 | |
| `birth_injury` | P10-P15 | |
| `congenital_malformation` | Q | any malformation code; mainly for exclusion |

## SNQ variables added September 2026

All from `SNQ data.xlsx`, joined on `glopnr`. Missing for every child not admitted to neonatal care. `Ja`/`Nej` fields become booleans; `us` becomes missing. Column names in the export are given as raw variables.

| Variable | Type | Raw variable | Notes |
|---|---|---|---|
| `neonatal_care_admission` | boolean | presence of an SNQ row | True/False for every matched child (False = no SNQ record). SNQ covers essentially all Swedish neonatal units, so this approximates admission to neonatal care |
| `snq_hypothermia_treatment` | boolean | `Behandlad med hypotermi` | therapeutic hypothermia (cooling) |
| `snq_seizures` | boolean | `Kramper` | |
| `snq_antiepileptic_treatment` | boolean | `AntiEp_beh under vtf` | |
| `snq_eeg_monitoring` | boolean | `EEG/aEEG övervakning` | |
| `snq_cns_haemorrhage` | boolean | `CNS - blödning` | |
| `snq_cns_infarct` | boolean | `Fokal/multifokal CNS infarkt` | |
| `snq_pvl` | boolean | `PVL (med cystor)` | periventricular leukomalacia |
| `snq_highest_ivh` | integer | `Högst IVH` | 0-4 (grade); `88 Ej undersökt` = missing |
| `snq_resuscitation` | boolean | `HLR-åtgärder` | `1`/`2` = True, `0 Nej (vitalt barn)` = False, palliation/`us` missing |
| `snq_resuscitation_over_10min` | boolean | `HLR-åtgärder` | `2 Ja (>= 10 min)` |
| `snq_hlr_extra_oxygen`, `snq_hlr_ventilation_mask`, `snq_hlr_cpap`, `snq_hlr_intubation`, `snq_hlr_chest_compressions`, `snq_hlr_adrenaline` | boolean | `HLR_*` | individual resuscitation measures |
| `snq_cpap`, `snq_high_flow` | boolean | `CPAP`, `Högflödesgrimma` | respiratory support during care |
| `snq_ventilator_conventional`, `snq_ventilator_hfv`, `snq_ventilator_nava` | boolean | `Resp konv`, `Resp HFV`, `Resp NAVA` | mechanical ventilation modes |
| `snq_nas`, `snq_pas` | boolean | `NAS`, `PAS` | SNQ respiratory diagnoses as labelled in the export (neonatal respiratory disturbance / pulmonary adaptation disturbance); consult the SNQ manual before use |
| `snq_mas`, `snq_rds`, `snq_pphn`, `snq_pneumothorax`, `snq_bpd` | boolean | same | meconium aspiration, RDS, persistent pulmonary hypertension, pneumothorax, bronchopulmonary dysplasia |
| `snq_infection` | boolean | `Barn med infektion` | |
| `snq_early_culture_verified_sepsis` | boolean | `Tidig bakt. sepsis, odlingsverif. (antal)` | count > 0 |
| `snq_hypoglycaemia` | boolean | `Hypoglukemi (<2,6 efter 3 tim)` | |
| `snq_inotropic_support` | boolean | `Inotroptstöd` | |
| `snq_erythrocyte_transfusion` | boolean | `EryTransf` | |
| `snq_malformation_or_chromosomal` | boolean | `Missb./kromosom avv.` | |
| `snq_died` | boolean | `Avliden enl. SNQ` | death during neonatal care |
| `snq_age_at_death_days` | integer | `Ålder vid dödsfall SNQ (dagar)` | |
| `snq_death_cause_perinatal_asphyxia` | boolean | `Dödors_Perinatal asfyxi` | only filled for deaths |
| `snq_died_death_register` | boolean | `Avliden enl. DOR` | death per the national cause-of-death register; only `Ja` is recorded, so False never occurs |
| `snq_admission_age_days` | integer | `1:a inskrivning, ålder (dagar)` | age at first admission |
| `snq_admissions` | integer | `Antal Vtf` | number of care episodes |
| `snq_care_days_inpatient`, `snq_care_days_neonatal` | integer | `Vårdtid, inneliggande`, `Vårdtid, neonatologi` | |
| `snq_apgar1`, `snq_apgar5`, `snq_apgar10` | integer | `Apgar1m/5m/10m` | SNQ's own Apgar record |
| `snq_ph_navelartar`, `snq_be_navelartar`, `snq_ph_navelven`, `snq_be_navelven` | float | `Artär pH/BE`, `Ven pH/BE` | cord gases as recorded by SNQ |
| `snq_postnatal_ph`, `snq_postnatal_be` | float | `Post pH`, `Post BE` | first postnatal blood gas |
| `snq_gestational_weeks`, `snq_birth_weight_g`, `snq_birth_weight_zscore` | numeric | `Grav.längd (v)`, `Födelsevikt`, `ZScore` | |
| `snq_iugr`, `snq_preeclampsia`, `snq_chorioamnionitis`, `snq_abruption_or_bleeding` | boolean | `Intrauterin tillväxthämning`, `Preeklampsi/Eklampsi`, `Amnionit`, `Ablatio/Blödning` | maternal conditions as recorded by SNQ |

## Composite outcome

### `severe_neonatal_outcome`
- Type: boolean (never missing for matched rows)
- Source: derived
- Derivation: True if any of: `apgar5 < 7`; arterial cord pH < 7.00; `metabolic_acidosis`; `hie_icd`; `severe_birth_asphyxia`; neonatal death (`died_after_days` recorded); `intubation_min` recorded; SNQ `hie`; `snq_hypothermia_treatment`; `snq_seizures`; SNQ `neonatal_convulsions`; `snq_died`; `snq_resuscitation_over_10min`; `snq_hlr_intubation`. SNQ components missing because the child was not admitted count as False.
- ML-use: intended primary output for intrapartum-hypoxia models; the components are available separately for ablations.

## Anonymized long tables

`mother_diagnoses.csv`, `child_diagnoses.csv` and `child_procedures.csv` hold the dated SPR long tables restricted to matched babies, with columns `BabyID`, `day_offset` (days from birth; negative = before birth) and `code` (ICD-10-SE diagnosis or KVÅ procedure code, as exported). Maternal diagnoses belong to the same pregnancy (`glopnr` is pregnancy-specific), so a negative offset means an antenatal diagnosis and 0 or -1 a diagnosis recorded around delivery. They are the complete code lists; the flags above are conveniences derived from them.
