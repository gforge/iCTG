# Stage 7 Registry Data Dictionary

## Overview

This document describes the variables currently written to the final Stage 7 registry file, `registry.csv`, in the CTG preprocessing pipeline. The purpose of this file is to document the origin, meaning, and derivation of each registry variable that is linked to the anonymized CTG data through `BabyID`.

The Stage 7 output consists of one row per matched pregnancy/child episode. Matching is performed between the CTG dataset and registry data using the maternal personal number and delivery date information. The resulting `registry.csv` is intended to serve as the metadata table accompanying the final anonymized CTG parquet file.

In the descriptions below:

- `gravniva.csv` refers to the main obstetric registry source.
- `SNQ data.xlsx` refers to the supplementary neonatal intensive care registry source.
- Variables marked as derived are calculated from one or more raw source variables rather than copied directly.

## Identifier

### `BabyID`
- Type: string
- Source: CTG preprocessing pipeline
- Description: An anonymized identifier representing one pregnancy/child episode.
- Derivation: Created earlier in the CTG pipeline and used as the key linking `registry.csv` to the final anonymized CTG parquet file.

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
- Description: Maternal age at delivery, expressed in completed years.
- Derivation: Calculated from the mother’s birth date encoded in `personnummer_mor` and the delivery date. Only year precision is retained.

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
- Description: Duration in seconds from established labour to birth.
- Derivation: Calculated as the difference between `etablerade_varkar_timestamp` and `birth_timestamp`.

## Labour start and end classification

### `forlossningsstart`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `forlossningsstart_basta_skattning`
- Description: Best-estimate classification of how labour started.
- Handling: Saved as raw text.

### `forlossningsslut`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `forlossningsslut_basta_skattning`
- Description: Best-estimate classification of how labour ended.
- Handling: Saved as raw text.

## Apgar scores

### `apgar1`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `apgar_1_min`
- Description: Apgar score at 1 minute.

### `apgar5`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `apgar_5_min`
- Description: Apgar score at 5 minutes.
- Note: Rows without `apgar5` are excluded before Stage 7 matching.

### `apgar10`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `apgar_10_min`
- Description: Apgar score at 10 minutes.

## Maternal background variables

### `fodelseland`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `fodelseland`
- Description: Mother’s country of birth.
- Handling: Saved as raw text.

### `utbildningsniva`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `utbildningsniva`
- Description: Maternal education level.
- Handling: Saved as raw text.

### `para_mhv1`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `para_mhv1`
- Description: Number of previous children.

### `langd_inskrivning_cm`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `langd_inskrivning_cm`
- Description: Maternal height in centimetres at registration.

### `bmi_inskrivning`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `bmi_inskrivning`
- Description: Maternal body mass index at registration.

## Tobacco use

### `tobak_3_manader_fore_graviditet`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `tobak_3_manader_fore_graviditet`
- Description: Smoking status three months before pregnancy.
- Handling: Saved as raw text.

### `tobak_inskrivning`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `tobak_inskrivning`
- Description: Smoking status at registration.
- Handling: Saved as raw text.

### `tobak_vecka_30_32`
- Type: string
- Source: `gravniva.csv`
- Raw variable: `tobak_vecka_30_32`
- Description: Smoking status during gestational week 30 to 32.
- Handling: Saved as raw text.

### `is_smoker`
- Type: boolean
- Source: derived from `gravniva.csv`
- Description: Indicator of maternal smoking.
- Derivation: Set to `True` if any of the three tobacco variables contains a smoking quantity entry such as `1–9 cigg/dag` or `10 eller fler cigg/dag`. Otherwise set to `False`.

## Maternal clinical variables

### `diabetes_mellitus`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `diabetes_mellitus`
- Description: Maternal diabetes mellitus indicator.
- Derivation: `Ja` is mapped to `True`, `Nej` to `False`, other values remain missing.

### `alkohol_audit_poang`
- Type: integer
- Source: `gravniva.csv`
- Raw variable: `alkohol_audit_poang`
- Description: Maternal alcohol use score.

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

## Cord blood pH and neonatal death

### `ph_navelartar`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `ph_navelartar`
- Description: Umbilical artery pH.

### `ph_navelven`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `ph_navelven`
- Description: Umbilical vein pH.

### `ph_navel_below7`
- Type: boolean
- Source: derived from `gravniva.csv`
- Description: Indicator for low umbilical pH.
- Derivation: Evaluated first on `ph_navelartar`; if arterial pH is missing, `ph_navelven` is used instead. Set to `True` when the available pH is below 7, otherwise `False`. Missing if neither pH is available.

### `avled_datum`
- Type: date
- Source: `gravniva.csv`
- Raw variable: `avled_datum`
- Description: Recorded date of neonatal death, when present.

### `died_after_days`
- Type: integer
- Source: derived from `gravniva.csv`
- Description: Number of days from birth to death within the neonatal period.
- Derivation: Calculated from `avled_datum - birth_day`, but only retained if the interval is between 0 and 28 days inclusive.

## Diagnosis and intervention indicators from maternal registry code fields

The following variables are derived from comma-separated code lists in `gravniva.csv`. Prefix matching means that a requested code such as `O13` will match more specific codes beginning with `O13`.

### `gestational_hypertension_without_significant_proteinuria`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Maternal gestational hypertension without significant proteinuria.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O13` or `O16`.

### `preeclampsia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Maternal preeclampsia/eclampsia indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O14` or `O15`.

### `gestational_or_pregestational_diabetes`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Maternal gestational or pregestational diabetes indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O24`.

### `uterine_rupture`
- Type: boolean
- Source: `gravniva.csv`
- Raw variables: `moderns_diagnoser_rad`, `moderns_atgarder_rad`
- Description: Uterine rupture indicator.
- Derivation: Set to `True` if maternal diagnosis contains `O710` or `O711`, or if maternal intervention contains `MCC00`.

### `sepsis`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Maternal sepsis indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `A41`.

### `placental_abruption`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Placental abruption indicator.
- Derivation: Implemented as requested using exact code `O711`.

### `heavy_vaginal_bleeding_before_or_during_delivery`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Heavy vaginal bleeding before or during delivery.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O46` or `O67`.

### `umbilical_cord_prolapse`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Umbilical cord prolapse indicator.
- Derivation: Set to `True` if exact code `O690` is present.

### `shoulder_dystocia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variables: `moderns_diagnoser_rad`, `barnets_diagnoser_rad`
- Description: Shoulder dystocia indicator.
- Derivation: Set to `True` if any maternal diagnosis code starts with `O66`, or if child diagnosis contains any of `P140`, `P141`, `P143`, `P148`, or `P149`.

### `labor_dystocia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Labour dystocia indicator.
- Derivation: Set to `True` if any of the exact codes `O620`, `O621`, `O628`, or `O629` are present.

### `use_of_oxytocin`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `moderns_atgarder_rad`
- Description: Use of oxytocin during labour/delivery.
- Derivation: Set to `True` if exact intervention code `DT036` or `DT037` is present.

## Diagnosis and intervention indicators from child registry code fields

### `severe_birth_asphyxia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variables: `barnets_diagnoser_rad`, `barnets_atgarder_rad`
- Description: Severe birth asphyxia indicator.
- Derivation: Set to `True` if child diagnosis contains `P210`, `P808`, or `P809`, or if child intervention contains `DV034`.

### `meconium_aspiration_syndrome`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `barnets_diagnoser_rad`
- Description: Meconium aspiration syndrome indicator.
- Derivation: Set to `True` if exact code `P240` is present.

### `treatment_for_hypoglycemia`
- Type: boolean
- Source: `gravniva.csv`
- Raw variable: `barnets_diagnoser_rad`
- Description: Treatment for neonatal hypoglycaemia indicator.
- Derivation: Set to `True` if any of `P703`, `P704A`, `P704B`, `P708`, or `P709` is present.

## SNQ-derived variables

SNQ variables are obtained by linking `gravniva.csv` to `SNQ data.xlsx` using `glopnr`. Because many children are not represented in SNQ, missing values are common for these variables and should not automatically be interpreted as negative findings.

### `highest_hie`
- Type: integer
- Source: `SNQ data.xlsx`
- Raw variable: `Högst HIE`
- Description: Highest recorded grade of hypoxic-ischaemic encephalopathy.

### `hie`
- Type: boolean
- Source: `SNQ data.xlsx`
- Raw variable: `HIE`
- Description: Indicator of hypoxic-ischaemic encephalopathy.
- Derivation: `Ja` is mapped to `True`, `Nej` to `False`.

### `intracranial_haemorrhage`
- Type: boolean
- Source: `SNQ data.xlsx`
- Raw variable: `ICD_kod`
- Description: Indicator of intracranial haemorrhage.
- Derivation: Set to `True` if any semicolon-separated SNQ ICD code starts with `P10` or `P52`.

### `neonatal_convulsions`
- Type: boolean
- Source: `SNQ data.xlsx`
- Raw variable: `ICD_kod`
- Description: Indicator of neonatal convulsions.
- Derivation: Set to `True` if any SNQ ICD code starts with `P90`.

### `neonatal_sepsis_or_pneumonia`
- Type: boolean
- Source: `SNQ data.xlsx`
- Raw variable: `ICD_kod`
- Description: Indicator of neonatal sepsis or pneumonia.
- Derivation: Set to `True` if any SNQ ICD code starts with `P23`, `P36`, or `P392`.

### `respiratorbehandling`
- Type: boolean
- Source: `SNQ data.xlsx` and `gravniva.csv`
- Raw variables: `KVÅ_kod`, `ventilation_pa_mask_min`, `intubation_min`, `hjartmassage_min`
- Description: Indicator of ventilator treatment or advanced neonatal resuscitation support.
- Derivation: Set to `True` if SNQ `KVÅ_kod` contains any code starting with `DG021`, `DG022`, or `DG0002`, or if any of the gravniva variables `ventilation_pa_mask_min`, `intubation_min`, or `hjartmassage_min` is non-empty.

## Notes on missing values

Missing values in `registry.csv` generally reflect one of the following:

- the source registry variable was not recorded,
- the information was not applicable,
- the child was not represented in the SNQ registry,
- or a derived field could not be calculated because one or more required source variables were missing.

These missing values should be handled explicitly in downstream statistical analysis or machine learning workflows rather than being automatically converted to zero or `False`.

## Notes on code parsing

- In `gravniva.csv`, diagnosis and intervention code fields are treated as comma-separated lists.
- In `SNQ data.xlsx`, code fields are treated as semicolon-separated lists.
- Prefix matching is used where specified, meaning that broader codes such as `O13` match more specific codes beginning with `O13`.
- Exact matching is used where a precise code was explicitly requested.
