# Stage 7 Registry Data Dictionary

## Overview

This document describes the variables currently written to the final (Stage 7) registry file, `registry.csv`, in the CTG preprocessing pipeline. The purpose of this file is to document the origin, meaning, and derivation of each registry variable that is linked to the anonymized CTG data through `BabyID`.

The Stage 7 output consists of one row per matched pregnancy/child episode. Matching is performed between the CTG dataset and registry data using the maternal personal number and delivery date information before anonymization. The resulting `registry.csv` is intended to serve as the metadata table accompanying the final anonymized CTG parquet file.

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
- ML-use: Used as output
- Prevalance: 60.1%

### `ph_navelven`
- Type: float
- Source: `gravniva.csv`
- Raw variable: `ph_navelven`
- Description: Umbilical vein pH. Ranges from 6.19 to 7.83. 95% of cases are above 7.17.
- ML-use: Used as output
- Prevalance: 69.5%

### `ph_navel_below7`
- Type: boolean
- Source: derived from `gravniva.csv`
- Description: Indicator for low umbilical pH.
- Derivation: Evaluated first on `ph_navelartar`; if arterial pH is missing, `ph_navelven` is used instead. Set to `True` when the available pH is below 7, otherwise `False`. Missing if neither pH is available.
- ML-use: Used as output, but risky due to very few true samples.
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
- Source: `gravniva.csv`
- Raw variable: `moderns_diagnoser_rad`
- Description: Placental abruption indicator.
- Derivation: Implemented as requested using exact code `O711`.
- ML-use: Not used, due to too few True samples
- Prevalance: 0.14%

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
