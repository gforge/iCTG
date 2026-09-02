# Cohort loss audit (2026-09-02)

Question from the clinical side: the registry-to-CTG map looks incomplete. This note records
where pregnancies were lost in the pipeline that produced the CTG1-CTG3 datasets, why, and
what was changed.

## Where the pregnancies went

Counts from the collaborator's per-stage summary (`figure_generation.py`), "registry-overlap
patients" = patients whose PatientID also occurs in `gravniva.csv`.

| Stage | Pregnancies (BabyIDs) | Registry-overlap patients |
|-------|----------------------:|--------------------------:|
| Raw exports | 220,816 | 83,640 |
| Stage 1: Timestamp >= 2014-12-31 | 133,793 | 81,955 |
| Stage 3: session filter, BabyID | 133,793 | 81,955 |
| Stage 4: drop BabyIDs with >30 % duplicate timestamps | 87,632 | 58,443 |
| Stage 5: drop BabyIDs with <1200 s non-zero FHR | 44,453 | 34,480 |
| Stage 7: registry match | 30,871 | |

Only 38 % of the registry-linked patients that survive the date filter reach the final
dataset. Almost all of the loss is caused by two pipeline rules, not by missing data.

## Cause 1: the raw exports overlap, and stage 4 punished it

Every export file covers the same period (2009-06 to 2024-12); the March/April 2026 exports
re-exported data that was already in the May/June 2025 exports. Scanning the converter output
(`/srv/data/input/iCTG/parquet`, 8.23e9 rows) by PatientID and file:

| Patients seen in N export files | Patients | Rows |
|---:|---:|---:|
| 1 | 70,808 | 1.51e9 |
| 2 | 50,978 | 2.29e9 |
| 3 | 24,850 | 1.74e9 |
| 4-16 | 21,761 | 2.68e9 |

97,589 of 168,397 patients (58 %) occur in more than one export, and their rows are 82 % of
all rows. For those pregnancies most timestamps are duplicated, so the old stage 4 rule
("drop the BabyID if more than 30 % of timestamps are duplicated") deleted them wholesale.

Fix: identical rows are collapsed with `SELECT DISTINCT` at the start of stage 3, and stage 4
now only counts *conflicting* duplicates (same second, different FHR/toco/quality) towards
the 30 % threshold. Exact re-exports no longer cost anything.

## Cause 2: stage 3 only looked at the last session

Stage 3 split recordings into sessions at gaps > 5 minutes and kept the last 60 minutes of
the *last session only*. Labour recordings are frequently fragmented (transfer to theatre,
repositioning, a brief reconnection, a signal-less tail while the monitor is still on). In
all of those cases the last session is short or empty, and stage 5 then dropped the whole
pregnancy for having < 20 minutes of signal, even though a long recording ended ten minutes
earlier. This selectively removes complicated deliveries.

Fix: the default window scope is now `pregnancy`: the anchor is the last non-zero FHR in the
whole pregnancy and the 60-minute window keeps rows from every session inside it. The old
behaviour is available as `--stage3-window-scope final_session`. BabyIDs are unchanged.

## Other, smaller drop points (unchanged, now measurable)

- Registry rows without a 12-digit `personnummer_mor`, without `apgar5`, or without a birth
  date are excluded before matching.
- Matching requires `ctg_date` (date of the last CTG sample) to equal the birth day or the
  day before. Births more than one day after the last recording are not matched.
- A registry row matching several BabyIDs, or a BabyID matching several registry rows
  (twins), is dropped.

`match_loss_report.py` attributes every registry birth to exactly one of these categories
and shows the distribution of "days between last CTG and birth" for the unmatched ones, so the
+-1 day window can be revisited with numbers.

## What to do

1. Re-run stages 1-7 from `/srv/data/input/iCTG/parquet` with the current code (config now
   defaults to the server paths). The stage 2 toco decode was vectorised, which makes the
   rerun practical.
2. Run `cohort_report.py` and `match_loss_report.py` and compare against the table above.
3. Regenerate the CTG3 dataset and retrain: FHR values also change slightly because the old
   FHR mean used integer division.
