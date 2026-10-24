# covid-us-pilot

Code pipeline for **LLM-driven behavioral modeling of COVID-19 dynamics** (US and China): weekly state-level panel assembly, an LLM behavioral layer, SEIR tracking, and counterfactual analyses.

## Layout

```
covid-us-pilot/
├── data/raw/                  raw data (script-downloaded; not committed)
├── data/panel/                weekly state-level panel outputs + quality reports
├── download_*.py              data acquisition (JHU CSSE, OxCGRT, YouGov, Google Mobility)
├── probe_*.py / verify_*.py   inspection & validation helpers
├── extract_and_inspect.py     unzip + structure inspection
├── build_panel.py             weekly panel construction (full run ~8s)
├── llm_behavior_*.py          LLM behavioral layer (US wavelift, China)
├── seir_track2_china.py       SEIR tracking (China)
└── export_source_data.py      figure source-data export
```

## Panel outputs (data/panel/)

| File | Content |
|---|---|
| us_state_week.csv | Main panel: 6,987 rows = 51 jurisdictions x 137 weeks (2020-03-02 to 2022-10-10, Monday-anchored). Columns: burden (4) + population + policy (11, pol_*) + mobility (6, mob_*) + state sample size |
| us_national_week.csv | National behavior table: 61 weeks (2020-03-30 to 2021-10-11); 7 weighted yougov_*_freq means + sample sizes + always_share |
| quality_report.md | Coverage, missingness, negative cells, low-sample weeks, California sampling check |

## Data sources (raw data not committed; scripts below populate data/raw/)

| Source | Content | Coverage | Notes |
|---|---|---|---|
| JHU CSSE confirmed/deaths (county x day) | 17.8 / 11.9 MB | 2020-01-22 to 2023-03-09, 3,342 counties | aggregated to state level via Province_State |
| OxCGRT policy (NAT_TOTAL + STATE_TOTAL) | 14.6 MB | 2020-01-01 to 2022-12-31, 51 jurisdictions x 1,096 days | no overseas territories |
| YouGov United States (response level) | 33,940 responses | effective coverage 2020-03-30 to 2021-10-11 (61 weeks) | see issues 1-2 |
| YouGov China | 15.9 MB | same period | used by China extension |
| Google Mobility US (6 place categories, % change) | 2020-02-15 to 2022-10-15 | state rows: sub_region_1 set, sub_region_2 empty |

## Known data issues (empirically confirmed at panel stage)

1. **YouGov US coverage is shorter than expected**: 33,940 responses fall in only 61 weeks (~556/week); no behavioral data after 2021-10, so the last 76 of 137 panel weeks rely on Mobility only. National weekly n: median 811 (min 1, max 1,879).
2. **Low-sample weeks (n<50)**: 9 scattered weeks in 2020-06 to 2021-07 (e.g. 2020-08-24 n=1), consistent with sample rotation rather than a contiguous gap; down-weight or exclude during calibration.
3. **PHQ4 psychological items** largely missing 2021-02-23 to 10-13.
4. **Mobility ends early** (2022-10-15), bounding the panel at 137 weeks.
5. **JHU negative weekly increments**: 27 cells (6 confirmed / 21 deaths) from JHU back-corrections, kept as-is; late-period case series need cautious interpretation (home self-tests unreported after 2022-03).

## Reproduction notes

- Populate raw data: `python download_m1.py`, then `python download_phase2.py`, then `python download_oxcgrt_usa.py` (skip-if-exists, atomic `.part` then `os.replace`).
- CDC data.cdc.gov (Socrata) returned 403 across endpoints; burden series use JHU CSSE instead.
- Panel keys: `state x week_start` (Monday anchor). Burden = state-aggregated cumulative differencing **by position** (`to_numpy()`), not index-aligned subtraction. YouGov `endtime` is minute-resolution; `.dt.normalize()` after week-start binning is required. Quality reports use `df.to_string(index=False)` (no tabulate dependency).
