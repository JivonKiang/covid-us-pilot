# -*- coding: utf-8 -*-
"""Export per-figure source data for Fig 1-5 (LLM-agent epidemic forecasting paper).

Output: e:\\ai agent\\outputs\\source_data\\fig{1..5}_source_data.csv
Schema: figure, panel, series, x, y, value, unit, note, source_file

Conventions
-----------
x      x coordinate (ISO date, numeric, or category label)
y      second coordinate when the panel is a 2-D plot (scatter / dumbbell), else ""
value  the primary plotted quantity
unit   unit of ``value``
note   provenance / derivation, including any transformation applied for display
Rows for purely schematic / hard-coded-text panels carry series="annotation",
       x = item label, value = "" and the displayed text in ``note``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"e:\ai agent\research\covid-us-pilot\data\panel")
OUT = Path(r"e:\ai agent\outputs\source_data")
OUT.mkdir(parents=True, exist_ok=True)

COLS = ["figure", "panel", "series", "x", "y", "value", "unit", "note", "source_file"]
BUF = []


def _num(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if np.isnan(f):
        return ""
    return round(f, 6)


def _x(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return "" if v is None else str(v)


def rec(fig, panel, series, x="", y="", value="", unit="", note="", source=""):
    BUF.append(dict(figure=fig, panel=panel, series=series, x=_x(x), y=_num(y),
                    value=_num(value), unit=unit, note=note, source_file=source))


def ann(fig, panel, item, text, source):
    rec(fig, panel, "annotation", x=item, note=text, source=source)


# =====================================================================  FIG 1
FIG = "1"
ann(FIG, "a", "engine schematic",
    "Two pipelines (US / CN) feeding one SEIR core; schematic only, no quantitative "
    "data. Drawing code: make_fig1.py::draw_schematic.", "make_fig1.py")

us_rmse = pd.read_csv(DATA / "h1_rmse.csv").groupby("model")["wrmse"].mean()
cn_ceil = pd.read_csv(DATA / "h1_rmse_china.csv").groupby("model")["wrmse"].mean()
cn_surge = pd.read_csv(DATA / "h1_rmse_wavelift.csv").groupby("model")["rmse"].mean()


def _gain(s, ref):
    return (s[ref] - s["llm_full"]) / s[ref] * 100.0


for win, s, fn in (("US steady (held-out)", us_rmse, "h1_rmse.csv"),
                   ("CN ceiling (stable)", cn_ceil, "h1_rmse_china.csv"),
                   ("CN surge (reopening)", cn_surge, "h1_rmse_wavelift.csv")):
    for ref, lab in (("persistence", "gain vs persistence"),
                     ("base_ar1", "gain vs baseline-AR1")):
        rec(FIG, "b", lab, x=win, value=_gain(s, ref), unit="% RMSE change",
            note=f"overall RMSE llm_full={s['llm_full']:.1f}, {ref}={s[ref]:.1f}; "
                 "overall = equal-weight mean over 6 behaviours (US / CN ceiling) or "
                 "over 8 units x 2 indicators (CN surge); positive = LLM better",
            source=fn)
rec(FIG, "b", "AR1 divergence", x="CN surge",
    value=cn_surge["base_ar1"] / cn_surge["persistence"], unit="ratio",
    note="baseline-AR1 RMSE divided by persistence RMSE in the reopening wave",
    source="h1_rmse_wavelift.csv")

hm = pd.read_csv(DATA / "h1_rmse.csv")
for _, r in hm.iterrows():
    rec(FIG, "c", r["model"], x=r["behavior"], value=r["wrmse"],
        unit="weighted RMSE (1-4 frequency scale)",
        note="heatmap cell; lower is better; cell shading normalised to the panel max",
        source="h1_rmse.csv")

obs = pd.read_csv(DATA / "us_national_week.csv", parse_dates=["week_start"])
pers = pd.read_csv(DATA / "behavior_baseline_pred.csv", parse_dates=["week_start"])
ar1 = pd.read_csv(DATA / "behavior_baseline_ar1_pred.csv", parse_dates=["week_start"])
llm = pd.read_csv(DATA / "llm_pred_long.csv", parse_dates=["week_start"])
BEH = "yougov_mask_freq"
UNIT_BEH = "self-reported frequency (1-4)"
for d, v in zip(obs["week_start"], obs[BEH]):
    rec(FIG, "d", "Observed (YouGov)", x=d, value=v, unit=UNIT_BEH,
        note="national weekly mask-wearing frequency", source="us_national_week.csv")
for d, v in zip(pers["week_start"], pers[f"{BEH}_pred"]):
    rec(FIG, "d", "Persistence", x=d, value=v, unit=UNIT_BEH, source="behavior_baseline_pred.csv")
for d, v in zip(ar1["week_start"], ar1[f"{BEH}_pred"]):
    rec(FIG, "d", "Baseline-AR1", x=d, value=v, unit=UNIT_BEH, source="behavior_baseline_ar1_pred.csv")
llm_mask = llm[(llm["config"] == "full") & (llm["behavior"] == BEH)].sort_values("week_start")
for d, v in zip(llm_mask["week_start"], llm_mask["pred"]):
    rec(FIG, "d", "LLM panel", x=d, value=v, unit=UNIT_BEH,
        note="held-out (LLM prediction) window only", source="llm_pred_long.csv")

ww = pd.read_csv(DATA / "wavelift_week.csv")
ww_nat = ww[ww["unit"] == "national"].sort_values("week").copy()
ww_nat["date"] = pd.Timestamp("2022-12-17") + pd.to_timedelta((ww_nat["week"] - 1) * 7, unit="D")
for d, v in zip(ww_nat["date"], ww_nat["BESI_log"]):
    rec(FIG, "e", "Observed (BESI log)", x=d, value=v, unit="BESI (log scale)",
        note="week index anchored at 2022-12-17 (Saturdays)", source="wavelift_week.csv")
wllm = pd.read_csv(DATA / "llm_pred_long_wavelift.csv", parse_dates=["week_start"])
wllm_nat = wllm[(wllm["unit"] == "national") & (wllm["indicator"] == "BESI_log")]
for cfg, lab in (("full", "LLM full"), ("no_inertia", "LLM -inertia"),
                 ("no_policy", "LLM -policy"), ("no_burden", "LLM -burden")):
    d = wllm_nat[wllm_nat["config"] == cfg].sort_values("week_start")
    for dd, v in zip(d["week_start"], d["pred"]):
        rec(FIG, "e", lab, x=dd, value=v, unit="BESI (log scale)",
            note="ablation" if cfg != "full" else "main model", source="llm_pred_long_wavelift.csv")

F_BEH = [("yougov_mask_freq", "Mask"), ("yougov_stay_home_freq", "Stay home"),
         ("yougov_avoid_restaurants_freq", "Restaurants"), ("yougov_avoid_crowds_freq", "Avoid crowds"),
         ("yougov_avoid_shops_freq", "Shops"), ("yougov_avoid_work_freq", "Avoid work")]
full_us = llm[llm["config"] == "full"]
cal_all = []
for beh, lab in F_BEH:
    p = full_us[full_us["behavior"] == beh][["week_start", "pred"]]
    o = obs[["week_start", beh]].rename(columns={beh: "obs"})
    m = p.merge(o, on="week_start", how="inner")
    rho = m["obs"].corr(m["pred"], method="spearman")
    cal_all.append(m)
    for pr, ob in zip(m["pred"], m["obs"]):
        rec(FIG, "f", lab, x=pr, y=ob, value=ob, unit=UNIT_BEH,
            note=f"Spearman rho = {rho:+.3f} (this behaviour)",
            source="llm_pred_long.csv + us_national_week.csv")
cal = pd.concat(cal_all, ignore_index=True)
rec(FIG, "f", "Spearman rho (all six)", x="pooled", value=cal["obs"].corr(cal["pred"], method="spearman"),
    unit="dimensionless", note=f"n = {len(cal)} paired weekly observations",
    source="llm_pred_long.csv + us_national_week.csv")

ho = pd.read_csv(DATA / "h1_overall.csv").set_index("model")
for beh in ["mask", "stay_home", "avoid_restaurants", "avoid_crowds", "avoid_shops",
            "avoid_work", "overall"]:
    for ref, lab in (("persistence", "gain vs persistence"), ("base_ar1", "gain vs AR(1)")):
        rec(FIG, "g", lab, x=beh, value=_gain(ho[beh], ref), unit="% RMSE gain",
            note="positive = LLM full better", source="h1_overall.csv")

# =====================================================================  FIG 2
FIG = "2"
traj = pd.read_csv(DATA / "seir_traj.csv", parse_dates=["week_start"])
for col, lab in (("obs_cases", "Observed"), ("pred_no_feedback", "No feedback"),
                 ("pred_rule_driven", "Behaviour: rules"), ("pred_llm_driven", "Behaviour: LLM")):
    for d, v in zip(traj["week_start"], traj[col]):
        rec(FIG, "a", lab, x=d, value=v, unit="weekly new cases", source="seir_traj.csv")

fits = pd.read_csv(DATA / "seir_fits.csv").set_index("config")
for cfg, lab in (("no_feedback", "no feedback"), ("rule_driven", "behaviour: rules"),
                 ("llm_driven", "behaviour: LLM")):
    tr, te = float(fits.loc[cfg, "train_rmse"]), float(fits.loc[cfg, "test_rmse"])
    rec(FIG, "b", lab, x="train RMSE", value=tr, unit="weekly RMSE (cases)", source="seir_fits.csv")
    rec(FIG, "b", lab, x="test RMSE", value=te, unit="weekly RMSE (cases)",
        note=f"test/train = x{te / tr:.2f}", source="seir_fits.csv")

traj_cn = pd.read_csv(DATA / "seir_traj_china.csv", parse_dates=["week_start"])
for prov in ["Hubei", "Guangdong", "Beijing"]:
    d = traj_cn[traj_cn.province == prov]
    for col, lab in (("obs_cases", "Observed"), ("pred_no_feedback", "No feedback"),
                     ("pred_rule_driven", "With feedback")):
        for dd, v in zip(d["week_start"], d[col]):
            rec(FIG, "c", f"{prov} | {lab}", x=dd, value=v, unit="weekly cases",
                note="province small multiple (independent y scale)",
                source="seir_traj_china.csv")

m14 = pd.read_csv(DATA / "m14_h2_traj_main.csv", parse_dates=["date"])
CN_N = 1.4117e9
for d, v in zip(m14["date"], m14["daily_new_inf"]):
    rec(FIG, "d", "Simulated daily infections", x=d, value=v, unit="infections/day",
        note="main M14 scenario (log y axis)", source="m14_h2_traj_main.csv")
for d, v in zip(m14["date"], m14["cum_inf"]):
    rec(FIG, "d", "Cumulative infected (%)", x=d, value=v / CN_N * 100, unit="% of population",
        note=f"cum_inf / N_CN (N = {CN_N:.4g}); right axis", source="m14_h2_traj_main.csv")
for d, v in (("2022-12-09", 4.25e5), ("2022-12-23", 2.867e6), ("2023-02-06", 1.37e5)):
    rec(FIG, "d", "Observed anchor", x=d, value=v, unit="fever-clinic visits/day",
        note="three observed anchoring points (in-panel hard-coded)", source="make_fig2.py")

fits_cn = pd.read_csv(DATA / "seir_fits_china.csv")
for lab, val, src in (
        ("US rules", float(fits.loc["rule_driven", "b_test_mean"]), "seir_fits.csv"),
        ("US LLM", float(fits.loc["llm_driven", "b_test_mean"]), "seir_fits.csv"),
        ("Hubei rules", float(fits_cn.query("province=='Hubei' and config=='rule_driven'")["b_test_mean"].iloc[0]),
         "seir_fits_china.csv")):
    rec(FIG, "e", "mean behaviour factor b (e1)", x=lab, value=val, unit="dimensionless",
        note="test window", source=src)
for lab, val, src, nt in (
        ("US", float(fits.loc["no_feedback", "test_rmse"] / fits.loc["llm_driven", "test_rmse"]),
         "seir_fits.csv", "no_feedback / llm_driven"),
        ("Hubei", float(fits_cn.query("province=='Hubei' and config=='no_feedback'")["test_rmse"].iloc[0] /
                        fits_cn.query("province=='Hubei' and config=='rule_driven'")["test_rmse"].iloc[0]),
         "seir_fits_china.csv", "no_feedback / rule_driven"),
        ("Guangdong", 1.0, "seir_fits_china.csv", "epsilon = 0 -> both configs identical; ratio fixed at 1.0"),
        ("Beijing", 1.0, "seir_fits_china.csv", "epsilon = 0 -> both configs identical; ratio fixed at 1.0")):
    rec(FIG, "e", "test RMSE ratio (e2)", x=lab, value=val, unit="ratio",
        note=nt + "; >1 means feedback improves the test fit", source=src)

FIG2F = [("", "US Delta", "CN zero", "CN exit"),
         ("Engine", "SEIR", "SEIR", "SEIR"),
         ("b(t)", "eps 0.25", "0.125 / 0", "off (b=0)"),
         ("Data", "weekly", "prov. wk", "3 anchors"),
         ("Patch", "none", "none", "none")]
for rowlab, *cells in FIG2F[1:]:
    for head, cell in zip(FIG2F[0][1:], cells):
        ann(FIG, "f", f"{head} | {rowlab}", cell, "make_fig2.py")

# =====================================================================  FIG 3
FIG = "3"
NODES = [
    "Preregister v1.0 - no data contact yet - SHA 86690c...",
    "v1.1 pre-data amendment - P-criteria untouched - SHA 36e53c...",
    "Forecast locked - P1 = Feb 27 - P3 = 14.4% - SHA f55eee...",
    "Truth harvest - 6 searches + 2 fetches - all after lock",
    "Mechanical unblind - H3 REFUTED (this figure)",
]
for i, t in enumerate(NODES, 1):
    ann(FIG, "a", f"node {i}", t, "make_fig3.py")

m13 = pd.read_csv(DATA / "m13_blindtest_traj_main.csv", parse_dates=["date"])
N13 = 1.4117e9
for d, v in zip(m13["date"], m13["daily_new_inf"]):
    rec(FIG, "b", "Predicted daily new infections", x=d, value=v, unit="infections/day",
        source="m13_blindtest_traj_main.csv")
for d, v in zip(m13["date"], m13["cum_inf"]):
    rec(FIG, "b", "Cumulative infected (%)", x=d, value=v / N13 * 100, unit="% of population",
        note=f"cum_inf / {N13:.4g}", source="m13_blindtest_traj_main.csv")
ann(FIG, "b", "true peak window", "Nov 30 - Dec 27 (3-proxy truth window)", "make_fig3.py")
ann(FIG, "b", "forecast peak", "P1 = Feb 27 (+62 d past the window right edge)", "make_fig3.py")
ann(FIG, "b", "mismatch", "+62 d past window right edge; cumulative Mar 5 = 14.4% vs truth floor 20.1%",
    "make_fig3.py")

grid = pd.read_csv(DATA / "m13_blindtest_grid.csv")
rec(FIG, "c", "P1 predicted peak", x="prediction", value=1.0, unit="date",
    note="2023-02-27 (main-scenario peak week)", source="m13_blindtest_grid.csv")
ann(FIG, "c", "P1 grid span", f"grid peak weeks {grid['P1_peak_week'].min()} to {grid['P1_peak_week'].max()}",
    "m13_blindtest_grid.csv")
rec(FIG, "c", "truth window", x="window left", value=1.0, unit="date", note="2022-11-30", source="make_fig3.py")
rec(FIG, "c", "truth window", x="window right", value=1.0, unit="date", note="2022-12-27", source="make_fig3.py")

p3_main = float(grid.loc[grid.is_main, "P3_cum_inf_rate_0305"].iloc[0]) * 100
rec(FIG, "d", "P3 predicted rate", x="prediction", value=p3_main, unit="% of population",
    note="main scenario cumulative infection rate by 2023-03-05", source="m13_blindtest_grid.csv")
rec(FIG, "d", "P3 grid span", x="grid max", value=float(grid["P3_cum_inf_rate_0305"].max()) * 100,
    unit="% of population", note="parameter-grid upper bound", source="m13_blindtest_grid.csv")
rec(FIG, "d", "truth envelope", x="lower bound", value=20.1, unit="% of population",
    note="deaths / IFR union provincial serology", source="make_fig3.py")
rec(FIG, "d", "truth envelope", x="upper bound", value=100.0, unit="% of population",
    note="envelope ceiling; floor is extraction-sensitive", source="make_fig3.py")
rec(FIG, "d", "provincial serology", x="lower bound", value=85.0, unit="% of population", source="make_fig3.py")
rec(FIG, "d", "provincial serology", x="upper bound", value=95.0, unit="% of population", source="make_fig3.py")
ann(FIG, "d", "verdict", "-5.7 pp MISS (prediction 14.4% vs truth floor 20.1%)", "make_fig3.py")

rec(FIG, "e", "infection doubling time", x="US cases", value=8.4, unit="days",
    note="inflated by the Christmas testing trough + home-test leakage", source="make_fig3.py")
rec(FIG, "e", "infection doubling time", x="CN wave", value=1.5, unit="days",
    note="range 1.0-2.0 d (effective); calibrated value 1.5 d", source="make_fig3.py")

for lab, x0, x1 in (("T1 Baoding etc.", 0.02, 0.34), ("T2 Beijing-GZ", 0.20, 0.58),
                    ("T3 Shanghai etc.", 0.42, 0.80)):
    ann(FIG, "f", f"tier {lab[:2]}", f"{lab}: city-ignition window x = {x0:.2f}-{x1:.2f} (schematic, axes fraction)",
        "make_fig3.py")
ann(FIG, "f", "contrast", "real wave: staggered city ignition; model: one homogeneous pool; "
                         "burn-through 85% in ~4-5 wk vs 14 wk to only 14.4%", "make_fig3.py")

for crit, pred, truth, verdict in (("P1 peak", "Feb 27", "Nov 30-Dec 27", "MISS"),
                                   ("P3 rate", "14.4%", "20.1-100%", "MISS"),
                                   ("H3 all", "-", "-", "REFUTED")):
    ann(FIG, "g", crit, f"prediction {pred} | truth {truth} | verdict {verdict}", "make_fig3.py")
ann(FIG, "g", "sensitivity", "envelope floor depends on the Economist extraction "
                             "(1.7M actual vs 6.8e5 counterfactual, predeclared); "
                             "under-prediction robust - serology puts truth at 85%+", "make_fig3.py")

# =====================================================================  FIG 4
FIG = "4"
cf1 = pd.read_csv(DATA / "m19_cf1_cn_early_traj.csv", parse_dates=["date_base", "date_q1a", "date_q1b"])
cf2 = pd.read_csv(DATA / "m19_cf2_cn_gradual_traj.csv", parse_dates=["date"])
cf3 = pd.read_csv(DATA / "m19_cf3_us_early_npi_traj.csv", parse_dates=["date"])
cf4 = pd.read_csv(DATA / "m19_cf4_vax_traj.csv", parse_dates=["date"])
m1 = pd.read_csv(DATA / "m19_cf1_metrics.csv")
m2 = pd.read_csv(DATA / "m19_cf2_metrics.csv")
m3 = pd.read_csv(DATA / "m19_cf3_metrics.csv")
m4 = pd.read_csv(DATA / "m19_cf4_metrics.csv")


def mv(df, key, col):
    return float(df[df.scenario.str.contains(key, case=False, regex=False)][col].iloc[0])


TRAJ = [("Q1a baseline", cf1["date_base"], cf1["base_daily"], "m19_cf1_cn_early_traj.csv"),
        ("Q1a early start", cf1["date_q1a"], cf1["q1a_daily"], "m19_cf1_cn_early_traj.csv"),
        ("Q1b baseline", cf1["date_base"], cf1["base_daily"], "m19_cf1_cn_early_traj.csv"),
        ("Q1b early start + I0/10", cf1["date_q1b"], cf1["q1b_daily"], "m19_cf1_cn_early_traj.csv"),
        ("Q2 baseline", cf2["date"], cf2["base_daily"], "m19_cf2_cn_gradual_traj.csv"),
        ("Q2 gradual ramp", cf2["date"], cf2["q2_daily"], "m19_cf2_cn_gradual_traj.csv"),
        ("Q3 baseline", cf3["date"], cf3["base_daily"], "m19_cf3_us_early_npi_traj.csv"),
        ("Q3 early NPI", cf3["date"], cf3["q3_daily"], "m19_cf3_us_early_npi_traj.csv"),
        ("Q4 baseline", cf4["date"], cf4["base_daily"], "m19_cf4_vax_traj.csv"),
        ("Q4 immune wall", cf4["date"], cf4["q4_daily"], "m19_cf4_vax_traj.csv")]
for lab, dx, dy, src in TRAJ:
    for d, v in zip(dx, dy):
        rec(FIG, "a", lab, x=d, value=v, unit="infections/day",
            note="displayed in the panel as % of that scenario's own baseline peak",
            source=src)

SC = [("Q1a", 0, mv(m1, "Q1a", "dPeakH"), mv(m1, "Q1a", "dCum"), "m19_cf1_metrics.csv"),
      ("Q1b", 10, mv(m1, "Q1b", "dPeakH"), mv(m1, "Q1b", "dCum"), "m19_cf1_metrics.csv"),
      ("Q2", 16, mv(m2, "Q2(", "dPeakH"), mv(m2, "Q2(", "dCum"), "m19_cf2_metrics.csv"),
      ("Q3", 95, mv(m3, "Q3", "dPeakH"), mv(m3, "Q3", "dCum"), "m19_cf3_metrics.csv"),
      ("Q4", 0, mv(m4, "Q4(", "dPeakH"), mv(m4, "Q4(", "dCum"), "m19_cf4_metrics.csv")]
for q, dp, dh, dc, src in SC:
    rec(FIG, "b", q, x="delta peak height", value=dh, unit="% vs own baseline",
        note="dumbbell: circle = peak height, square = cumulative; fill differs per question",
        source=src)
    rec(FIG, "b", q, x="delta cumulative", value=dc, unit="% vs own baseline", source=src)
rec(FIG, "b", "Q4 VE sensitivity (low), height", x="delta peak height",
    value=mv(m4, "低界", "dPeakH"), unit="% vs own baseline", source="m19_cf4_metrics.csv")
rec(FIG, "b", "Q4 VE sensitivity (low), cumulative", x="delta cumulative",
    value=mv(m4, "低界", "dCum"), unit="% vs own baseline", source="m19_cf4_metrics.csv")
rec(FIG, "b", "Q4 VE sensitivity (high), height", x="delta peak height",
    value=mv(m4, "高界", "dPeakH"), unit="% vs own baseline", source="m19_cf4_metrics.csv")
rec(FIG, "b", "Q4 VE sensitivity (high), cumulative", x="delta cumulative",
    value=mv(m4, "高界", "dCum"), unit="% vs own baseline", source="m19_cf4_metrics.csv")

for q, dp, dh, dc, src in SC:
    rec(FIG, "c", q, x="delay of the modelled peak", value=dp, unit="days",
        note="epidemiological (peak-relative) time axis, NOT the calendar shift used in Fig 5b",
        source=src)

TAB = [("Q1a", "start date only", "no change (time-shift identity)", "0 d / 0.00% / 0.00%", "supported"),
       ("Q1b", "start date + I0 / 10", "peak height and cumulative < 5%", "+10 d / -0.26% / -0.37%",
        "timing unexpected"),
       ("Q2", "beta ramp over 42 d", "peak height -10 to -40%", "+16 d / -0.19% / -2.00%", "height unexpected"),
       ("Q3", "beta x 0.6, first 5 wk", "+0-21 d, height -15 to -50%", "+95 d / -9.20% / -6.49%",
        "unexpected (R_eff < 1)"),
       ("Q4", "immune wall delta = 0.116%", "peak height and cumulative < 5%", "0 d / -0.21% / -0.12%",
        "supported (null result)")]
for q, tested, expect, meas, verdict in TAB:
    ann(FIG, "d", q, f"tested: {tested} | expected: {expect} | measured: {meas} | verdict: {verdict}",
        "make_fig4.py")

for q, dp, dh, dc, src in SC:
    if q == "Q1a":
        continue
    rec(FIG, "e", q, x=dh, y=dc, value=dc, unit="% vs own baseline",
        note="x = delta peak height, y = delta cumulative; dotted y = x reference",
        source=src)

DELTA = [float(m4.loc[m4.scenario.str.contains("低界", regex=False), "delta"].iloc[0]) * 100,
         float(m4.loc[m4.scenario.str.contains("Q4(", regex=False), "delta"].iloc[0]) * 100,
         float(m4.loc[m4.scenario.str.contains("高界", regex=False), "delta"].iloc[0]) * 100]
VE_LOW = (mv(m4, "低界", "dPeakH"), mv(m4, "低界", "dCum"))
VE_HIGH = (mv(m4, "高界", "dPeakH"), mv(m4, "高界", "dCum"))
for lab, dlt, dh, dc in (("VE low (95% CI)", DELTA[0], VE_LOW[0], VE_LOW[1]),
                         ("VE central", DELTA[1], mv(m4, "Q4(", "dPeakH"), mv(m4, "Q4(", "dCum")),
                         ("VE high (95% CI)", DELTA[2], VE_HIGH[0], VE_HIGH[1])):
    rec(FIG, "f", lab, x=dlt, y=dh, value=dh, unit="% change in infections",
        note=f"x = immune wall delta (% of population); cumulative change = {dc:+.4f}%",
        source="m19_cf4_metrics.csv")

# =====================================================================  FIG 5
FIG = "5"
N_CN5, N_US5 = 1.41e9, 3.3e8
CN_SPEC = [("Baseline", "m19_cf1_cn_early_traj.csv", "date_base", "base_daily"),
           ("Q1a early start", "m19_cf1_cn_early_traj.csv", "date_q1a", "q1a_daily"),
           ("Q1b early start", "m19_cf1_cn_early_traj.csv", "date_q1b", "q1b_daily"),
           ("Q2 gradual ramp", "m19_cf2_cn_gradual_traj.csv", "date", "q2_daily"),
           ("Q4 vaccination", "m19_cf4_vax_traj.csv", "date", "q4_daily")]
for lab, fn, dc, cc in CN_SPEC:
    d = pd.read_csv(DATA / fn, parse_dates=[dc])
    y = d[cc].to_numpy(float)
    sn = 1 - np.cumsum(y) / N_CN5
    for dd, v in zip(d[dc], sn):
        rec(FIG, "a", f"CN | {lab}", x=dd, value=v, unit="share still susceptible (S/N)",
            note=f"N_CN = {N_CN5:.3g}; S/N = 1 - cumsum(daily)/N", source=fn)
for lab, cc in (("Baseline", "base_daily"), ("Q3 early NPI", "q3_daily")):
    d = pd.read_csv(DATA / "m19_cf3_us_early_npi_traj.csv", parse_dates=["date"])
    y = d[cc].to_numpy(float)
    sn = 1 - np.cumsum(y) / N_US5
    for dd, v in zip(d["date"], sn):
        rec(FIG, "a", f"US | {lab}", x=dd, value=v, unit="share still susceptible (S/N)",
            note=f"N_US = {N_US5:.3g}; S/N = 1 - cumsum(daily)/N", source="m19_cf3_us_early_npi_traj.csv")

base_cn_peak, base_us_peak = 97699011.0, 699157.0
PTS = [("Q1a", -91, (97699011 / base_cn_peak - 1) * 100),
       ("Q1b", -81, (97443871 / base_cn_peak - 1) * 100),
       ("Q2", 16, (97512768 / base_cn_peak - 1) * 100),
       ("Q4", 0, (97492483 / base_cn_peak - 1) * 100),
       ("Q3", 95, (634826 / base_us_peak - 1) * 100)]
for q, dx, dy in PTS:
    rec(FIG, "b", q, x=dx, y=dy, value=dy, unit="% change in peak height",
        note="x = calendar shift in peak timing (days, relative to the baseline peak date); "
             "peak counts hard-coded in make_fig5.py (CN baseline peak 97,699,011/day; US 699,157/day)",
        source="make_fig5.py")
ann(FIG, "b", "tolerance band", "+/-1% peak-height band (axhspan -1.0 to 1.0)", "make_fig5.py")

G5 = pd.read_csv(DATA / "m14_h2_grid.csv", parse_dates=["peak_week_start", "daily_peak_date"])
T0 = G5.daily_peak_date.min()
WKF = (G5.daily_peak_date - T0).dt.days / 7.0
HIT3 = (G5.C1_hit & G5.C2_hit & G5.C3_hit).astype(int)
for i in range(len(G5)):
    note_c = ("filled contour: attack rate by 2023-02-05; displayed field is a cubic-spline "
              "refinement of this grid")
    rec(FIG, "c", "attack rate", x=G5.beta.iloc[i], y=G5.I0.iloc[i],
        value=G5.cum_rate_0205.iloc[i], unit="fraction of population", note=note_c,
        source="m14_h2_grid.csv")
    rec(FIG, "d", "peak week", x=G5.beta.iloc[i], y=G5.I0.iloc[i], value=WKF.iloc[i],
        unit="weeks after the earliest simulated peak",
        note="isochrones drawn on a Gaussian-smoothed field", source="m14_h2_grid.csv")
    rec(FIG, "e", "three-check pass", x=G5.beta.iloc[i], y=G5.I0.iloc[i], value=int(HIT3.iloc[i]),
        unit="1 = all three checks pass", note="C1 peak timing, C2 attack rate, C3 shape",
        source="m14_h2_grid.csv")
    rec(FIG, "g", "grid cell", x=G5.beta.iloc[i], y=G5.I0.iloc[i], value=G5.cum_rate_0205.iloc[i] * 100,
        unit="% cumulative infection by 5 Feb", note=f"all three checks pass = {int(HIT3.iloc[i])}",
        source="m14_h2_grid.csv")
cum_pct = G5.cum_rate_0205.to_numpy() * 100
bins = np.arange(0, 102.5, 2.5)
for lab, mask in (("hist: not all checks", HIT3 == 0), ("hist: all three checks", HIT3 == 1)):
    cnt, _ = np.histogram(cum_pct[mask.to_numpy()], bins=bins)
    for edge, c in zip(bins[:-1], cnt):
        rec(FIG, "g", lab, x=round(float(edge + 1.25), 2), value=int(c), unit="parameter cells",
            note="2.5%-wide bins, stacked histogram", source="m14_h2_grid.csv")

I0V = np.sort(G5.I0.unique())
for i0 in (1e4, 1e5, 1e6, 1e7):
    k = int(np.argmin(np.abs(I0V - i0)))
    sel = G5[G5.I0 == I0V[k]].sort_values("beta")
    for b, v in zip(sel["beta"], sel["cum_rate_0205"]):
        rec(FIG, "f", f"I0 = 1e{int(round(np.log10(max(I0V[k], 1.0))))}", x=b, y=v, value=v,
            unit="cumulative infection rate",
            note="line = nearest available grid I0 to the nominal decade", source="m14_h2_grid.csv")
rec(FIG, "f", "observed", x="reference line", value=0.201, unit="cumulative infection rate",
    note="observed 20.1% attack rate (horizontal reference)", source="make_fig5.py")

# =====================================================================  write
df = pd.DataFrame(BUF, columns=COLS).drop_duplicates()
for f, g in df.groupby("figure"):
    p = OUT / f"fig{f}_source_data.csv"
    g.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"fig{f}: {len(g)} rows -> {p.name}")
print("total rows:", len(df))