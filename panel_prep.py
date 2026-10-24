# -*- coding: utf-8 -*-
"""M6-10 共用数据准备：国家级周表 + 驱动变量（与 behavior_baseline.py 完全同口径）
df：可用周（n_total>=50），含 week_start/n_total/7 行为列/stringency/log_burden
"""
import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
PANEL = ROOT + r"\data\panel"

BEHAV = ["yougov_mask_freq", "yougov_hands_freq", "yougov_stay_home_freq",
         "yougov_avoid_restaurants_freq", "yougov_avoid_crowds_freq",
         "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]
BEHAV6 = [b for b in BEHAV if b != "yougov_hands_freq"]
SHORT = {"yougov_mask_freq": "mask", "yougov_stay_home_freq": "stay_home",
         "yougov_avoid_restaurants_freq": "avoid_restaurants",
         "yougov_avoid_crowds_freq": "avoid_crowds",
         "yougov_avoid_shops_freq": "avoid_shops",
         "yougov_avoid_work_freq": "avoid_work"}
STABLE = ["yougov_avoid_crowds_freq", "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]
DRIFT = ["yougov_mask_freq", "yougov_stay_home_freq", "yougov_avoid_restaurants_freq"]
N_TEST = 12


def load_df():
    nat = pd.read_csv(PANEL + r"\us_national_week.csv", parse_dates=["week_start"])
    panel = pd.read_csv(PANEL + r"\us_state_week.csv", parse_dates=["week_start"])
    pop_sum = panel.groupby("week_start").apply(
        lambda g: np.average(g["pol_stringency"], weights=g["population"])
        if g["pol_stringency"].notna().all() else g["pol_stringency"].mean(),
        include_groups=False)
    burden = panel.groupby("week_start").agg(cases=("new_cases", "sum"),
                                             pop=("population", "sum")).reset_index()
    burden["per100k"] = burden["cases"] / burden["pop"] * 1e5
    drv = pd.DataFrame({"week_start": pop_sum.index,
                        "stringency": pop_sum.to_numpy()}).merge(
        burden[["week_start", "per100k"]], on="week_start")
    drv["log_burden"] = np.log10(drv["per100k"].clip(lower=0.01))
    df = nat.merge(drv, on="week_start", how="left")
    df = df[df["n_total"] >= 50].reset_index(drop=True)
    return df, burden
