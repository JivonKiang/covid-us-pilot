# -*- coding: utf-8 -*-
"""阶段B预检: 本地资产对开放潮窗口(2022-11-26~2023-01-25)的覆盖 + wave1 全量解析."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
PANEL = ROOT + r"\data\panel"
RAW = ROOT + r"\data\raw"
W1 = ("2022-11-26", "2023-01-25")

print("=== 1. cn_prov_week.csv ===")
prov = pd.read_csv(PANEL + r"\cn_prov_week.csv")
print("列:", list(prov.columns))
if "week_start" in prov.columns:
    print("时间范围:", prov["week_start"].min(), "..", prov["week_start"].max())
    print("省份数:", prov["province"].nunique() if "province" in prov.columns else "?")

print("\n=== 2. oxcgrt_chn.csv ===")
ox = pd.read_csv(RAW + r"\oxcgrt_chn.csv")
print("列:", list(ox.columns)[:20], "... 共", len(ox.columns), "列")
for c in ["Date", "date", "DateString"]:
    if c in ox.columns:
        ox[c] = pd.to_datetime(ox[c], format="%Y%m%d" if ox[c].dtype != object or ox[c].astype(str).str.len().max() == 8 else None, errors="coerce")
        sub = ox[(ox[c] >= "2022-11-01") & (ox[c] <= "2023-02-28")]
        print(f"时间列 {c}: {ox[c].min()} .. {ox[c].max()}, 窗口内行数 {len(sub)}")
        sc = [x for x in ox.columns if "stringency" in x.lower()]
        print("stringency 列:", sc)
        if len(sub) and sc:
            print(sub.groupby(c)[sc[0]].describe().head(8))
        break

print("\n=== 3. JHU 省级时序文件 ===")
for pat in [r"\jhu*", r"\*time_series*", r"\*csse*"]:
    fs = glob.glob(RAW + pat) + glob.glob(RAW + r"\**" + pat, recursive=True)
    if fs:
        print("候选:", fs[:10])

print("\n=== 4. wave1 全量解析 ===")
w1_files = sorted(glob.glob(RAW + r"\baidu_epi\**\*November 26*\*.xlsx", recursive=True))
print(f"wave1 文件: {len(w1_files)}")
rows = []
for fp in w1_files:
    b = os.path.basename(fp)
    prov_name = b.split("疫情大数据")[0].replace(".xlsx", "")
    df = pd.read_excel(fp, sheet_name="Sheet1")
    df.columns = ["time", "bhii", "besi"][: len(df.columns)]
    # 时间 "11.26" -> 日期 (wave1: 11/26-12/31=2022, 1/x=2023)
    dates = []
    for t in df["time"]:
        m, d = str(t).split(".")
        y = 2022 if int(m) == 11 or int(m) == 12 else 2023
        dates.append(pd.Timestamp(y, int(m), int(d)))
    df["date"] = pd.date_range(W1[0], W1[0], periods=1)[0]  # placeholder
    df["date"] = pd.Series(dates)
    exp = pd.date_range(W1[0], W1[1])
    miss = len(exp) - df["date"].nunique()
    rows.append({"prov": prov_name, "days": len(df), "unique_days": df["date"].nunique(),
                 "miss_days": miss, "bhii_nan": df["bhii"].isna().sum(),
                 "besi_nan": df["besi"].isna().sum(),
                 "bhii_min": df["bhii"].min(), "besi_min": df["besi"].min(),
                 "bhii_max": df["bhii"].max()})
r = pd.DataFrame(rows)
print(r.to_string())
print("\n异常文件 (miss>0 或 NaN>0 或 bhii_min<=0):")
print(r[(r["miss_days"] > 0) | (r["bhii_nan"] > 0) | (r["besi_nan"] > 0) |
        (r["bhii_min"] <= 0) | (r["besi_min"] <= 0)].to_string())
