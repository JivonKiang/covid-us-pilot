# -*- coding: utf-8 -*-
"""阶段B预检2修正版: 时间列浮点吞零验证 + JHU 省级窗口覆盖 + OxCGRT 12月后动态."""
import glob
import re

import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
RAW = ROOT + r"\data\raw"


def parse_time_to_date(t, year_of_prev):
    """时间 '11.26'/'1.10' -> Timestamp; year_of_prev: 上一行的年份 (跨年判断用)."""
    s = f"{float(t):.2f}"  # 1.1 -> '1.10', 11.26 -> '11.26'
    m, d = s.split(".")
    m, d = int(m), int(d)
    y = 2022 if m == 11 or m == 12 else 2023
    return pd.Timestamp(y, m, d)


print("=== 1. wave1 时间列修正解析 (安徽 + 全国) ===")
for tag, pat in [("安徽", r"\baidu_epi\**\*November 26*\*安徽*"),
                 ("全国", r"\baidu_epi\**\*November 26*\*全国*")]:
    fp = glob.glob(RAW + pat, recursive=True)[0]
    df = pd.read_excel(fp, sheet_name="Sheet1")
    df.columns = ["time", "bhii", "besi"][: len(df.columns)]
    dates = [parse_time_to_date(t, None) for t in df["time"]]
    s = pd.Series(dates)
    exp = pd.date_range("2022-11-26", "2023-01-25")
    missing = sorted(set(exp) - set(s))
    print(f"[{tag}] 行={len(df)} unique日期={s.nunique()} 范围={s.min().date()}..{s.max().date()} "
          f"缺失={len(missing)} {missing if missing else ''} 重复={len(s) - s.nunique()}")

print("\n=== 2. JHU global 中国省级窗口覆盖 ===")
jhu = pd.read_csv(RAW + r"\jhu_ts_confirmed_global.csv")
date_cols = [c for c in jhu.columns if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", c)]
w_cols = [c for c in date_cols if pd.Timestamp(c) >= pd.Timestamp("2022-11-26")]
print(f"窗口内日期列: {len(w_cols)} ({w_cols[0]} .. {w_cols[-1]})")
chn = jhu[(jhu["Country/Region"] == "China") & (jhu["Province/State"].notna())]
print(f"中国省级行: {len(chn)} -> {sorted(chn['Province/State'].unique())}")
sub = chn.set_index("Province/State")[w_cols]
diffs = sub.diff(axis=1).iloc[:, 1:]
active = (diffs.abs() > 0).sum(axis=1)
print("窗口内有新增的省份数:", (active > 0).sum(), "/", len(sub))
for p in ["Beijing", "Guangdong", "Henan", "Anhui"]:
    if p in sub.index:
        print(p, dict(sub.loc[p, w_cols[-4:]]))

print("\n=== 3. OxCGRT 2022-11..2023-01 stringency 动态 ===")
ox = pd.read_csv(RAW + r"\oxcgrt_chn.csv")
ox["Date"] = pd.to_datetime(ox["Date"], format="%Y%m%d")
sc = ox[ox["RegionName"].notna()][["Date", "RegionName", "StringencyIndex_Average"]]
for d in ["2022-11-26", "2022-12-07", "2022-12-20", "2023-01-01", "2023-01-08", "2023-01-25"]:
    day = sc[sc["Date"] == d]
    v = day["StringencyIndex_Average"]
    print(f"{d}: n={len(day)} mean={v.mean():.1f} min={v.min():.1f} max={v.max():.1f} nan={v.isna().sum()}")
print("窗口内 RegionName 数:", sc[sc['Date'] >= '2022-11-26']["RegionName"].nunique())
print("样例北京 12 月逐周:", sc[(sc["RegionName"] == "Beijing") & (sc["Date"] >= "2022-12-01") &
      (sc["Date"] <= "2022-12-31")].set_index("Date")["StringencyIndex_Average"].iloc[::5].to_dict())
