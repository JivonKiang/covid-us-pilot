# -*- coding: utf-8 -*-
"""M11 前置: YouGov 中国 i12 系列可用性验证（中美同款行为列）."""
import pandas as pd

RAW = r"e:\ai agent\research\covid-us-pilot\data\raw"
CN = RAW + r"\yougov_china.csv"
US = RAW + r"\yougov\united-states.csv"

BEH = {"i12_health_1": "mask", "i12_health_6": "stay_home",
       "i12_health_7": "avoid_restaurants", "i12_health_8": "avoid_crowds",
       "i12_health_11": "avoid_shops", "i12_health_13": "avoid_work",
       "i12_health_20": "hands"}

cn = pd.read_csv(CN, low_memory=False)
cn["date"] = pd.to_datetime(cn["endtime"], errors="coerce")
cn["week_start"] = (cn["date"] - pd.to_timedelta(cn["date"].dt.weekday, unit="D")).dt.normalize()

print("=== 中国 i12 非空率 + 取值分布 ===")
for src, name in BEH.items():
    if src not in cn.columns:
        print(f"  {src} ({name}): 列不存在")
        continue
    s = cn[src]
    ok = s.notna() & ~s.astype(str).str.strip().isin(["", " "])
    vc = s[ok].value_counts().head(5)
    print(f"  {src} ({name}): 有效 {ok.sum()} ({ok.mean()*100:.0f}%) | 取值 {dict(vc)}")

print("\n=== 中国 mask 周轨迹 (有效>=30 的周) ===")
FREQ = {"Not at all": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3, "Always": 4}
m = cn["i12_health_1"].map(FREQ)
ok = m.notna()
traj = cn.assign(_v=m)[ok].groupby("week_start")["_v"].agg(["size", "mean"])
print(traj[traj["size"] >= 30].round(3))

print("\n=== 美国对照: mask 前几周均值 ===")
usc = ["RecordNo", "endtime", "i12_health_1"]
us = pd.read_csv(US, usecols=usc, low_memory=False)
us["date"] = pd.to_datetime(us["endtime"], format="%d/%m/%Y %H:%M", errors="coerce")
us["week_start"] = (us["date"] - pd.to_timedelta(us["date"].dt.weekday, unit="D")).dt.normalize()
um = us["i12_health_1"].map(FREQ)
ut = us.assign(_v=um)[um.notna()].groupby("week_start")["_v"].agg(["size", "mean"])
print(ut.head(8).round(3))
