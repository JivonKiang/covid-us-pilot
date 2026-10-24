# -*- coding: utf-8 -*-
"""M11 前置: YouGov 中国数据探查 —— 波次日历、城市分布、m 系列行为列中美对齐."""
import pandas as pd

RAW = r"e:\ai agent\research\covid-us-pilot\data\raw"
CN = RAW + r"\yougov_china.csv"
US = RAW + r"\yougov\united-states.csv"

cn = pd.read_csv(CN, low_memory=False)
t = pd.to_datetime(cn["endtime"], errors="coerce")
cn["_m"] = t.dt.to_period("M")

print("=== 1. 月 x 波次样本量 ===")
print(cn.groupby(["_m"]).size())

print("\n=== 2. city 分布 top15 ===")
print(cn["city"].value_counts().head(15))

print("\n=== 3. m 系列非空率 (中国) ===")
mcn = [c for c in cn.columns if c.startswith("m")]
for c in mcn:
    n = cn[c].notna().sum()
    if n > 0:
        print(f"  {c}: {n} ({n/len(cn)*100:.0f}%)")

print("\n=== 4. 美国文件 m 系列列名 ===")
us_cols = pd.read_csv(US, nrows=2, low_memory=False).columns
mus = [c for c in us_cols if c[:2] in ("m1", "m2", "m3", "m4", "m5", "m6", "m7")]
print(mus)

print("\n=== 5. 中国 m 取值分布 (关键列) ===")
for c in ["m2", "m3", "m4_1", "m4_2", "m4_3", "m4_4", "m5_1", "m6_1", "m6_2", "m7_1"]:
    if c in cn.columns:
        print(f"  {c}:", dict(cn[c].value_counts(dropna=True).head(4)))
