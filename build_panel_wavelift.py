# -*- coding: utf-8 -*-
"""阶段B面板: 开放潮窗口 2022-11-26~2023-01-20 (W1..W8, W9 弃用).
BESI/BHII 周均值 + JHU 周新增 + OxCGRT 周stringency. 口径见 M11 汇报「阶段 B 预登记」."""
import glob
import re

import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
RAW = ROOT + r"\data\raw"
OUT = ROOT + r"\data\panel\wavelift_week.csv"

W0 = pd.Timestamp(2022, 11, 26)  # W1 周六起始
N_WEEKS = 8
UNITS = {"全国": "national", "广东": "Guangdong", "河南": "Henan", "四川": "Sichuan",
         "北京": "Beijing", "江苏": "Jiangsu", "湖北": "Hubei", "浙江": "Zhejiang"}
EXCLUDE_JHU = {"Hong Kong", "Macau", "Unknown"}  # 全国 burden = 大陆31省级单位


def parse_time(t):
    s = f"{float(t):.2f}"  # 修正 Excel 浮点吞零: 1.1 -> '1.10'
    m, d = s.split(".")
    y = 2022 if int(m) >= 11 else 2023
    return pd.Timestamp(y, int(m), int(d))


# --- 1. 百度 wave1 -> 日度 -> 周均值 ---
files = glob.glob(RAW + r"\baidu_epi\**\*November 26*\*.xlsx", recursive=True)
rows = []
for f in files:
    name = f.replace("/", "\\").split("\\")[-1]
    hit = [u for k, u in UNITS.items() if k in name]
    if not hit:
        continue
    df = pd.read_excel(f, sheet_name="Sheet1")
    df.columns = ["time", "bhii", "besi"][: len(df.columns)]
    for _, r in df.iterrows():
        dt = parse_time(r["time"])
        wk = (dt - W0).days // 7 + 1
        if 1 <= wk <= N_WEEKS:
            rows.append(dict(unit=hit[0], date=dt, week=wk,
                             BESI=float(r["besi"]), BHII=float(r["bhii"])))
baidu = pd.DataFrame(rows)
print(f"[baidu] 日度行={len(baidu)} 单元={sorted(baidu.unit.unique())} "
      f"范围={baidu.date.min().date()}..{baidu.date.max().date()}")

baidu_w = baidu.groupby(["unit", "week"]).agg(
    BESI=("BESI", "mean"), BHII=("BHII", "mean"), n_days=("date", "nunique")).reset_index()
baidu_w["BESI_log"] = np.log10(baidu_w["BESI"].clip(lower=1e-6))
baidu_w["BHII_log"] = np.log10(baidu_w["BHII"].clip(lower=1e-6))
bad = baidu_w[baidu_w["n_days"] != 7]
if len(bad):
    print("[baidu][warn] 非整周:", bad.to_dict("records"))

# --- 2. JHU 中国省级 -> 周新增 (大陆31省; national=求和) ---
jhu = pd.read_csv(RAW + r"\jhu_ts_confirmed_global.csv")
date_cols = [c for c in jhu.columns if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", c)]
chn = jhu[(jhu["Country/Region"] == "China") & ~jhu["Province/State"].isin(EXCLUDE_JHU)]
print(f"[jhu] 大陆省级行={len(chn)}")
cum = chn.set_index("Province/State")[date_cols].T
cum.index = pd.to_datetime(cum.index, format="%m/%d/%y")
daily = cum.sort_index().diff().clip(lower=0)
daily = daily.loc[(daily.index >= W0) & (daily.index < W0 + pd.Timedelta(days=7 * N_WEEKS))]
dd = daily.reset_index().melt(id_vars="index", var_name="unit", value_name="cases")
dd = dd.rename(columns={"index": "date"})
dd["week"] = (dd["date"] - W0).dt.days // 7 + 1
jhu_prov = dd.groupby(["unit", "week"])["cases"].sum().reset_index()
jhu_nat = dd.groupby("week")["cases"].sum().reset_index()
jhu_nat["unit"] = "national"
jhu_w = pd.concat([jhu_prov, jhu_nat], ignore_index=True)
jhu_w["log_burden"] = np.log10(jhu_w["cases"].clip(lower=0) + 1)

# --- 3. OxCGRT 省级 -> 周均值 (national=31省均值) ---
ox = pd.read_csv(RAW + r"\oxcgrt_chn.csv")
ox["Date"] = pd.to_datetime(ox["Date"], format="%Y%m%d")
win = ox[(ox["Date"] >= W0) & (ox["Date"] < W0 + pd.Timedelta(days=7 * N_WEEKS))].copy()
win["week"] = (win["Date"] - W0).dt.days // 7 + 1
ox_prov = win.groupby(["RegionName", "week"])["StringencyIndex_Average"].mean().reset_index()
ox_prov = ox_prov.rename(columns={"RegionName": "unit", "StringencyIndex_Average": "stringency"})
ox_nat = win.groupby("week")["StringencyIndex_Average"].mean().reset_index()
ox_nat["RegionName"] = "national"
ox_nat = ox_nat.rename(columns={"RegionName": "unit", "StringencyIndex_Average": "stringency"})
ox_w = pd.concat([ox_prov, ox_nat], ignore_index=True)

# --- 4. 合并输出 ---
panel = baidu_w.merge(jhu_w, on=["unit", "week"], how="left") \
               .merge(ox_w, on=["unit", "week"], how="left")
panel = panel.sort_values(["unit", "week"]).reset_index(drop=True)
panel.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"[out] {OUT}  rows={len(panel)} (预期 8单元 x 8周 = 64)")
print("\n缺失统计:")
print(panel.isna().sum().to_dict())
key = panel[panel["unit"].isin(["national", "Beijing", "Guangdong"])]
print("\n样例 (national/Beijing/Guangdong 各周):")
print(key[["unit", "week", "BESI", "BHII", "cases", "stringency"]].round(1).to_string(index=False))
