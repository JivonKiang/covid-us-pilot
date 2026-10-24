# -*- coding: utf-8 -*-
"""M11 中国省级周面板：JHU global(中国省) + OxCGRT CHN(省) + YouGov China(国家级行为)。

输出（列结构与 us_state_week.csv / us_national_week.csv 同构，便于复用下游管线）：
  data/panel/cn_prov_week.csv      省 × 周主面板（3 省 × 49 周）
  data/panel/cn_national_week.csv  YouGov 中国国家级周度行为表（加权）
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "panel")
os.makedirs(OUT, exist_ok=True)

# ---------- 常量 ----------
WEEKS = pd.date_range("2020-01-06", "2020-12-07", freq="7D")  # 周一锚点（覆盖 YouGov 1/5-12/8）
PROVINCES = ["Beijing", "Guangdong", "Hubei"]
POP_2020 = {  # 第七次全国人口普查（2020-11）
    "Beijing": 21_893_095,
    "Guangdong": 126_012_510,
    "Hubei": 57_752_557,
}
POL_COLS = {  # 与美国侧同映射；中国文件若缺列自动跳过并报告
    "StringencyIndex_Average": "pol_stringency",
    "ContainmentHealthIndex_Average": "pol_containment",
    "GovernmentResponseIndex_Average": "pol_govresp",
    "C1M_School closing": "pol_c1_school",
    "C2M_Workplace closing": "pol_c2_workplace",
    "C3M_Cancel public events": "pol_c3_events",
    "C4M_Restrictions on gatherings": "pol_c4_gather",
    "C6M_Stay at home requirements": "pol_c6_stayhome",
    "C7M_Restrictions on internal movement": "pol_c7_internal",
    "H6M_Facial Coverings": "pol_h6_masks",
    "E1_Income support": "pol_e1_income",
}
BEH_COLS = {  # 与美国侧同映射
    "i12_health_1": "yougov_mask_freq",
    "i12_health_6": "yougov_stay_home_freq",
    "i12_health_7": "yougov_avoid_restaurants_freq",
    "i12_health_8": "yougov_avoid_crowds_freq",
    "i12_health_11": "yougov_avoid_shops_freq",
    "i12_health_13": "yougov_avoid_work_freq",
    "i12_health_20": "yougov_hands_freq",
}
FREQ_MAP = {"Not at all": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3, "Always": 4}

week_ends = WEEKS + pd.Timedelta(days=6)

print("== 1/4 JHU 负担（中国省） ==")
META = ["Province/State", "Country/Region", "Lat", "Long"]
cums = {}
for kind, fname in (("c", "jhu_ts_confirmed_global.csv"), ("d", "jhu_ts_deaths_global.csv")):
    df = pd.read_csv(os.path.join(RAW, fname))
    df = df[df["Country/Region"] == "China"]
    date_cols = [c for c in df.columns if c not in META]
    cum = df.set_index("Province/State")[date_cols].T
    cum.index = pd.to_datetime(cum.index, format="%m/%d/%y")
    cum = cum[[p for p in PROVINCES]]  # 缺省会 KeyError，主动暴露
    cums[kind] = cum.astype(float)
    print(f"  {fname}: 中国省级 {cum.shape[1]} 行, 日期 {cum.index[0].date()} -> {cum.index[-1].date()}")


def weekly_new(cum: pd.DataFrame) -> pd.DataFrame:
    """周新增 = 周日累计 - 上周日累计（按位置对齐）。"""
    a = cum.reindex(week_ends)
    b = cum.reindex(week_ends - pd.Timedelta(days=7))
    return pd.DataFrame(a.to_numpy() - b.to_numpy(), index=WEEKS, columns=cum.columns)


new_cases = weekly_new(cums["c"])
new_deaths = weekly_new(cums["d"])
neg_c = int((new_cases < 0).sum().sum())
neg_d = int((new_deaths < 0).sum().sum())
print(f"  周新增为负的格子: 确诊 {neg_c}, 死亡 {neg_d}（数据修正导致，不改数）")

pop_v = pd.Series({p: POP_2020[p] for p in PROVINCES}).astype(float)
rate_c = new_cases.div(pop_v * 1e-5, axis=1)   # 每 10 万人
rate_d = new_deaths.div(pop_v * 1e-5, axis=1)

burden = pd.concat(
    {
        "new_cases": new_cases.stack(),
        "new_deaths": new_deaths.stack(),
        "new_cases_per100k": rate_c.stack(),
        "new_deaths_per100k": rate_d.stack(),
    },
    axis=1,
).reset_index()
burden.columns = ["week_start", "state", "new_cases", "new_deaths",
                  "new_cases_per100k", "new_deaths_per100k"]
burden["population"] = burden["state"].map(POP_2020)

print("== 2/4 OxCGRT 政策（中国省） ==")
o = pd.read_csv(os.path.join(RAW, "oxcgrt_chn.csv"), low_memory=False)
o = o[(o["Jurisdiction"] == "STATE_TOTAL") & (o["RegionName"].isin(PROVINCES))].copy()
o["date"] = pd.to_datetime(o["Date"].astype(str), format="%Y%m%d")
o["week_start"] = o["date"] - pd.to_timedelta(o["date"].dt.weekday, unit="D")
missing_pol = [c for c in POL_COLS if c not in o.columns]
if missing_pol:
    print(f"  [警告] 中国文件缺政策列: {missing_pol}")
present_pol = {s: d for s, d in POL_COLS.items() if s in o.columns}
ox_missing_raw = {}
for src in present_pol:
    o[src] = pd.to_numeric(o[src], errors="coerce")
    ox_missing_raw[present_pol[src]] = float(o[src].isna().mean())
pol_week = (o.groupby(["RegionName", "week_start"])[list(present_pol)].mean()
            .rename(columns=present_pol).reset_index()
            .rename(columns={"RegionName": "state"}))
print(f"  三省日度行数: {len(o)}, 周聚合后 {len(pol_week)} 行")

print("== 3/4 YouGov 行为（中国国家级） ==")
y = pd.read_csv(os.path.join(RAW, "yougov_china.csv"), low_memory=False)
y["date"] = pd.to_datetime(y["endtime"], errors="coerce")
y["week_start"] = (y["date"] - pd.to_timedelta(y["date"].dt.weekday, unit="D")).dt.normalize()
for src, dst in BEH_COLS.items():
    y[src] = y[src].map(FREQ_MAP)

nat = y.dropna(subset=["week_start"]).groupby("week_start")
rows = {}
rows["n_total"] = nat.size()
w = nat["weight"].sum()
for src, dst in BEH_COLS.items():
    ok = y[src].notna()
    prod = (y.loc[ok].assign(_x=y.loc[ok, src] * y.loc[ok, "weight"])
            .groupby("week_start")[["_x", "weight"]].sum())
    rows[dst] = prod["_x"] / prod["weight"]
    rows[dst + "_n"] = y[ok].groupby("week_start").size()
ok_always = y["i12_health_1"].eq(4)
rows["yougov_mask_always_share"] = (
    y[ok_always].groupby("week_start")["weight"].sum() / w
)
nat_week = pd.DataFrame(rows).reset_index().rename(columns={"index": "week_start"})
nat_week = nat_week.sort_values("week_start").reset_index(drop=True)

print("== 4/4 合并 ==")
grid = pd.MultiIndex.from_product(
    [PROVINCES, WEEKS], names=["state", "week_start"]).to_frame(index=False)
panel = (grid
         .merge(burden, on=["state", "week_start"], how="left")
         .merge(pol_week, on=["state", "week_start"], how="left")
         .merge(nat_week, on="week_start", how="left"))

panel_path = os.path.join(OUT, "cn_prov_week.csv")
panel.to_csv(panel_path, index=False)
nat_path = os.path.join(OUT, "cn_national_week.csv")
nat_week.to_csv(nat_path, index=False)
print(f"面板: {panel.shape[0]} 行 = {len(PROVINCES)} 省 × {len(WEEKS)} 周 -> {panel_path}")
print(f"国家级行为表: {nat_week.shape[0]} 周 -> {nat_path}")

# ---------- 质量信息 ----------
print("\n---- 质量信息 ----")
cover = {c: 100.0 * panel[c].notna().mean() for c in panel.columns if c not in ("state", "week_start")}
print("面板列覆盖率（%）:")
for c, v in sorted(cover.items(), key=lambda kv: -kv[1]):
    print(f"  {c}: {v:.0f}%")
print(f"\n政策原始日度缺失率: " + ", ".join(f"{k} {v*100:.1f}%" for k, v in sorted(ox_missing_raw.items(), key=lambda kv: -kv[1])))
n_ok = int((nat_week["n_total"] >= 30).sum())
print(f"\n行为周样本量: 中位 {nat_week['n_total'].median():.0f}, "
      f"范围 {nat_week['n_total'].min():.0f}-{nat_week['n_total'].max():.0f}, "
      f"n>=30 的周数 {n_ok}/{len(nat_week)}")
print(f"首周 {nat_week['week_start'].iloc[0].date()} n={int(nat_week['n_total'].iloc[0])}, "
      f"末周 {nat_week['week_start'].iloc[-1].date()} n={int(nat_week['n_total'].iloc[-1])}")
print("\n六行为全程均值（weight 加权, 0-4）:")
for dst in ["yougov_mask_freq", "yougov_stay_home_freq", "yougov_avoid_restaurants_freq",
            "yougov_avoid_crowds_freq", "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]:
    m = np.average(nat_week[dst].dropna(), weights=nat_week.loc[nat_week[dst].notna(), "n_total"])
    print(f"  {dst}: {m:.2f}")
print("\n抽样检查（湖北 2020-02-03 / 2020-04-06 两周）:")
sample = panel[(panel["state"] == "Hubei")
               & (panel["week_start"].isin([pd.Timestamp("2020-02-03"), pd.Timestamp("2020-04-06")]))]
print(sample[["state", "week_start", "new_cases", "new_deaths", "pol_c1_school",
              "yougov_mask_freq", "n_total"]].to_string(index=False))
