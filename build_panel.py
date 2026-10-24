# -*- coding: utf-8 -*-
"""M1-3 周度州级面板构建：JHU 负担 + OxCGRT 政策 + Google Mobility + YouGov 行为。

输出：
  data/panel/us_state_week.csv      州 × 周主面板（51 辖区 × 133 周）
  data/panel/us_national_week.csv   YouGov 国家级周度行为表（加权）
  data/panel/quality_report.md      数据质量报告
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "panel")
os.makedirs(OUT, exist_ok=True)

# ---------- 常量 ----------
WEEKS = pd.date_range("2020-03-02", "2022-10-10", freq="7D")  # 周一锚点
DC_MAP = {"Washington DC": "District of Columbia"}

POL_COLS = {
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
MOB_COLS = {
    "retail_and_recreation_percent_change_from_baseline": "mob_retail",
    "grocery_and_pharmacy_percent_change_from_baseline": "mob_grocery",
    "parks_percent_change_from_baseline": "mob_parks",
    "transit_stations_percent_change_from_baseline": "mob_transit",
    "workplaces_percent_change_from_baseline": "mob_workplaces",
    "residential_percent_change_from_baseline": "mob_residential",
}
BEH_COLS = {  # YouGov 行为题 -> 面板列名（0=Not at all ... 4=Always）
    "i12_health_1": "yougov_mask_freq",
    "i12_health_6": "yougov_stay_home_freq",
    "i12_health_7": "yougov_avoid_restaurants_freq",
    "i12_health_8": "yougov_avoid_crowds_freq",
    "i12_health_11": "yougov_avoid_shops_freq",
    "i12_health_13": "yougov_avoid_work_freq",
    "i12_health_20": "yougov_hands_freq",
}
FREQ_MAP = {"Not at all": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3, "Always": 4}

print("== 1/5 JHU 负担 ==")
META_C = ["UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Province_State",
          "Country_Region", "Lat", "Long_", "Combined_Key"]
dfc = pd.read_csv(os.path.join(RAW, "jhu_ts_confirmed_us.csv"))
date_cols = [c for c in dfc.columns if c not in META_C]
cum_c = dfc.groupby("Province_State")[date_cols].sum().T
cum_c.index = pd.to_datetime(cum_c.index, format="%m/%d/%y")

dfd = pd.read_csv(os.path.join(RAW, "jhu_ts_deaths_us.csv"))
META_D = META_C + ["Population"]
date_cols_d = [c for c in dfd.columns if c not in META_D]
cum_d = dfd.groupby("Province_State")[date_cols_d].sum().T
cum_d.index = pd.to_datetime(cum_d.index, format="%m/%d/%y")
pop = dfd.groupby("Province_State")["Population"].sum()

# 统一 51 辖区（50 州 + DC），显式剔除领地（Puerto Rico/Guam 等）与邮轮
VALID_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
}
states_raw = (set(cum_d.columns) & set(cum_c.columns)) & VALID_STATES
missing = VALID_STATES - states_raw
assert len(states_raw) == 51, f"辖区数异常: {len(states_raw)}, 缺失: {sorted(missing)}"
STATES = [s for s in sorted(states_raw) if s != "District of Columbia"]
STATES.append("District of Columbia")  # DC 放最后
print(f"辖区数: {len(STATES)}; 确认含 DC: {'District of Columbia' in STATES}")
pop = pop.reindex(STATES)

week_ends = WEEKS + pd.Timedelta(days=6)  # 每周锚点周一 -> 周日


def weekly_new(cum: pd.DataFrame) -> pd.DataFrame:
    """周新增 = 周日累计 - 上周日累计（7 天差分，按位置对齐避免 index 并集）。"""
    cum = cum[STATES]
    a = cum.reindex(week_ends).astype(float)
    b = cum.reindex(week_ends - pd.Timedelta(days=7)).astype(float)
    return pd.DataFrame(a.to_numpy() - b.to_numpy(), index=WEEKS, columns=cum.columns)


new_cases = weekly_new(cum_c)
new_deaths = weekly_new(cum_d)
neg_c = int((new_cases < 0).sum().sum())
neg_d = int((new_deaths < 0).sum().sum())
print(f"周新增为负的格子: 确诊 {neg_c}, 死亡 {neg_d}（数据修正导致，报告记录不改数）")

pop_v = pop.reindex(STATES).astype(float).values
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
# stack 后 MultiIndex 顺序为 (week_start, state)，第一层来自 WEEKS 轴
burden.columns = ["week_start", "state", "new_cases", "new_deaths",
                  "new_cases_per100k", "new_deaths_per100k"]
pop_frame = pd.DataFrame({"state": STATES, "population": pop_v})
burden = burden.merge(pop_frame, on="state", how="left")

print("== 2/5 OxCGRT 政策 ==")
o = pd.read_csv(os.path.join(RAW, "oxcgrt_usa_state.csv"), low_memory=False)
o = o[o["Jurisdiction"] == "STATE_TOTAL"].copy()
o["state"] = o["RegionName"].replace(DC_MAP)
o["date"] = pd.to_datetime(o["Date"].astype(str), format="%Y%m%d")
o["week_start"] = o["date"] - pd.to_timedelta(o["date"].dt.weekday, unit="D")
ox_missing_raw = {}
for src, dst in POL_COLS.items():
    o[src] = pd.to_numeric(o[src], errors="coerce")
    ox_missing_raw[dst] = float(o[src].isna().mean())
pol_week = o.groupby(["state", "week_start"])[list(POL_COLS)].mean().rename(
    columns={s: d for s, d in POL_COLS.items()}).reset_index()

print("== 3/5 Google Mobility ==")
mobs = []
for year in (2020, 2021, 2022):
    f = os.path.join(RAW, "mobility", f"{year}_US_Region_Mobility_Report.csv")
    m = pd.read_csv(f, low_memory=False)
    mobs.append(m)
m = pd.concat(mobs, ignore_index=True)
m = m[m["sub_region_1"].notna() & m["sub_region_2"].isna()].copy()
m["state"] = m["sub_region_1"]
m["date"] = pd.to_datetime(m["date"])
m["week_start"] = m["date"] - pd.to_timedelta(m["date"].dt.weekday, unit="D")
mob_week = m.groupby(["state", "week_start"])[list(MOB_COLS)].mean().rename(
    columns=MOB_COLS).reset_index()
mob_range = (m["date"].min(), m["date"].max())
print(f"Mobility 日期范围: {mob_range[0].date()} -> {mob_range[1].date()}")

print("== 4/5 YouGov 行为 ==")
usecols = ["RecordNo", "endtime", "state", "weight"] + list(BEH_COLS)
y = pd.read_csv(os.path.join(RAW, "yougov", "united-states.csv"),
                usecols=usecols, low_memory=False)
y["state"] = y["state"].replace({" ": np.nan, "": np.nan})
y["date"] = pd.to_datetime(y["endtime"], format="%d/%m/%Y %H:%M", errors="coerce")
y["week_start"] = (y["date"] - pd.to_timedelta(y["date"].dt.weekday, unit="D")).dt.normalize()

for src, dst in BEH_COLS.items():
    y[src] = y[src].map(FREQ_MAP)  # 五档 -> 0-4，其它值（含空）-> NaN

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

# 州 × 周 YouGov 回答数（面板辅助列）
n_state = (y.dropna(subset=["week_start", "state"])
           .groupby(["state", "week_start"]).size().rename("n_yougov_state").reset_index())

print("== 5/5 合并 ==")
grid = pd.MultiIndex.from_product(
    [STATES, WEEKS], names=["state", "week_start"]).to_frame(index=False)
panel = (grid
         .merge(burden, on=["state", "week_start"], how="left")
         .merge(pol_week, on=["state", "week_start"], how="left")
         .merge(mob_week, on=["state", "week_start"], how="left")
         .merge(n_state, on=["state", "week_start"], how="left"))

panel_path = os.path.join(OUT, "us_state_week.csv")
panel.to_csv(panel_path, index=False)
nat_path = os.path.join(OUT, "us_national_week.csv")
nat_week.to_csv(nat_path, index=False)
print(f"面板: {panel.shape[0]} 行 = {len(STATES)} 辖区 × {len(WEEKS)} 周 -> {panel_path}")
print(f"国家级行为表: {nat_week.shape[0]} 周 -> {nat_path}")

# ---------- 质量报告 ----------
print("== 质量报告 ==")


def cover(col: str) -> float:
    return 100.0 * panel[col].notna().mean()


weeks_idx = panel["week_start"].unique()
sections = []
sections.append("# us_state_week 数据质量报告\n")
sections.append(f"- 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}")
sections.append(f"- 维度：{len(STATES)} 辖区 × {len(WEEKS)} 周 = {len(panel)} 行")
sections.append(f"- 周定义：周一锚点，{WEEKS[0].date()} → {WEEKS[-1].date()}"
                f"（最后一周截至今可用数据，Mobility 止于 {mob_range[1].date()}）")
sections.append(f"- 州人口合计：{pop_v.sum()/1e6:.1f} 百万（含 DC）\n")

sections.append("## 列覆盖率\n")
sections.append("| 列组 | 覆盖率 |")
sections.append("|---|---|")
groups = {
    "负担 (new_cases)": ["new_cases", "new_deaths"],
    "率 (per100k)": ["new_cases_per100k", "new_deaths_per100k"],
    "政策 (11 列均值)": list(POL_COLS.values()),
    "Mobility (6 列均值)": list(MOB_COLS.values()),
    "YouGov 州样本数": ["n_yougov_state"],
}
for g, cols in groups.items():
    cs = ", ".join(f"{c} {cover(c):.0f}%" for c in cols)
    sections.append(f"| {g} | {cs} |")
sections.append("")

sections.append("## 负担数据\n")
sections.append(f"- 周新增为负的格子：确诊 {neg_c} / {len(panel)}，死亡 {neg_d}（JHU 数据回溯修正，未改数）")
sections.append(f"- 2022-03 后 JHU 确诊报告趋弱（居家自测不再上报），解读末期确诊需谨慎\n")

sections.append("## 政策数据（OxCGRT 州级，原始日度缺失率）\n")
sections.append("| 列 | 缺失率 |")
sections.append("|---|---|")
for dst, mv in sorted(ox_missing_raw.items(), key=lambda kv: -kv[1]):
    sections.append(f"| {dst} | {mv*100:.1f}% |")
sections.append("")

sections.append("## 行为数据\n")
mob_miss = {c: 100.0 * m[c].isna().mean() for c in MOB_COLS}
worst = max(mob_miss, key=mob_miss.get)
sections.append(f"- Mobility 州级行缺失率最高：{worst} {mob_miss[worst]:.1f}%")
y_week_n = nat_week["n_total"]
low = nat_week.loc[y_week_n < 50, ["week_start", "n_total"]]
sections.append(f"- YouGov 国家级周样本量：中位 {y_week_n.median():.0f}，"
                f"最小 {y_week_n.min():.0f}，最大 {y_week_n.max():.0f}")
if len(low):
    sections.append(f"- 样本量 <50 的周（{len(low)} 个，散布于 2020-06~2021-07，"
                    "与 YouGov 样本轮换相关，见数据边界说明）：")
    for _, r in low.iterrows():
        sections.append(f"  - {pd.Timestamp(r['week_start']).date()}: n={int(r['n_total'])}")
first = nat_week.iloc[0]
sections.append(f"- 首周（{pd.Timestamp(first['week_start']).date()}）样本：口罩题 n={int(first['yougov_mask_freq_n'])}，"
                f"洗手题 n={int(first['yougov_hands_freq_n'])}，总样本 n={int(first['n_total'])}")
state_wk = panel.groupby("state")["n_yougov_state"].mean()
sections.append(f"- YouGov 州×周平均样本量：中位 {state_wk.median():.1f}，"
                f"仅作种子参考（校准主口径用国家级）\n")

sections.append("## 抽样检查（加州 2020-04 与 2021-04 各一周）\n")
sample = panel[(panel["state"] == "California")
               & (panel["week_start"].isin([pd.Timestamp("2020-04-06"), pd.Timestamp("2021-04-05")]))]
sections.append(sample.to_string(index=False))
sections.append("")

qp = os.path.join(OUT, "quality_report.md")
with open(qp, "w", encoding="utf-8") as f:
    f.write("\n".join(sections) + "\n")
print(f"质量报告 -> {qp}")
