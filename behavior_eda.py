# -*- coding: utf-8 -*-
"""M3-6 行为层效度预检 · 探索性分析（EDA，产出候选假设，不算验证结论）
W2【探索】：行为指标与政策/负担的领先滞后结构。
行为-政策：corr(behavior_t, policy_{t-k})，政策先行 k 周，行为响应。
行为-负担：corr(burden_{t+k}, behavior_t)，行为先行 k 周，负担后应。
"""
import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
PANEL = ROOT + r"\data\panel"

nat = pd.read_csv(PANEL + r"\us_national_week.csv", parse_dates=["week_start"])
panel = pd.read_csv(PANEL + r"\us_state_week.csv", parse_dates=["week_start"])

BEHAV = ["yougov_mask_freq", "yougov_hands_freq", "yougov_stay_home_freq",
         "yougov_avoid_restaurants_freq", "yougov_avoid_crowds_freq",
         "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]
NICE = {b: b.replace("yougov_", "").replace("_freq", "") for b in BEHAV}

# 国家级驱动序列：政策=人口加权均值；负担=全国 per100k
pop_sum = panel.groupby("week_start").apply(
    lambda g: np.average(g["pol_stringency"], weights=g["population"])
    if g["pol_stringency"].notna().all() else g["pol_stringency"].mean(),
    include_groups=False)
drv = pd.DataFrame({
    "week_start": pop_sum.index,
    "stringency": pop_sum.to_numpy(),
})
burden = panel.groupby("week_start").agg(cases=("new_cases", "sum"),
                                         pop=("population", "sum")).reset_index()
burden["per100k"] = burden["cases"] / burden["pop"] * 1e5
drv = drv.merge(burden[["week_start", "per100k"]], on="week_start")
drv["log_burden"] = np.log10(drv["per100k"].clip(lower=0.01))

# 合并（行为 61 周为主表），剔除低样本周 n<50
df = nat.merge(drv, on="week_start", how="left")
df = df[df["n_total"] >= 50].reset_index(drop=True)
print(f"可用周: {len(df)} / 61（剔除 n<50 后）")
print(f"行为窗口: {df['week_start'].min().date()} -> {df['week_start'].max().date()}")

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return np.nan
    return pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")

lines = ["# 行为层效度预检 · 探索性分析（EDA）\n",
         f"- 可用周：{len(df)}（剔除 n<50 的低样本周后）",
         f"- 窗口：{df['week_start'].min().date()} -> {df['week_start'].max().date()}",
         "- 注：EDA 产出为候选假设，不得事后补登记冒充验证。\n"]

# 行为 -> 政策（政策先行 k 周）
lines.append("## 政策先行 k 周 -> 行为响应（Spearman ρ）\n")
lines.append("| 行为指标 | lag0 | lag1 | lag2 | lag3 | lag4 |")
lines.append("|---|---|---|---|---|---|")
for b in BEHAV:
    row = [NICE[b]]
    for k in range(5):
        row.append(f"{spearman(df[b], df['stringency'].shift(k)):+.2f}")
    lines.append("| " + " | ".join(row) + " |")

# 负担 -> 行为（行为先行 k 周）
lines.append("\n## 行为先行 k 周 -> 负担响应（log10 per100k，Spearman ρ）\n")
lines.append("| 行为指标 | lead0 | lead1 | lead2 | lead3 | lead4 |")
lines.append("|---|---|---|---|---|---|")
for b in BEHAV:
    row = [NICE[b]]
    for k in range(5):
        row.append(f"{spearman(df['log_burden'].shift(-k), df[b]):+.2f}")
    lines.append("| " + " | ".join(row) + " |")

out = PANEL + r"\behavior_eda.md"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"EDA -> {out}")
print("\n".join(lines[6:]))
