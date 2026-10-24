# -*- coding: utf-8 -*-
"""探查中国 YouGov 有效周(n>=30)的连续性 + 三省负担周轨迹形态。"""
import pandas as pd

PANEL = r"e:\ai agent\research\covid-us-pilot\data\panel"

nat = pd.read_csv(PANEL + r"\cn_national_week.csv", parse_dates=["week_start"])
ok = nat[nat["n_total"] >= 30]["week_start"].tolist()
print(f"有效周 n>=30: {len(ok)} 个")
weeks_all = set(nat["week_start"])
for w in sorted(ok):
    prev = w - pd.Timedelta(days=7)
    mark = "连续" if prev in weeks_all else "断档前"
    n = int(nat.loc[nat["week_start"] == w, "n_total"].iloc[0])
    print(f"  {w.date()}  n={n}  {mark}")

# 最长连续段（按日历连续，步长 7 天）
ok_sorted = sorted(ok)
runs, cur = [], [ok_sorted[0]]
for a, b in zip(ok_sorted, ok_sorted[1:]):
    if (b - a).days == 7:
        cur.append(b)
    else:
        runs.append(cur)
        cur = [b]
runs.append(cur)
runs.sort(key=len, reverse=True)
print(f"\n连续段数: {len(runs)}, 最长段 {len(runs[0])} 周: "
      f"{runs[0][0].date()} -> {runs[0][-1].date()}")

panel = pd.read_csv(PANEL + r"\cn_prov_week.csv", parse_dates=["week_start"])
print("\n三省周新增确诊（非零周轨迹摘要）:")
for prov in ["Hubei", "Guangdong", "Beijing"]:
    g = panel[panel["state"] == prov].sort_values("week_start")
    nz = g[g["new_cases"] > 0]
    print(f"  {prov}: 非零周 {len(nz)}/{len(g)}, "
          f"峰值 {g['new_cases'].max():.0f} @ {g.loc[g['new_cases'].idxmax(), 'week_start'].date()}")
    tail = g.tail(12)[["week_start", "new_cases", "new_deaths"]]
    print(tail.to_string(index=False, header=False))
