# -*- coding: utf-8 -*-
"""探查 yougov_china.csv 的完整时间覆盖: 按年/月分布, 确认开放潮 (2022-12/2023) 是否有波次."""
import pandas as pd

y = pd.read_csv(r"e:\ai agent\research\covid-us-pilot\data\raw\yougov_china.csv",
                low_memory=False)
t = pd.to_datetime(y["endtime"], errors="coerce")
print(f"总行数 {len(y)}, endtime 无效 {t.isna().sum()}")
print(f"范围: {t.min()} -> {t.max()}")
print("\n按年:")
print(t.dt.year.value_counts().sort_index().to_string())
print("\n按月 (最后 24 个月):")
mc = t.dt.to_period("M").value_counts().sort_index()
print(mc.tail(24).to_string())
print(f"\n2021+ 行数: {(t >= '2021-01-01').sum()}")
beh = ["i12_health_1", "i12_health_6", "i12_health_7", "i12_health_8",
       "i12_health_11", "i12_health_13", "i12_health_20"]
late = y[t >= "2021-01-01"]
if len(late):
    print("2021+ 行为列有效率:", {b: round(late[b].notna().mean(), 2) for b in beh})
