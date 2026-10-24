# -*- coding: utf-8 -*-
"""验证 YouGov 关键行为列的取值编码与时间/州字段质量。"""
import os
from collections import Counter

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(BASE, "data", "raw", "yougov", "united-states.csv")

cols = ["RecordNo", "endtime", "state", "inputstate", "qweek", "weight",
        "i1_health", "i13_health",
        "i12_health_1", "i12_health_6", "i12_health_7", "i12_health_8",
        "i12_health_11", "i12_health_13", "i12_health_20"]
df = pd.read_csv(P, usecols=cols, low_memory=False)
print(f"读取 {len(df)} 行")

for c in ["i1_health", "i13_health", "i12_health_1", "i12_health_6", "i12_health_8", "i12_health_20"]:
    vc = df[c].value_counts(dropna=False).head(8)
    print(f"\n{c} 取值分布:")
    for v, n in vc.items():
        print(f"  {v!r}: {n}")

print("\nweight 统计:", df["weight"].describe().round(3).to_dict())

print("\nendtime 范围:", df["endtime"].min(), "->", df["endtime"].max())
print("qweek 示例:", df["qweek"].dropna().unique()[:5])
print("state 唯一值数:", df["state"].nunique())
print("state 示例:", sorted(df["state"].dropna().unique())[:8])
print("inputstate 唯一值数:", df["inputstate"].nunique())
print("inputstate 示例:", sorted(df["inputstate"].dropna().unique())[:8])
