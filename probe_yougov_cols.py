# -*- coding: utf-8 -*-
"""探查 YouGov 美国数据的全部列名，筛选行为类候选列。"""
import csv
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(BASE, "data", "raw", "yougov", "united-states.csv")
OUT = os.path.join(BASE, "data", "yougov_columns.txt")

with open(P, encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(header))
print(f"共 {len(header)} 列，全部列名已写入 data/yougov_columns.txt")

# 行为候选：口罩、回避聚集、洗手、保持距离、症状、接触
pat = re.compile(
    r"mask|facial|cover|avoid|behav|i1_health|i12_health|i13|vaccine|wash|sanitis"
    r"|distance|social|gather|crowd|worried|concern|doctor|symptom|test",
    re.I,
)
cands = [c for c in header if pat.search(c)]
print(f"\n候选行为列 {len(cands)} 个:")
for c in cands:
    print(" ", c)
