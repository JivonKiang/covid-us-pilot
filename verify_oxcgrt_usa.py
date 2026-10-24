# -*- coding: utf-8 -*-
"""验证 OxCGRT_USA_state.csv 州级数据可用性：RegionName 州数、政策列、缺失率。"""
import csv
import os
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(BASE, "data", "raw", "oxcgrt_usa_state.csv")

with open(TARGET, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

print("列名（全部 61 列）:")
print(header)

i_name = header.index("RegionName")
i_code = header.index("RegionCode")
i_jur = header.index("Jurisdiction")

states = sorted({r[i_name] for r in rows if r[i_jur] == "STATE_TOTAL" and r[i_name]})
print(f"\nSTATE_TOTAL 行的 RegionName 种类数: {len(states)}")
print(f"州/辖区列表: {states}")

# 每个州的行数（应都≈1096 天）
cnt = Counter(r[i_name] for r in rows if r[i_jur] == "STATE_TOTAL")
row_counts = sorted(set(cnt.values()))
print(f"每州行数分布: {row_counts}（共 {len(cnt)} 个辖区）")

# 抽查主政策指标列的缺失率（州级，2020-2022 全期）
main_cols = [c for c in header if c.startswith(("C1_", "C2_", "C3_", "C4_", "C5_", "C6_", "C7_", "C8_", "H6_", "H8_", "V2_"))]
print(f"\n主政策列缺失率（州级行）:")
idx = {c: header.index(c) for c in main_cols}
n_state = sum(1 for r in rows if r[i_jur] == "STATE_TOTAL")
for c in main_cols:
    k = idx[c]
    miss = sum(1 for r in rows if r[i_jur] == "STATE_TOTAL" and r[k] == "")
    print(f"  {c}: {miss}/{n_state} ({miss/n_state*100:.1f}%)")
