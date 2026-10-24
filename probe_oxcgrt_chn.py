# -*- coding: utf-8 -*-
"""快速摸清 oxcgrt_chn.csv 列结构与辖区分布。"""
import csv
import os
from collections import Counter

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
path = os.path.join(RAW, "oxcgrt_chn.csv")

with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    row1 = next(reader)

print("列数:", len(header))
print("前 10 列:", header[:10])
c_cols = [c for c in header if c.startswith("C") and "Flag" not in c and "Wildcard" not in c]
print("C 系主指标列:", c_cols)

# 列索引
idx = {c: i for i, c in enumerate(header)}
jur_col = "Jurisdiction" if "Jurisdiction" in idx else None
region_col = "RegionName" if "RegionName" in idx else None
print(f"Jurisdiction 列: {jur_col}, RegionName 列: {region_col}")

jur_counter = Counter()
date_lo, date_hi = None, None
n_nat = 0
with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        key = (row[idx[jur_col]] if jur_col else "?",
               row[idx[region_col]] if region_col else "?")
        jur_counter[key] += 1
        d = row[idx["Date"]]
        date_lo = d if date_lo is None or d < date_lo else date_lo
        date_hi = d if date_hi is None or d > date_hi else date_hi
        if region_col and not row[idx[region_col]]:
            n_nat += 1

print(f"\n日期范围: {date_lo} -> {date_hi}, 全国级行数(RegionName空): {n_nat}")
print("辖区分布 top8:")
for k, v in jur_counter.most_common(8):
    print(f"  {k}: {v} 行")

# 2020 年全国行 C 系非空率
c_idx = {c: idx[c] for c in c_cols}
n_2020 = 0
nonempty = Counter()
with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        d = row[idx["Date"]]
        if not d.startswith("2020"):
            continue
        if region_col and row[idx[region_col]]:
            continue  # 只看全国行
        n_2020 += 1
        for c, i in c_idx.items():
            if row[i]:
                nonempty[c] += 1
print(f"\n2020 年全国行数: {n_2020}, C 系非空率:")
for c in c_cols:
    print(f"  {c}: {nonempty[c]}/{n_2020} ({100*nonempty[c]/max(n_2020,1):.0f}%)")
