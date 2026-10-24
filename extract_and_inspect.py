# -*- coding: utf-8 -*-
"""M1 数据管线 · 解压与结构检查（不依赖 pandas，标准库即可）"""
import csv
import io
import os
import sys
import zipfile
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
MOB = os.path.join(RAW, "mobility")
YGV = os.path.join(RAW, "yougov")
os.makedirs(MOB, exist_ok=True)
os.makedirs(YGV, exist_ok=True)


def unzip_select(zip_path: str, out_dir: str, needle: str):
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if needle in info.filename:
                target = os.path.join(out_dir, os.path.basename(info.filename))
                if not os.path.exists(target) or os.path.getsize(target) != info.file_size:
                    with z.open(info) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                out.append(target)
    return out


def peek_csv(path: str, max_cols: int = 14, first_data_col_idx: int = None):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        n_rows = sum(1 for _ in r)
    return header, n_rows


def main():
    # 1) 解压
    us_files = unzip_select(os.path.join(RAW, "yougov_united-states.zip"), YGV, "")
    print("YouGov 解压:", [os.path.basename(p) for p in us_files])
    us_mob = unzip_select(os.path.join(RAW, "region_mobility_reports.zip"), MOB, "_US_Region_Mobility_Report.csv")
    print("Mobility US 解压:", [os.path.basename(p) for p in sorted(us_mob)])

    # 2) JHU confirmed（第一列前 12 列 + 最后 2 个日期列 + 行数）
    hdr, n = peek_csv(os.path.join(RAW, "jhu_ts_confirmed_us.csv"))
    print(f"\n[JHU confirmed] {n:,} 行 × {len(hdr)} 列")
    print("   前 12 列:", hdr[:12])
    print("   末 2 列:", hdr[-2:])

    hdr, n = peek_csv(os.path.join(RAW, "jhu_ts_deaths_us.csv"))
    print(f"[JHU deaths]    {n:,} 行 × {len(hdr)} 列")
    print("   前 14 列:", hdr[:14])
    print("   末 2 列:", hdr[-2:])

    # 3) OxCGRT：取美国行数与日期范围
    dates = []
    n_us = 0
    with open(os.path.join(RAW, "oxcgrt_policy_latest.csv"), "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("CountryCode") == "USA":
                n_us += 1
                dates.append(row.get("Date", ""))
    print(f"\n[OxCGRT] USA 行数 {n_us:,}，日期 {min(dates) if dates else '-'} → {max(dates) if dates else '-'}")

    # 4) YouGov US
    for p in us_files:
        hdr, n = peek_csv(p, max_cols=40)
        print(f"\n[YouGov US] {os.path.basename(p)}: {n:,} 行 × {len(hdr)} 列")
        print("   前 40 列名:", hdr[:40])

    # 5) Mobility US（2020）
    if us_mob:
        p2020 = [p for p in us_mob if "2020" in p][0]
        hdr, n = peek_csv(p2020)
        print(f"\n[Mobility 2020_US] {n:,} 行 × {len(hdr)} 列")
        print("   列名:", hdr)

    # 6) pandas 可用性
    try:
        import pandas  # noqa
        print(f"\n[pandas] 可用，版本 {pandas.__version__}")
    except ImportError:
        print("\n[pandas] 未安装 —— 构建 panel 前需 pip install pandas")


if __name__ == "__main__":
    main()
