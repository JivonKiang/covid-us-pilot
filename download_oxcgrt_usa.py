# -*- coding: utf-8 -*-
"""通过 GitHub API 的 download_url 下载 OxCGRT 美国州级全量政策数据并验证。"""
import csv
import json
import os
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
API = "https://gh-proxy.com/https://api.github.com/repos/OxCGRT/covid-policy-tracker/contents/data/United%20States"

TARGET = os.path.join(RAW, "oxcgrt_usa_state.csv")
MIN_SIZE = 5_000_000


def api_download_url() -> str:
    req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        items = json.loads(r.read().decode("utf-8"))
    for it in items:
        if it["name"] == "OxCGRT_USA_latest.csv":
            return it["download_url"]
    raise RuntimeError("未找到 OxCGRT_USA_latest.csv")


def fetch(url: str, dest: str, min_size: int) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"SKIP (已有 {os.path.getsize(dest)/1e6:.1f} MB)")
        return
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(part, dest)
    print(f"OK ({os.path.getsize(dest)/1e6:.1f} MB)")


url = api_download_url()
url = "https://gh-proxy.com/" + url  # raw 直连被墙，走镜像前缀（download_url 分支已是 master）
print(f"download_url: {url}")
fetch(url, TARGET, MIN_SIZE)

# 删除截断的旧文件
OLD = os.path.join(RAW, "oxcgrt_policy_latest.csv")
if os.path.exists(OLD):
    os.remove(OLD)
    print("已删除截断旧文件 oxcgrt_policy_latest.csv")

# ---- 验证 ----
print("\n---- 验证 ----")
jur_dates = {}
n_rows = 0
with open(TARGET, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    j_idx = header.index("Jurisdiction") if "Jurisdiction" in header else None
    d_idx = header.index("Date")
    for row in reader:
        n_rows += 1
        jur = row[j_idx] if j_idx is not None else "N/A"
        date = row[d_idx]
        if jur not in jur_dates:
            jur_dates[jur] = [date, date]
        else:
            if date < jur_dates[jur][0]:
                jur_dates[jur][0] = date
            if date > jur_dates[jur][1]:
                jur_dates[jur][1] = date

print(f"总行数: {n_rows}, 列数: {len(header)}")
print(f"Jurisdiction 种类数: {len(jur_dates)}")
sample = sorted(jur_dates.items())
for jur, (lo, hi) in sample[:5]:
    print(f"  {jur}: {lo} -> {hi}")
print("  ...")
for jur, (lo, hi) in sample[-5:]:
    print(f"  {jur}: {lo} -> {hi}")

policy_cols = [c for c in header if c[:2] in {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"} and "_" not in c]
print(f"\n主政策指标列: {policy_cols}")
