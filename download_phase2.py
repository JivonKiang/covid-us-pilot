# -*- coding: utf-8 -*-
"""M1 数据管线 · 第二批下载：YouGov 美国/中国行为数据 + Google Mobility 98MB 存档"""
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

GH = "https://gh-proxy.com/https://raw.githubusercontent.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-pilot/0.1"

FILES = [
    ("yougov_united-states.zip", f"{GH}/YouGov-Data/covid-19-tracker/master/data/united-states.zip", 1_000_000),
    ("yougov_china.csv", f"{GH}/YouGov-Data/covid-19-tracker/master/data/china.csv", 10_000_000),
    ("region_mobility_reports.zip", "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip", 90_000_000),
]


def fetch(dest: str, url: str, min_size: int):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"SKIP {os.path.basename(dest)} ({os.path.getsize(dest):,} B 已存在)", flush=True)
        return
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    t0 = time.time()
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        n = 0
        while True:
            chunk = resp.read(1 << 22)  # 4 MB
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
            if n >= (1 << 24) and (n % (1 << 24)) < (1 << 22):
                print(f"   ... {n/(1<<20):.0f} MB", flush=True)
    os.replace(tmp, dest)
    print(f"OK   {os.path.basename(dest)} {os.path.getsize(dest):,} B in {time.time()-t0:.1f}s", flush=True)


def main():
    for name, url, min_size in FILES:
        try:
            fetch(os.path.join(RAW, name), url, min_size)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
