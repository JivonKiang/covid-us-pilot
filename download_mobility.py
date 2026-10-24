# -*- coding: utf-8 -*-
"""下载 Google Mobility 存档 zip（98MB，直连 gstatic）——行为数据的另一半"""
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(BASE, "data", "raw", "region_mobility_reports.zip")
URL = "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip"
MIN_SIZE = 90_000_000

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-pilot/0.1"


def main():
    if os.path.exists(DEST) and os.path.getsize(DEST) >= MIN_SIZE:
        print(f"SKIP 已存在 {os.path.getsize(DEST):,} B")
        return
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    t0 = time.time()
    tmp = DEST + ".part"
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        n = 0
        while True:
            chunk = resp.read(1 << 22)  # 4 MB
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
            if n // (1 << 24) != (n - len(chunk)) // (1 << 24):
                print(f"   ... {n:,} B  ({n/(1<<20):.0f} MB)", flush=True)
    os.replace(tmp, DEST)
    print(f"OK {os.path.getsize(DEST):,} B in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
