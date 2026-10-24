# -*- coding: utf-8 -*-
"""M1 数据管线 · 批量下载（带检查点：已存在且达到最小尺寸则跳过）

网络约束（实测 2026-10-22）：
- raw.githubusercontent.com / github.com 直连被墙 → 一律走 gh-proxy.com 镜像
- gstatic（Google Mobility 存档）直连可用
- data.cdc.gov 直连 403（弃用，负担序列改用 JHU CSSE）

数据源映射：
- 政策时间线  OxCGRT latest.csv
- 病例/死亡    JHU CSSE time_series_covid19_{confirmed,deaths}_US.csv（县级，死亡含年龄组）
- 行为         Google Mobility Region_Mobility_Report_CSVs.zip（2022-10 起停更，历史完整）
- 行为         YouGov covid-19-tracker（先拉 README 探结构）
"""
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

GH = "https://gh-proxy.com/https://raw.githubusercontent.com"

FILES = [
    # (输出文件, URL, 最小合法字节数)
    ("oxcgrt_policy_latest.csv", f"{GH}/OxCGRT/covid-policy-tracker/main/data/OxCGRT_latest.csv", 5_000_000),
    ("jhu_ts_confirmed_us.csv", f"{GH}/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv", 5_000_000),
    ("jhu_ts_deaths_us.csv", f"{GH}/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv", 5_000_000),
    ("yougov_readme.md", f"{GH}/YouGov-Data/covid-19-tracker/master/README.md", 2_000),
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-pilot/0.1"


def fetch(dest: str, url: str, min_size: int) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        return f"SKIP {os.path.basename(dest)} ({os.path.getsize(dest):,} B 已存在)"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    t0 = time.time()
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        n = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
    os.replace(tmp, dest)
    size = os.path.getsize(dest)
    assert size >= min_size, f"{dest} 下载不完整: {size} B"
    return f"OK   {os.path.basename(dest)} {size:,} B in {time.time()-t0:.1f}s"


def main():
    print(f"raw 目录: {RAW}")
    for name, url, min_size in FILES:
        try:
            print(fetch(os.path.join(RAW, name), url, min_size), flush=True)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
