# -*- coding: utf-8 -*-
"""M1 数据源探查 v2：CDC 加规范头 + 备选数据集；OxCGRT 走镜像；YouGov 候选路径"""
import sys
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

CANDIDATES = [
    ("cdc_csv_hdr", "https://data.cdc.gov/resource/9mfq-cbt36.csv?$limit=2"),
    ("cdc_weekly_deaths", "https://data.cdc.gov/resource/r8kw-7aab.json?$limit=2"),
    ("cdc_hospital_ts", "https://data.cdc.gov/resource/6yh2-fyrv.json?$limit=2"),
    ("oxcgrt_ghproxy", "https://gh-proxy.com/https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/main/data/OxCGRT_latest.csv"),
    ("oxcgrt_mirror_gh", "https://mirror.ghproxy.com/https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/main/data/OxCGRT_latest.csv"),
    ("oxcgrt_jsdelivr", "https://cdn.jsdelivr.net/gh/OxCGRT/covid-policy-tracker@main/data/OxCGRT_latest.csv"),
    ("github_main_reach", "https://github.com/OxCGRT/covid-policy-tracker"),
    ("yougov_covid", "https://yougov.com/covid"),
    ("yougov_docs_co19", "https://docs.yougov.com/co19/"),
    ("yougov_datapage", "https://docs.yougov.com/co19/YouGov_COVID19_datapage.html"),
]


def probe(name: str, url: str):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/json,text/csv,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            head = resp.read(300)
            return resp.status, resp.headers.get("Content-Type", "?"), head
    except urllib.error.HTTPError as e:
        return e.code, "HTTPError", b""
    except Exception as e:
        return -1, type(e).__name__ + ": " + str(e)[:120], b""


def main():
    print("=== 数据源探查 v2 ===")
    for name, url in CANDIDATES:
        code, ctype, head = probe(name, url)
        print(f"[{name}] HTTP {code} | {ctype}")
        if code == 200 and head:
            print("   head:", head.decode("utf-8", "ignore")[:160].replace("\n", " "))


if __name__ == "__main__":
    main()
