# -*- coding: utf-8 -*-
"""M1 数据源探查：4 个源的可达性 + 元信息（CDC API / Google Mobility 存档 / OxCGRT / YouGov）"""
import sys
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

SOURCES = {
    "cdc_api_cases_deaths": "https://data.cdc.gov/resource/9mfq-cbt36.json?$limit=2",
    "mobility_zip":         "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip",
    "oxcgrt_policy":        "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/main/data/OxCGRT_latest.csv",
    "yougov_page":          "https://yougov.com/covid/international-research",
}


def probe(name: str, url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-pilot/0.1)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            code = resp.status
            ctype = resp.headers.get("Content-Type", "?")
            clen = resp.headers.get("Content-Length", "?")
            head = resp.read(300) if code == 200 else b""
            return code, ctype, clen, head
    except urllib.error.HTTPError as e:
        return e.code, "HTTPError", "?", b""
    except Exception as e:
        return -1, type(e).__name__ + ": " + str(e)[:150], "?", b""


def main():
    print("=== 数据源探查 ===")
    for name, url in SOURCES.items():
        code, ctype, clen, head = probe(name, url)
        print(f"[{name}] HTTP {code} | type={ctype} | len={clen}")
        if code == 200 and head:
            print("   head:", head.decode("utf-8", "ignore")[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
