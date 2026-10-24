# -*- coding: utf-8 -*-
"""YouGov data/ 目录全量列表（GET GitHub API via gh-proxy，读全量不截断）"""
import sys
import json
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

API = "https://gh-proxy.com/https://api.github.com/repos/YouGov-Data/covid-19-tracker/contents/data"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-pilot/0.1"


def main():
    req = urllib.request.Request(API, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            items = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.reason}")
        return
    files = [it for it in items if it.get("type") == "file"]
    print(f"data/ 目录共 {len(files)} 个文件:")
    for it in files:
        print(f"   {it['name']:<40} {it['size']/1e6:8.1f} MB")
    us = [it for it in files if "united" in it["name"].lower() or it["name"].lower().startswith(("us", "usa"))]
    if us:
        print(">>> 美国候选:", [it["name"] for it in us])


if __name__ == "__main__":
    main()
