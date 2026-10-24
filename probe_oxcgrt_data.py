# -*- coding: utf-8 -*-
"""OxCGRT 仓库 data/ 目录全量列表——找覆盖 2022 的完整政策数据文件"""
import sys
import json
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

API = "https://gh-proxy.com/https://api.github.com/repos/OxCGRT/covid-policy-tracker/contents/data"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-pilot/0.1"


def main():
    req = urllib.request.Request(API, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            items = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.reason}")
        return
    print("OxCGRT data/ 目录:")
    for it in items:
        size = it["size"] / 1e6 if it.get("type") == "file" else None
        label = f"{size:8.1f} MB" if size is not None else "   DIR "
        print(f"   {label}  {it['name']}")


if __name__ == "__main__":
    main()
