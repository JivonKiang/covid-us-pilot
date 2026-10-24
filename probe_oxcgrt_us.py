# -*- coding: utf-8 -*-
"""列出 OxCGRT 仓库 data/United States/ 目录内容，确认州级政策文件。"""
import json
import urllib.request

API = "https://gh-proxy.com/https://api.github.com/repos/OxCGRT/covid-policy-tracker/contents/data/United%20States"
req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    items = json.loads(r.read().decode("utf-8"))

print(f"data/United States/ 共 {len(items)} 项:")
for it in items:
    size = it.get("size", 0)
    if it["type"] == "dir":
        print(f"      DIR   {it['name']}")
    else:
        print(f"  {size / 1e6:8.1f} MB  {it['name']}")
