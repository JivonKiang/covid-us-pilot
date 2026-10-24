# -*- coding: utf-8 -*-
"""探查 JHU / OxCGRT 仓库目录结构，确认中国侧数据文件的确切路径。"""
import json
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def list_dir(repo: str, path: str) -> list:
    url = f"https://gh-proxy.com/https://api.github.com/repos/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


print("=== OxCGRT data/ 根目录 ===")
try:
    for it in list_dir("OxCGRT/covid-policy-tracker", "data"):
        size = it.get("size", 0)
        print(f"  [{it['type']}] {it['name']}  {size/1e6:.1f}MB" if it["type"] == "file" else f"  [{it['type']}] {it['name']}")
except Exception as e:
    print(f"  失败: {e}")

print("\n=== JHU csse_covid_19_time_series/ 目录（只列 global 文件） ===")
try:
    for it in list_dir("CSSEGISandData/COVID-19", "csse_covid_19_time_series"):
        if "global" in it["name"].lower():
            print(f"  [{it['type']}] {it['name']}  {it.get('size', 0)/1e6:.1f}MB")
except Exception as e:
    print(f"  失败: {e}")
