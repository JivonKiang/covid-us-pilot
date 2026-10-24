# -*- coding: utf-8 -*-
"""从 Mendeley 页面 HTML 提取文件下载链接/UUID"""
import re, json

P = r"e:\ai agent\research\covid-us-pilot\data\raw\_mend_page.html"
html = open(P, encoding="utf-8", errors="ignore").read()
print("len:", len(html))

# 所有 UUID
uuids = re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", html)
print("uuids:", list(dict.fromkeys(uuids))[:20])

# 下载相关 URL 模式
for pat in ["public-files", "file_downloaded", "download", "prod-dcd"]:
    hits = re.findall(r"[\"'']([^\"']{0,120}" + pat + r"[^\"']{0,160})[\"'']", html)
    print(f"\n-- {pat}: {len(hits)}")
    for h in list(dict.fromkeys(hits))[:10]:
        print("  ", h[:200])

# window.__PRELOADED / INITIAL state JSON
for key in ["__PRELOADED_STATE__", "INITIAL_STATE", "window.__DATA__", "fileTree", "files"]:
    idx = html.find(key)
    if idx >= 0:
        print(f"\n== {key} at {idx}:")
        print(html[idx:idx+300].replace("\n", " ")[:300])
