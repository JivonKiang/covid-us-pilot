# -*- coding: utf-8 -*-
"""找 Mendeley article id + 试 API 变体拿文件列表"""
import re, json, urllib.request

html = open(r"e:\ai agent\research\covid-us-pilot\data\raw\_mend_page.html",
            encoding="utf-8", errors="ignore").read()

# INITIAL_STATE 全文里搜 article / file 相关
m = re.search(r"INITIAL_STATE = (\{.*?\});?\s*</script>", html, re.S)
if m:
    try:
        st = json.loads(m.group(1))
        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    if any(t in k.lower() for t in ["article", "file", "uuid", "download"]):
                        print(f"{path}.{k} = {str(v)[:150]}")
                    walk(v, path + "." + k)
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
        walk(st)
    except Exception as e:
        print("json err:", e)
        # 找数字 id 兜底
        for pat in [r'"article_id"\s*:\s*(\d+)', r'"id"\s*:\s*(\d{6,9})', r'articles/(\d{6,9})']:
            f = re.findall(pat, m.group(1))
            print(pat, "->", f[:5])
else:
    print("no INITIAL_STATE json")

# 直接试 API 变体
for u in [
    "https://data.mendeley.com/api/datasets/wdnynv3tns/versions/2/files",
    "https://data.mendeley.com/api/datasets/wdnynv3tns/2/files",
]:
    try:
        req = urllib.request.Request(u, headers={"Accept": "application/json",
                                                 "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read().decode("utf-8", "ignore")[:500]
        print("\nOK", u, "\n  ->", body)
    except Exception as e:
        print("\nFAIL", u, "->", str(e)[:120])
