# -*- coding: utf-8 -*-
"""确认 YouGov 仓库中美国数据文件的确切路径（HEAD 探测 + GitHub API 目录列表尝试）"""
import sys
import json
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

GH = "https://gh-proxy.com/https://raw.githubusercontent.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-pilot/0.1"

# GitHub API 目录列表（gh-proxy 对 api.github.com 的转发不保证可用，先试）
API = "https://gh-proxy.com/https://api.github.com/repos/YouGov-Data/covid-19-tracker/contents/data"

# 候选文件名（README 说 data/country-name.csv，分支已确认 master）
CAND = [
    "united%20states.csv",
    "United%20States.csv",
    "us.csv",
    "usa.csv",
    "united_states.csv",
]


def head(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Length", "?")
    except urllib.error.HTTPError as e:
        return e.code, "-"
    except Exception as e:
        return -1, type(e).__name__ + ":" + str(e)[:80]


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(4000)
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        return -1, (type(e).__name__ + ":" + str(e)[:80]).encode()


def main():
    code, body = get(API)
    if code == 200:
        try:
            names = [it["name"] for it in json.loads(body if isinstance(body, bytes) else body) if it.get("type") == "file"]
            print(f"[api_list] OK, {len(names)} 个文件:")
            for n in names:
                print("   ", n)
        except Exception as e:
            print(f"[api_list] 200 但解析失败: {e}; body 前 200: {body[:200]!r}")
    else:
        print(f"[api_list] HTTP {code} → 改用 HEAD 逐个试候选文件名")

    print("--- HEAD 探测 data/ 候选文件名 ---")
    for c in CAND:
        url = f"{GH}/YouGov-Data/covid-19-tracker/master/data/{c}"
        code, clen = head(url)
        print(f"   data/{c}  → HTTP {code}  size={clen}")


if __name__ == "__main__":
    main()
