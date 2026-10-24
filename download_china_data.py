# -*- coding: utf-8 -*-
"""下载中国侧 M11 数据：JHU global 时序（含中国省级）+ OxCGRT 全量（含中国国家级）。

不依赖 GitHub API（gh-proxy 对 api.github.com 返回 403/404），直接构造 raw 路径
+ 镜像前缀，每个文件给候选 URL 逐个尝试。落盘后流式验证关键内容。
"""
import csv
import os
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

MIRRORS = ["https://gh-proxy.com/", "https://ghproxy.net/", ""]

FILES = {
    # 名称: (raw 候选路径, 最小字节数) —— 路径已用 GitHub 连接器核实
    "jhu_ts_confirmed_global.csv": (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
        "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
        1_500_000,
    ),
    "jhu_ts_deaths_global.csv": (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
        "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
        1_000_000,
    ),
    "oxcgrt_chn.csv": (
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/"
        "data/China/OxCGRT_CHN_latest.csv",
        8_000_000,
    ),
}


def fetch(name: str, raw_url: str, min_size: int) -> None:
    dest = os.path.join(RAW, name)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"[{name}] SKIP (已有 {os.path.getsize(dest)/1e6:.1f} MB)")
        return
    last_err = None
    for mir in MIRRORS:
        url = mir + raw_url
        part = dest + ".part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            size = os.path.getsize(part)
            if size < min_size:
                raise RuntimeError(f"文件过小 {size/1e6:.1f}MB < {min_size/1e6:.0f}MB")
            os.replace(part, dest)
            print(f"[{name}] OK ({size/1e6:.1f} MB)  via {mir or 'direct'}")
            return
        except Exception as e:
            last_err = e
            print(f"[{name}] 失败 via {mir or 'direct'}: {e}")
            if os.path.exists(part):
                os.remove(part)
    raise RuntimeError(f"[{name}] 所有镜像均失败: {last_err}")


for name, (raw_url, min_size) in FILES.items():
    fetch(name, raw_url, min_size)

# ---- 验证 1：JHU 中国省级 ----
print("\n---- JHU 中国省级验证 ----")
for name in ("jhu_ts_confirmed_global.csv", "jhu_ts_deaths_global.csv"):
    path = os.path.join(RAW, name)
    cn_rows = {}
    date_lo, date_hi = None, None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        prov_i, country_i = header.index("Province/State"), header.index("Country/Region")
        dates = header[4:]
        for row in reader:
            if row[country_i] != "China":
                continue
            prov = row[prov_i]
            # 行内累计数序列（找非零范围）
            vals = [int(float(x)) if x else 0 for x in row[4:]]
            first_nz = next((i for i, v in enumerate(vals) if v > 0), None)
            last_nz = next((len(vals) - 1 - i for i, v in enumerate(reversed(vals)) if v > 0), None)
            cn_rows[prov] = (dates[first_nz] if first_nz is not None else "-",
                             dates[last_nz] if last_nz is not None else "-",
                             vals[-1] if vals else 0)
            date_lo = dates[0]
            date_hi = dates[-1]
    print(f"\n{name}: 全球日期范围 {date_lo} -> {date_hi}")
    print(f"中国省级/地区行数: {len(cn_rows)}")
    for prov in ("Hubei", "Guangdong", "Beijing"):
        if prov in cn_rows:
            lo, hi, last = cn_rows[prov]
            print(f"  {prov}: {lo} -> {hi}  累计确诊 {last:,}" if "confirmed" in name else f"  {prov}: {lo} -> {hi}  累计死亡 {last:,}")
        else:
            print(f"  {prov}: 未找到！")
    # 打印几个样例行名
    print(f"  行名样例: {list(cn_rows)[:5]}")

# ---- 验证 2：OxCGRT 中国 ----
print("\n---- OxCGRT 中国验证 ----")
path = os.path.join(RAW, "oxcgrt_chn.csv")
with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    cn_i = header.index("CountryName")
    jur_i = header.index("Jurisdiction") if "Jurisdiction" in header else None
    date_i = header.index("Date")
    n_total, n_cn = 0, 0
    lo, hi = None, None
    # C1-C8 主政策列非空计数
    p_idx = {c: header.index(c) for c in header
             if len(c) >= 2 and c[:2] in {f"C{i}" for i in range(1, 9)} and "_" not in c}
    p_nonempty = {c: 0 for c in p_idx}
    for row in reader:
        n_total += 1
        if row[cn_i] != "China":
            continue
        n_cn += 1
        d = row[date_i]
        lo = d if lo is None or d < lo else lo
        hi = d if hi is None or d > hi else hi
        for c, i in p_idx.items():
            if row[i]:
                p_nonempty[c] += 1
print(f"总行数 {n_total:,}，中国行数 {n_cn:,}，中国日期范围 {lo} -> {hi}")
for c in p_idx:
    print(f"  {c}: 中国非空 {p_nonempty[c]}/{n_cn} ({100*p_nonempty[c]/max(n_cn,1):.0f}%)")
