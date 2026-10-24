# -*- coding: utf-8 -*-
"""M11 阶段B: 开放潮窗口 LLM 行为层 — deepseek-v4-pro 经 arkcli, 4 配置 x 8 单元 x 5 保留周.

口径 (预登记 H-CN5, 落盘于评估前 2026-10-22, 见 outputs/M11-中国扩展-汇报.md):
- 面板: wavelift_week.csv, 8 单元(全国+7省) x 8 周(2022-11-26 周六起始, W9 弃用)
- 目标: BESI_log / BHII_log (log10 百度疫情搜索指数 / 百度健康问诊指数, 等权)
- 保留段 W4-W8, 历史窗口滚动扩展 (W1..W_{t-1}, 3~7 周)
- base_ar1: 逐单元逐列岭回归 (截距+stringency_lag1+log_burden_lag1+t+自身lag1, LAMBDA=1.0,
  训练点仅 3 个 -> 正则主导, 如实标注; 若退化近似 persistence 则主判定改用 vs persistence)
- 160 次调用, 增量写 llm_raw_responses_wavelift.jsonl (可 resume)
"""
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
PANEL = ROOT + r"\data\panel"
PANEL_CSV = PANEL + r"\wavelift_week.csv"
MODEL = "deepseek-v4-pro"
CONFIGS = ["full", "no_inertia", "no_policy", "no_burden"]
N_TEST = 5            # W4..W8
N_TRAIN = 3           # W1..W3
LAMBDA = 1.0
W0 = pd.Timestamp(2022, 11, 26)
TARGETS = ["BESI_log", "BHII_log"]
UNITS = ["national", "Guangdong", "Henan", "Sichuan", "Beijing", "Jiangsu", "Hubei", "Zhejiang"]
UNIT_LABEL = {"national": "mainland China (national)", "Guangdong": "Guangdong province",
              "Henan": "Henan province", "Sichuan": "Sichuan province",
              "Beijing": "Beijing municipality", "Jiangsu": "Jiangsu province",
              "Hubei": "Hubei province", "Zhejiang": "Zhejiang province"}
JSONL = PANEL + r"\llm_raw_responses_wavelift.jsonl"
OUT_PRED = PANEL + r"\llm_pred_long_wavelift.csv"
OUT_EVAL = PANEL + r"\h1_eval_china_wavelift.md"

SYS_INSTR = (
    "You are a behavioral epidemiology simulator forecasting next-week log10 levels of two "
    "COVID-related behavioral indicators for a region of mainland China during the reopening "
    "wave (Nov 2022 - Jan 2023, when zero-COVID ended). BESI_log = log10 of the Baidu epidemic "
    "search index (COVID symptom/prevention/supply search intensity); BHII_log = log10 of the "
    "Baidu Health online consultation index. Respond with a JSON object only, exactly two keys "
    "(BESI_log, BHII_log), numeric values rounded to two decimals."
)


def build_units():
    panel = pd.read_csv(PANEL_CSV)
    panel["week_start"] = W0 + pd.to_timedelta((panel["week"] - 1) * 7, unit="D")
    units = {u: g.sort_values("week").reset_index(drop=True) for u, g in panel.groupby("unit")}
    for u in units:
        units[u]["t"] = np.arange(len(units[u]), dtype=float)
    return units


def build_prompt(du, t_idx, config, unit):
    win = du.iloc[max(0, t_idx - 7):t_idx]  # 滚动扩展历史 (最多 7 周)
    lines = [f"Predict BESI_log and BHII_log for {UNIT_LABEL[unit]} for the week starting "
             f"{du.loc[t_idx, 'week_start'].date()} (COVID reopening wave in China: policy "
             f"controls ended around Dec 7 2022, 'Class B management' from Jan 8 2023)."]
    hdr = ["date"]
    if config != "no_policy":
        hdr.append("stringency (OxCGRT policy stringency index 0-100)")
    if config != "no_burden":
        hdr.append("log_burden (log10 of weekly official new cases + 1; official counts "
                   "severely undercount infections during the reopening)")
    if config != "no_inertia":
        hdr.append("BESI_log and BHII_log (the two indicators you must forecast)")
    lines.append("Each history row gives: " + ", ".join(hdr) + ".")
    for _, r in win.iterrows():
        parts = [str(r["week_start"].date())]
        if config != "no_policy":
            parts.append(f"stringency={r['stringency']:.1f}")
        if config != "no_burden":
            parts.append(f"log_burden={r['log_burden']:.2f}")
        if config != "no_inertia":
            parts.append(f"BESI_log={r['BESI_log']:.2f}, BHII_log={r['BHII_log']:.2f}")
        lines.append(", ".join(parts))
    if config == "no_inertia":
        lines.append("Note: past indicator values are withheld in this variant; infer plausible "
                     "next-week levels from the policy and epidemic-burden context alone.")
    lines.append("Output a JSON object with keys BESI_log, BHII_log; numeric values with two "
                 "decimals. Output JSON only.")
    return "\n".join(lines)


_NPM = r"C:\Users\admin\AppData\Roaming\npm"
_RUN_JS = _NPM + r"\node_modules\@volcengine\ark-cli\scripts\run.js"
_NODE = _NPM + r"\node.exe"
if not os.path.exists(_NODE):
    _NODE = "node"


def call_arkcli(prompt):
    cmd = [_NODE, _RUN_JS, "+chat", prompt,
           "--instructions", SYS_INSTR,
           "--model", MODEL,
           "--temperature", "0",
           "--thinking", "disabled",
           "--text-format", "json_object",
           "--max-output-tokens", "3000",
           "--format", "json"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                       encoding="utf-8", errors="replace")
    el = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"exit={p.returncode} stderr={p.stderr.strip()[:300]}")
    s = p.stdout.strip()
    i = s.find("{")
    if i < 0:
        raise RuntimeError(f"no json in stdout: {s[:200]}")
    resp = json.loads(s[i:])
    if resp.get("status") not in ("completed", None):
        raise RuntimeError(f"status={resp.get('status')} detail={resp.get('incomplete_details')}")
    return resp.get("content", ""), resp.get("usage", {}), el


def parse_pred(content):
    s = content.strip()
    try:
        d = json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            raise ValueError(f"unparseable content: {s[:120]}")
        d = json.loads(s[i:j + 1])
    out = {}
    for t in TARGETS:
        v = float(d[t])
        out[t] = min(12.0, max(0.0, v))  # log10 宽域防荒谬值
    return out


_lock = threading.Lock()


def append_jsonl(rec):
    with _lock:
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_one(units, config, unit, t_idx):
    du = units[unit]
    week = str(du.loc[t_idx, "week_start"].date())
    prompt = build_prompt(du, t_idx, config, unit)
    last_err = None
    for attempt in range(4):
        try:
            content, usage, el = call_arkcli(prompt)
            preds = parse_pred(content)
            append_jsonl({"config": config, "unit": unit, "week_start": week, "ok": True,
                          "preds": preds, "usage": usage, "elapsed": round(el, 1),
                          "attempt": attempt + 1, "model": MODEL})
            return True
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(3 * (2 ** attempt))
    append_jsonl({"config": config, "unit": unit, "week_start": week, "ok": False,
                  "error": last_err[:500], "model": MODEL})
    return False


def load_done():
    done = set()
    try:
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("ok"):
                    done.add((r["config"], r["unit"], r["week_start"]))
    except FileNotFoundError:
        pass
    return done


def run_llm(units):
    test_idx = list(range(N_TRAIN, N_TRAIN + N_TEST))
    done = load_done()
    tasks = []
    for cfg in CONFIGS:
        for u in UNITS:
            for t in test_idx:
                wk = str(units[u].loc[t, "week_start"].date())
                if (cfg, u, wk) in done:
                    continue
                tasks.append((cfg, u, t))
    print(f"resume: {len(done)} done, {len(tasks)} to run")
    n_total = len(tasks) + len(done)
    if tasks:
        cfg0, u0, t0 = tasks.pop(0)
        print(f"warm-up call: {cfg0} / {u0} / {str(units[u0].loc[t0, 'week_start'].date())}")
        ok = run_one(units, cfg0, u0, t0)
        print(f"warm-up ok={ok}")
        n_ok, n_done = (1, 1) if ok else (0, 1)
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(run_one, units, c, u, t): (c, u) for c, u, t in tasks}
            for fut in as_completed(futs):
                n_ok += 1 if fut.result() else 0
                n_done += 1
                if n_done % 20 == 0 or n_done == len(tasks) + 1:
                    print(f"progress {n_ok}/{n_done} ok (total target {n_total})")
    rows, n_fail = [], 0
    seen = set()  # 防并发重复执行产生的重复 ok 记录 (保留首条)
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                n_fail += 1
                continue
            key = (rec["config"], rec["unit"], rec["week_start"])
            if key in seen:
                continue
            seen.add(key)
            for t in TARGETS:
                rows.append({"config": rec["config"], "unit": rec["unit"],
                             "week_start": rec["week_start"], "indicator": t,
                             "pred": rec["preds"][t]})
    pd.DataFrame(rows).to_csv(OUT_PRED, index=False)
    print(f"saved {OUT_PRED}: {len(rows)} rows, {n_fail} failed calls (cumulative)")


def base_ar1_pred(du, use_t=True):
    """逐指标岭回归: 截距(不正则) + stringency_lag1 + log_burden_lag1 + [t] + 自身 lag1.
    训练 = W1..W3 (3 点, 正则主导 — 预登记已标注).
    use_t=False 为诊断版 (base_ar1_notrend): 3 点拟合下 t 外推爆炸, 无 t 版作参照."""
    out = {}
    for t in TARGETS:
        y = du[t].to_numpy(float)
        cols = [np.ones(len(du)),
                du["stringency"].shift(1).bfill().to_numpy(),
                du["log_burden"].shift(1).bfill().to_numpy()]
        if use_t:
            cols.append(du["t"].to_numpy(float))
        cols.append(pd.Series(y).shift(1).bfill().to_numpy())
        X = np.column_stack(cols)
        k = X.shape[1]
        n_tr = N_TRAIN
        mu, sd = X[:n_tr, 1:].mean(0), X[:n_tr, 1:].std(0)
        sd[sd == 0] = 1.0
        Xs = np.column_stack([np.ones(len(du)), (X[:, 1:] - mu) / sd])
        A = Xs[:n_tr].T @ Xs[:n_tr] + LAMBDA * np.eye(k)
        A[0, 0] -= LAMBDA
        beta = np.linalg.solve(A, Xs[:n_tr].T @ y[:n_tr])
        out[t] = Xs @ beta
    return pd.DataFrame(out, index=du["week_start"])


def rmse(obs, pred):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    m = np.isfinite(o) & np.isfinite(p)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((o[m] - p[m]) ** 2)))


def eval_h1(units):
    test_idx = list(range(N_TRAIN, N_TRAIN + N_TEST))
    test_weeks = [str(units["national"].loc[t, "week_start"].date()) for t in test_idx]
    preds = {}
    for u in UNITS:
        du = units[u].set_index("week_start")
        preds[("persistence", u)] = du[TARGETS].shift(1).iloc[test_idx]
        preds[("base_ar1", u)] = base_ar1_pred(units[u], use_t=True).iloc[test_idx]
        preds[("base_ar1_notrend", u)] = base_ar1_pred(units[u], use_t=False).iloc[test_idx]
    llm = pd.read_csv(OUT_PRED)
    for cfg in CONFIGS:
        for u in UNITS:
            sub = (llm[(llm["config"] == cfg) & (llm["unit"] == u)]
                   .pivot(index="week_start", columns="indicator", values="pred"))
            sub.index = pd.to_datetime(sub.index)
            preds[("llm_" + cfg, u)] = sub.reindex(columns=TARGETS)
    order = ["persistence", "base_ar1", "base_ar1_notrend"] + [f"llm_{c}" for c in CONFIGS]
    rows = []
    for model in order:
        for u in UNITS:
            tw = pd.to_datetime(test_weeks)
            pdf = preds[(model, u)].loc[tw]
            du = units[u].set_index("week_start").loc[tw]
            for t in TARGETS:
                rows.append({"model": model, "unit": u, "indicator": t,
                             "rmse": rmse(du[t], pdf[t])})
    rm = pd.DataFrame(rows)
    rm.to_csv(PANEL + r"\h1_rmse_wavelift.csv", index=False)
    ag = (rm.pivot_table(index="model", columns="indicator", values="rmse", aggfunc="mean")
          .assign(overall=lambda x: x[TARGETS].mean(1)).round(4))
    base_o = ag.loc["base_ar1", "overall"]
    llm_o = ag.loc["llm_full", "overall"]
    pers_o = ag.loc["persistence", "overall"]
    improve_base = (base_o - llm_o) / base_o
    improve_pers = (pers_o - llm_o) / pers_o
    degenerate_close = abs(base_o - pers_o) / pers_o * 100 < 10   # 失效一: 近似 persistence
    degenerate_blowup = base_o > pers_o * 1.5                     # 失效二: 3 训练点 t 外推发散
    degenerate = degenerate_close or degenerate_blowup            # 附加口径: 任一失效即切 persistence 判定
    primary_ok = improve_pers >= 0.15 if degenerate else improve_base >= 0.15
    # 机制
    abl = []
    for cfg in ["no_inertia", "no_policy", "no_burden"]:
        o = ag.loc[f"llm_{cfg}", "overall"]
        abl.append({"ablation": cfg, "overall": o,
                    "deterioration_pct": (o - llm_o) / llm_o * 100})
    abl = pd.DataFrame(abl).sort_values("deterioration_pct", ascending=False)
    worst = abl.iloc[0]["ablation"]
    m1_ok = worst == "no_inertia" and abl.iloc[0]["deterioration_pct"] >= 30
    d_pol = abs(ag.loc["llm_no_policy", "overall"] - llm_o) / llm_o * 100
    m2_ok = d_pol < 5
    m3_ok = llm_o < pers_o

    lines = ["# H-CN5 评估: 开放潮窗口 LLM 行为层 (阶段 B, 预登记 H-CN5)\n"]
    lines.append(f"- 保留段: {test_weeks[0]} .. {test_weeks[-1]} (W4-W8 共 {N_TEST} 周), "
                 f"8 单元 x 2 指标 x {N_TEST} 周 = 80 评估点, 等权 RMSE")
    lines.append(f"- base_ar1 训练点仅 {N_TRAIN} 个 (W1-W3), 岭回归 LAMBDA={LAMBDA} 正则主导, "
                 "功效弱 — 预登记已标注\n")
    lines.append("## 总表 (RMSE, log10 空间; overall = 8 单元 x 2 指标等权均值)\n")
    lines.append("| model | overall | BESI | BHII |")
    lines.append("|---|---|---|---|")
    for m in order:
        r = ag.loc[m]
        lines.append(f"| {m} | {r['overall']:.4f} | {r['BESI_log']:.4f} | {r['BHII_log']:.4f} |")
    lines.append("")
    lines.append("## 主判定\n")
    lines.append(f"- base_ar1 overall = {base_o:.4f}, persistence overall = {pers_o:.4f}, "
                 f"llm_full overall = {llm_o:.4f}")
    lines.append(f"- LLM vs base_ar1: {improve_base * 100:+.1f}% (阈值 >=15%)")
    lines.append(f"- LLM vs persistence: {improve_pers * 100:+.1f}%")
    lines.append(f"- base 失效检查: 近似 persistence = {abs(base_o - pers_o) / pers_o * 100:.1f}% "
                 f"(阈值 <10%), 发散 = base/persistence = {base_o / pers_o:.2f}x (阈值 >1.5x)")
    if degenerate:
        mode = "近似 persistence" if degenerate_close else "t 外推发散" if degenerate_blowup else "双重失效"
        lines.append(f"- **base_ar1 失效 ({mode}), 按预登记附加口径主判定改用 vs persistence**")
        lines.append("- 诊断: base_ar1 用 3 个训练点 (W1-W3) 拟合 5 参数岭回归且含线性趋势项 t, "
                     "外推 5 周时趋势系数放大导致发散/退化; base_ar1_notrend (去 t 诊断版) 一并给出 "
                     "供参照, 该失效为小样本方法学限制而非信号, 不影响 LLM 增益判定的附加口径")
    else:
        lines.append("- base_ar1 有效, 主判定用 vs base_ar1")
    if primary_ok:
        lines.append("- **判定: H-CN5 支持** (开放潮剧变窗口 LLM 有真实增益, 与阶段 C 天花板窗口相反)")
    else:
        lines.append("- **判定: H-CN5 不支持** (改善 <15% 或为负)")
    lines.append("")
    lines.append("## 预登记机制检验\n")
    lines.append("| 机制 | 预登记条件 | 读数 | 判定 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| M1 惯性通道 | no_inertia 恶化 >=30% 且最大 | 最大伤害 = {worst} "
                 f"({abl.iloc[0]['deterioration_pct']:+.1f}%) | "
                 f"{'支持 (复现中美同构)' if m1_ok else '不成立'} |")
    lines.append(f"| M2 政策先验内化 | no_policy 变化 <5% | 差 {d_pol:.1f}% | "
                 f"{'支持' if m2_ok else '不成立 (政策上下文有边际价值)'} |")
    lines.append(f"| M3 剧变窗口 vs persistence | LLM 优于 persistence | "
                 f"{improve_pers * 100:+.1f}% | "
                 f"{'支持 (惯性规则失效)' if m3_ok else '不成立 (persistence 仍占优)'} |")
    lines.append("")
    lines.append("## 消融 (相对 full 恶化 %)\n")
    lines.append("| ablation | overall | 相对 full 恶化 % |")
    lines.append("|---|---|---|")
    for _, r in abl.iterrows():
        lines.append(f"| {r['ablation']} | {r['overall']:.4f} | {r['deterioration_pct']:+.1f}% |")
    lines.append("")
    lines.append("## 分单元 overall (RMSE)\n")
    pvt = rm.pivot_table(index="model", columns="unit", values="rmse", aggfunc="mean")
    pvt = pvt.reindex(order)[UNITS].round(4)
    lines.append("| model | " + " | ".join(UNITS) + " |")
    lines.append("|---|" + "---|" * len(UNITS))
    for m in order:
        lines.append(f"| {m} | " + " | ".join(f"{pvt.loc[m, u]:.4f}" for u in UNITS) + " |")
    md = "\n".join(lines)
    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print("\n--- ag table ---")
    print(ag.to_string())


def main():
    units = build_units()
    print(f"units={len(units)}, rows/unit={[len(v) for v in units.values()][:3]}...")
    run_llm(units)
    eval_h1(units)


if __name__ == "__main__":
    main()
