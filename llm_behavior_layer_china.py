# -*- coding: utf-8 -*-
"""M11 阶段C: 中国侧 LLM 行为层 — deepseek-v4-pro 经 arkcli, 4 配置 x 12 保留周.

口径 (预登记 H-CN4, 2026-10-22):
- df: 49 周骨架, 行为列 = 锚周(n>=30)真值 + 其余插值扩展 (与 SEIR 版 b_filled 同源)
- 保留段 = 最后 12 日历周 (2020-09-28 起), RMSE 只在真值锚周上算, n_total 加权
- 4 配置: full / no_inertia(去行为历史) / no_policy(去stringency) / no_burden(去log_burden)
- 注意与美国侧差异: 行为历史含插值复制值 (中国锚周稀疏, 如实标注)
- 增量写 llm_raw_responses_china.jsonl (可 resume); 评估内嵌 -> h1_eval_china.md
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
MODEL = "deepseek-v4-pro"
CONFIGS = ["full", "no_inertia", "no_policy", "no_burden"]
WINDOW = 8
N_TEST = 12
MIN_N = 30
LAMBDA = 1.0
JSONL = PANEL + r"\llm_raw_responses_china.jsonl"
OUT_PRED = PANEL + r"\llm_pred_long_china.csv"
OUT_EVAL = PANEL + r"\h1_eval_china.md"

BEHAV6 = ["yougov_mask_freq", "yougov_stay_home_freq", "yougov_avoid_restaurants_freq",
          "yougov_avoid_crowds_freq", "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]
SHORT = {"yougov_mask_freq": "mask", "yougov_stay_home_freq": "stay_home",
         "yougov_avoid_restaurants_freq": "avoid_restaurants",
         "yougov_avoid_crowds_freq": "avoid_crowds",
         "yougov_avoid_shops_freq": "avoid_shops", "yougov_avoid_work_freq": "avoid_work"}
STABLE = ["yougov_avoid_crowds_freq", "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]
DRIFT = ["yougov_mask_freq", "yougov_stay_home_freq", "yougov_avoid_restaurants_freq"]

BEHAV_DEF = {
    "mask": "wearing a face mask outside the home",
    "stay_home": "staying at home except for essential trips",
    "avoid_restaurants": "avoiding eating at restaurants or bars",
    "avoid_crowds": "avoiding crowded places",
    "avoid_shops": "avoiding shops other than for groceries or medicine",
    "avoid_work": "avoiding going to work outside the home (non-remote workers)",
}

SYS_INSTR = (
    "You are a behavioral epidemiology simulator forecasting next-week frequencies of "
    "six protective behaviors of the general public in mainland China during the COVID-19 "
    "pandemic. Scale for each behavior: 0 = not at all, 1 = rarely, 2 = sometimes, "
    "3 = often, 4 = always. Respond with a JSON object only, exactly six keys (mask, "
    "stay_home, avoid_restaurants, avoid_crowds, avoid_shops, avoid_work), numeric values "
    "rounded to one decimal in [0,4]."
)


def build_df():
    """49 周骨架: 锚周真值 + 插值扩展; drv = 人口加权 stringency + log_burden."""
    nat = pd.read_csv(PANEL + r"\cn_national_week.csv", parse_dates=["week_start"])
    weeks_all = pd.date_range("2020-01-06", "2020-12-07", freq="7D")
    nat = nat.set_index("week_start").reindex(weeks_all)
    nat["n_total"] = nat["n_total"].fillna(0.0)
    for b in BEHAV6:
        nat.loc[nat["n_total"] < MIN_N, b] = np.nan
        nat[b] = nat[b].ffill().bfill()
    prov = pd.read_csv(PANEL + r"\cn_prov_week.csv", parse_dates=["week_start"])
    prov["new_cases"] = prov["new_cases"].fillna(0.0)
    g = prov.groupby("week_start")
    drv = pd.DataFrame({
        "cases": g["new_cases"].sum(),
        "pop": g["population"].sum(),
        "stringency": g.apply(
            lambda x: np.average(x["pol_stringency"], weights=x["population"]),
            include_groups=False),
    }).reset_index()
    drv["per100k"] = drv["cases"] / drv["pop"] * 1e5
    drv["log_burden"] = np.log10(drv["per100k"].clip(lower=0.01))
    df = (nat[["n_total"] + BEHAV6].reset_index()
          .rename(columns={"index": "week_start"})
          .merge(drv[["week_start", "stringency", "log_burden"]], on="week_start", how="left"))
    df["t"] = np.arange(len(df))
    n_obs = int((nat["n_total"] >= MIN_N).sum())
    return df, n_obs


def build_prompt(df, t_idx, config):
    win = df.iloc[max(0, t_idx - WINDOW):t_idx]
    lines = [f"Predict protective behaviors of the general public in mainland China for the "
             f"week starting {df.loc[t_idx, 'week_start'].date()}."]
    hdr = ["date"]
    if config != "no_policy":
        hdr.append("stringency (policy stringency index 0-100)")
    if config != "no_burden":
        hdr.append("log_burden (log10 of weekly new cases per 100k)")
    if config != "no_inertia":
        hdr.append("behavior frequencies on the same 0-4 scale as your output")
    lines.append("Each history row gives: " + ", ".join(hdr) + ".")
    for _, r in win.iterrows():
        parts = [str(r["week_start"].date())]
        if config != "no_policy":
            parts.append(f"stringency={r['stringency']:.1f}")
        if config != "no_burden":
            parts.append(f"log_burden={r['log_burden']:.2f}")
        if config != "no_inertia":
            for b in BEHAV6:
                parts.append(f"{SHORT[b]}={r[b]:.2f}")
        lines.append(", ".join(parts))
    if config == "no_inertia":
        lines.append("Note: past behavior frequencies are withheld in this variant; infer "
                     "plausible next-week levels from the policy and epidemic-burden context alone.")
    lines.append("Behavior definitions: " + "; ".join(f"{k}: {v}" for k, v in BEHAV_DEF.items()))
    lines.append("Output a JSON object with keys mask, stay_home, avoid_restaurants, "
                 "avoid_crowds, avoid_shops, avoid_work; values one decimal in [0,4]. "
                 "Output JSON only.")
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
    for b in BEHAV6:
        v = float(d[SHORT[b]])
        out[b] = min(4.0, max(0.0, v))
    return out


_lock = threading.Lock()


def append_jsonl(rec):
    with _lock:
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_one(df, config, t_idx):
    week = str(df.loc[t_idx, "week_start"].date())
    prompt = build_prompt(df, t_idx, config)
    last_err = None
    for attempt in range(4):
        try:
            content, usage, el = call_arkcli(prompt)
            preds = parse_pred(content)
            append_jsonl({"config": config, "week_start": week, "ok": True,
                          "preds": preds, "usage": usage, "elapsed": round(el, 1),
                          "attempt": attempt + 1, "model": MODEL})
            return (config, t_idx, True, preds)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(3 * (2 ** attempt))
    append_jsonl({"config": config, "week_start": week, "ok": False,
                  "error": last_err[:500], "model": MODEL})
    return (config, t_idx, False, last_err)


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
                    done.add((r["config"], r["week_start"]))
    except FileNotFoundError:
        pass
    return done


def run_llm(df):
    n = len(df)
    test_idx = list(range(n - N_TEST, n))
    done = load_done()
    tasks = []
    for cfg in CONFIGS:
        for t in test_idx:
            wk = str(df.loc[t, "week_start"].date())
            if (cfg, wk) in done:
                continue
            tasks.append((cfg, t))
    print(f"resume: {len(done)} done, {len(tasks)} to run")
    if tasks:
        cfg0, t0 = tasks.pop(0)
        print(f"warm-up call: {cfg0} / {str(df.loc[t0, 'week_start'].date())}")
        r = run_one(df, cfg0, t0)
        print(f"warm-up ok={r[2]}")
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(run_one, df, c, t): (c, t) for c, t in tasks}
            n_ok = 1 if r[2] else 0
            n_done = 1
            for fut in as_completed(futs):
                res = fut.result()
                n_done += 1
                n_ok += 1 if res[2] else 0
                if n_done % 10 == 0 or n_done == len(tasks) + 1:
                    print(f"progress {n_ok}/{n_done} ok")
    # 汇总
    rows, n_fail = [], 0
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                n_fail += 1
                continue
            for b in BEHAV6:
                rows.append({"config": rec["config"], "week_start": rec["week_start"],
                             "behavior": b, "pred": rec["preds"][b]})
    pd.DataFrame(rows).to_csv(OUT_PRED, index=False)
    print(f"saved {OUT_PRED}: {len(rows)} rows, {n_fail} failed calls (cumulative)")


def base_ar1_pred(df, n_train):
    """逐行为岭回归 (美国 behavior_baseline base_ar1 口径): 特征=政策lag1/负担lag1/t/自身lag1."""
    out = {}
    for b in BEHAV6:
        b_arr = df[b].to_numpy(float)
        X = np.column_stack([
            np.ones(len(df)),  # 截距列 (对齐美国版: A[0,0] -= λ 使截距不正则)
            df["stringency"].shift(1).bfill().to_numpy(),
            df["log_burden"].shift(1).bfill().to_numpy(),
            df["t"].to_numpy(float),
            pd.Series(b_arr).shift(1).bfill().to_numpy()])
        y = b_arr
        Xtr = X[:n_train]
        mu, sd = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0)
        sd[sd == 0] = 1.0
        Xs = np.column_stack([np.ones(len(df)), (X[:, 1:] - mu) / sd])  # 截距列不标准化
        A = Xs[:n_train].T @ Xs[:n_train] + LAMBDA * np.eye(X.shape[1])
        A[0, 0] -= LAMBDA
        beta = np.linalg.solve(A, Xs[:n_train].T @ y[:n_train])
        out[b] = Xs @ beta
    return pd.DataFrame(out, index=df["week_start"], columns=BEHAV6)


def wrmse(obs, pred, w):
    m = obs.notna() & pred.notna() & w.notna() & (w > 0)
    if m.sum() == 0:
        return np.nan
    o, p, ww = obs[m].to_numpy(), pred[m].to_numpy(), w[m].to_numpy()
    return float(np.sqrt(np.sum(ww * (o - p) ** 2) / np.sum(ww)))


def eval_h1(df, n_obs):
    n = len(df)
    test_idx = list(range(n - N_TEST, n))
    test = df.iloc[test_idx].set_index("week_start")
    anchors = test.index[test["n_total"] >= MIN_N]
    print(f"保留段 {test.index[0].date()} .. {test.index[-1].date()}, 真值锚周 {len(anchors)} 个: "
          f"{[str(a.date()) for a in anchors]}")
    w = test["n_total"]
    preds = {"persistence": df[BEHAV6].shift(1).iloc[test_idx].set_index(df.loc[test_idx, "week_start"]),
             "base_ar1": base_ar1_pred(df, n - N_TEST).loc[anchors]}
    llm = pd.read_csv(OUT_PRED, parse_dates=["week_start"])
    for cfg in CONFIGS:
        sub = (llm[llm["config"] == cfg]
               .pivot(index="week_start", columns="behavior", values="pred")
               .reindex(columns=BEHAV6))
        preds[f"llm_{cfg}"] = sub
    order = ["persistence", "base_ar1"] + [f"llm_{c}" for c in CONFIGS]
    short6 = [SHORT[b] for b in BEHAV6]
    stable_s = [SHORT[b] for b in STABLE]
    drift_s = [SHORT[b] for b in DRIFT]
    rows = []
    for model in order:
        pdf = preds[model].loc[anchors]
        for b in BEHAV6:
            rows.append({"model": model, "behavior": SHORT[b], "wrmse": wrmse(test[b], pdf[b], w)})
    rmse = pd.DataFrame(rows)
    ag = (rmse.pivot(index="model", columns="behavior", values="wrmse")
          .assign(overall=lambda x: x[short6].mean(1),
                  stable=lambda x: x[stable_s].mean(1),
                  drift=lambda x: x[drift_s].mean(1))
          .round(4))
    base_o = ag.loc["base_ar1", "overall"]
    llm_o = ag.loc["llm_full", "overall"]
    pers_o = ag.loc["persistence", "overall"]
    improve = (base_o - llm_o) / base_o
    stable_ok = ag.loc["llm_full", "stable"] <= ag.loc["base_ar1", "stable"]
    h1_pass = (improve >= 0.15) and stable_ok
    # 消融
    abl = []
    for cfg in ["no_inertia", "no_policy", "no_burden"]:
        o = ag.loc[f"llm_{cfg}", "overall"]
        abl.append({"ablation": cfg, "overall": o,
                    "deterioration_pct": (o - llm_o) / llm_o * 100})
    abl = pd.DataFrame(abl).sort_values("deterioration_pct", ascending=False)
    worst = abl.iloc[0]["ablation"]
    # 机制 2: full vs no_burden 配置不敏感
    d_burden = abs(ag.loc["llm_no_burden", "overall"] - llm_o) / llm_o * 100
    # 机制 3: no_inertia 是否最大伤害
    inert_max = worst == "no_inertia"
    lines = ["# H1-CN 评估: 中国侧 LLM 行为层 (2020 天花板窗口, 预登记 H-CN4)\n"]
    lines.append(f"- 保留段: {test.index[0].date()} .. {test.index[-1].date()} (12 周), "
                 f"真值锚周 {len(anchors)} 个, n_total 加权 RMSE, overall=6 行为均值")
    lines.append(f"- 行为历史含插值复制值 (锚周 {n_obs}/49, 中国 YouGov 波次稀疏) — 与美国侧差异, 如实标注\n")
    lines.append("## 总表\n")
    lines.append("| model | overall | stable组 | drift组 | mask | stay_home | avoid_rest | "
                 "avoid_crowds | avoid_shops | avoid_work |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in order:
        r = ag.loc[m]
        lines.append(f"| {m} | {r['overall']:.4f} | {r['stable']:.4f} | {r['drift']:.4f} "
                     f"| {r['mask']:.4f} | {r['stay_home']:.4f} | {r['avoid_restaurants']:.4f} "
                     f"| {r['avoid_crowds']:.4f} | {r['avoid_shops']:.4f} | {r['avoid_work']:.4f} |")
    lines.append("")
    lines.append("## H1-CN 主判定\n")
    lines.append(f"- base_ar1 overall = {base_o:.4f}, llm_full overall = {llm_o:.4f}, "
                 f"改善 = {improve * 100:.1f}% (阈值 >=15%)")
    lines.append(f"- persistence overall = {pers_o:.4f} (参照系: 恒定行为窗口的天然下界)")
    lines.append(f"- LLM vs persistence: {(pers_o - llm_o) / pers_o * 100:+.1f}%")
    if h1_pass:
        lines.append("- **判定: H1-CN 支持 (意外发现)** -> 按预登记单独登记候选假设")
    else:
        lines.append(f"- **判定: H1-CN 支持「预期阴性」** (改善 {improve * 100:.1f}% < 15%, "
                     "与预登记一致: 天花板窗口无信息量, 对 LLM 同样成立)")
    lines.append("")
    lines.append("## 预登记机制检验\n")
    lines.append("| 机制 | 预登记条件 | 读数 | 判定 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 1 天花板方差小 | LLM 难击败 persistence | vs persistence "
                 f"{(pers_o - llm_o) / pers_o * 100:+.1f}% | "
                 f"{'支持' if llm_o >= pers_o else '否证 (LLM 优于 persistence)'} |")
    lines.append(f"| 2 LLM 先验画像现成 | full 与 no_burden overall 差 <3% | 差 {d_burden:.1f}% | "
                 f"{'支持 (配置不敏感)' if d_burden < 3 else '否证 (burden 有边际价值)'} |")
    lines.append(f"| 3 惯性通道质量低 | no_inertia 非最大伤害 | 最大伤害 = {worst} "
                 f"({abl.iloc[0]['deterioration_pct']:+.1f}%) | "
                 f"{'支持' if not inert_max else '否证 (惯性通道有效)'} |")
    lines.append("")
    lines.append("## 消融 (相对 full 恶化 %)\n")
    lines.append("| ablation | overall | 相对 full 恶化 % |")
    lines.append("|---|---|---|")
    for _, r in abl.iterrows():
        lines.append(f"| {r['ablation']} | {r['overall']:.4f} | {r['deterioration_pct']:+.1f}% |")
    md = "\n".join(lines)
    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        f.write(md)
    rmse.to_csv(PANEL + r"\h1_rmse_china.csv", index=False)
    print("\n" + md)
    print("\n--- ag table ---")
    print(ag.to_string())


def main():
    df, n_obs = build_df()
    print(f"df rows={len(df)}, 锚周 n_obs={n_obs}")
    run_llm(df)
    eval_h1(df, n_obs)


if __name__ == "__main__":
    main()
