# -*- coding: utf-8 -*-
"""M7: LLM 行为层 —— deepseek-v4-pro 经 arkcli +chat 逐保留周预测 6 项行为频率.

4 配置 (full / no_inertia / no_policy / no_burden) x 12 保留周 = 48 次调用.
提示词仅含截至 t-1 的 8 周窗口; 温度 0; thinking disabled; json_object 输出.
增量写 llm_raw_responses.jsonl (可 resume); 最终输出 llm_pred_long.csv.
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from panel_prep import BEHAV6, N_TEST, ROOT, SHORT, load_df

MODEL = "deepseek-v4-pro"
CONFIGS = ["full", "no_inertia", "no_policy", "no_burden"]
WINDOW = 8
JSONL = ROOT + r"\data\panel\llm_raw_responses.jsonl"
OUT = ROOT + r"\data\panel\llm_pred_long.csv"

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
    "six protective behaviors of the US general public during the COVID-19 pandemic. "
    "Scale for each behavior: 0 = not at all, 1 = rarely, 2 = sometimes, 3 = often, 4 = always. "
    "Respond with a JSON object only, exactly six keys (mask, stay_home, avoid_restaurants, "
    "avoid_crowds, avoid_shops, avoid_work), numeric values rounded to one decimal in [0,4]."
)


def build_prompt(df, t_idx, config):
    win = df.iloc[max(0, t_idx - WINDOW):t_idx]
    lines = [f"Predict protective behaviors of the US public for the week starting "
             f"{df.loc[t_idx, 'week_start'].date()}."]
    lines.append("Each history row gives: date, stringency (policy stringency index 0-100), "
                 "log_burden (log10 of weekly new cases per 100k).")
    if config != "no_inertia":
        lines.append("Behavior history columns are frequencies on the same 0-4 scale as your output.")
    for _, r in win.iterrows():
        parts = [str(r["week_start"].date()),
                 f"stringency={r['stringency']:.1f}",
                 f"log_burden={r['log_burden']:.2f}"]
        if config != "no_inertia":
            for b in BEHAV6:
                v = r[b]
                if pd.notna(v):
                    parts.append(f"{SHORT[b]}={v:.2f}")
        lines.append(", ".join(parts))
    if config == "no_inertia":
        lines.append("Note: past behavior frequencies are withheld in this variant; infer plausible "
                     "next-week levels from the policy and epidemic-burden context alone.")
    lines.append("Behavior definitions: " + "; ".join(f"{k}: {v}" for k, v in BEHAV_DEF.items()))
    lines.append("Output a JSON object with keys mask, stay_home, avoid_restaurants, avoid_crowds, "
                 "avoid_shops, avoid_work; values one decimal in [0,4]. Output JSON only.")
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


def append_jsonl(rec):
    with _lock:
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_lock = threading.Lock()


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


def main():
    df, _ = load_df()
    n = len(df)
    test_idx = list(range(n - N_TEST, n))
    print(f"df rows={n}, test weeks={n - N_TEST}..{n - 1} "
          f"({df.loc[test_idx[0], 'week_start'].date()} .. {df.loc[test_idx[-1], 'week_start'].date()})")

    done = load_done()
    tasks = []
    for cfg in CONFIGS:
        for t in test_idx:
            wk = str(df.loc[t, "week_start"].date())
            if (cfg, wk) in done:
                continue
            tasks.append((cfg, t))
    print(f"resume: {len(done)} done, {len(tasks)} to run")

    # 预热: 第一个任务主线程同步跑, 防 SSO 并发刷新竞态
    if tasks:
        cfg0, t0 = tasks.pop(0)
        wk0 = str(df.loc[t0, "week_start"].date())
        print(f"warm-up call: {cfg0} / {wk0}")
        r = run_one(df, cfg0, t0)
        print(f"warm-up ok={r[2]}")

    results = [r] if tasks else []
    if tasks:
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

    # 汇总 -> llm_pred_long.csv
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
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {OUT}: {len(rows)} rows, {n_fail} failed calls (cumulative across runs)")


if __name__ == "__main__":
    main()
