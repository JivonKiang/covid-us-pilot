# -*- coding: utf-8 -*-
"""M8: H1 三方对比评估 + H1c 消融归因.

模型: persistence(参照) / base(岭回归无AR1) / base_ar1(强基线) / LLM(full) + 消融三配置.
口径: Delta 保留集 12 周 (df 后 12 行), 6 行为, 加权 RMSE (w=n_total), overall=6 指标均值.
H1: LLM vs base_ar1, overall 改善 >=15% 且稳定组(avoid_crowds/shops/work)组均值不输.
H1c: 消融 (去惯性/去政策/去负担) 中去惯性伤害最大 => 支持惯性主导.
输出: data/panel/h1_rmse.csv + data/panel/h1_eval.md
"""
import numpy as np
import pandas as pd

from panel_prep import BEHAV6, DRIFT, N_TEST, ROOT, STABLE, load_df

PANEL = ROOT + r"\data\panel"
LLM_CFG = ["full", "no_inertia", "no_policy", "no_burden"]
BASE_FILES = {
    "base": PANEL + r"\behavior_baseline_pred.csv",
    "base_ar1": PANEL + r"\behavior_baseline_ar1_pred.csv",
}

SHORT6 = {b: b.replace("yougov_", "").replace("_freq", "") for b in BEHAV6}


def load_preds(df):
    """返回 {model: DataFrame(index=week_start, columns=BEHAV6)} 只含保留集 12 周."""
    weeks = df["week_start"]
    out = {}

    # persistence: t-1 周观测
    pers = {}
    for b in BEHAV6:
        s = df[b]
        pers[b] = s.shift(1)
    out["persistence"] = pd.DataFrame(pers).set_index(df["week_start"])

    # 岭回归基线
    for name, path in BASE_FILES.items():
        bp = pd.read_csv(path, parse_dates=["week_start"])
        m = bp[weeks.isin(bp["week_start"])]
        bdf = df.merge(bp[["week_start"] + [b + "_pred" for b in BEHAV6]], on="week_start", how="left")
        out[name] = bdf.set_index("week_start")[[b + "_pred" for b in BEHAV6]].rename(
            columns={b + "_pred": b for b in BEHAV6})

    # LLM
    llm = pd.read_csv(PANEL + r"\llm_pred_long.csv", parse_dates=["week_start"])
    for cfg in LLM_CFG:
        sub = llm[llm["config"] == cfg].pivot(index="week_start", columns="behavior", values="pred")
        out[f"llm_{cfg}"] = sub.reindex(columns=BEHAV6)
    return out


def wrmse(obs, pred, w):
    m = obs.notna() & pred.notna() & w.notna()
    if m.sum() == 0:
        return np.nan
    o, p, ww = obs[m].to_numpy(), pred[m].to_numpy(), w[m].to_numpy()
    return float(np.sqrt(np.sum(ww * (o - p) ** 2) / np.sum(ww)))


def main():
    df, _ = load_df()
    test = df.iloc[len(df) - N_TEST:]
    obs_test = test.set_index("week_start")
    w_test = df["n_total"].set_axis(df["week_start"]).loc[obs_test.index]
    preds = load_preds(df)

    rows = []
    for model, pdf in preds.items():
        for b in BEHAV6:
            r = wrmse(obs_test[b], pdf.loc[obs_test.index, b], w_test)
            rows.append({"model": model, "behavior": SHORT6[b], "wrmse": r})
    rmse = pd.DataFrame(rows)
    rmse.to_csv(PANEL + r"\h1_rmse.csv", index=False)

    stable_s = [SHORT6[b] for b in STABLE]
    drift_s = [SHORT6[b] for b in DRIFT]

    # overall + 分组
    agg = []
    for model in preds:
        sub = rmse[rmse["model"] == model].set_index("behavior")["wrmse"]
        agg.append({
            "model": model,
            "overall": sub.mean(),
            "stable": sub[stable_s].mean(),
            "drift": sub[drift_s].mean(),
            "mask": sub["mask"], "stay_home": sub["stay_home"],
            "avoid_restaurants": sub["avoid_restaurants"],
            "avoid_crowds": sub["avoid_crowds"],
            "avoid_shops": sub["avoid_shops"],
            "avoid_work": sub["avoid_work"],
            "n_valid": int(rmse[(rmse["model"] == model)]["wrmse"].notna().sum()),
        })
    ag = pd.DataFrame(agg).set_index("model").round(4)
    ag.to_csv(PANEL + r"\h1_overall.csv")

    # H1 判定
    base_o = ag.loc["base_ar1", "overall"]
    llm_o = ag.loc["llm_full", "overall"]
    improve = (base_o - llm_o) / base_o
    stable_ok = ag.loc["llm_full", "stable"] <= ag.loc["base_ar1", "stable"]
    h1_pass = (improve >= 0.15) and stable_ok

    # H1c 消融: 相对 full 的 overall 恶化幅度
    abl = []
    for cfg in ["no_inertia", "no_policy", "no_burden"]:
        o = ag.loc[f"llm_{cfg}", "overall"]
        abl.append({"ablation": cfg, "overall": o,
                    "deterioration_pct": (o - llm_o) / llm_o * 100})
    abl = pd.DataFrame(abl).sort_values("deterioration_pct", ascending=False)
    worst = abl.iloc[0]["ablation"]
    h1c_pass = worst == "no_inertia" and abl.iloc[0]["deterioration_pct"] > 0

    # markdown 报告
    lines = []
    lines.append("# H1 评估: LLM 行为层 vs 规则基线 (Delta 保留集)\n")
    lines.append(f"- 评估窗口: {test['week_start'].iloc[0].date()} .. {test['week_start'].iloc[-1].date()} "
                 f"(12 周, 2021 Delta 波)\n- 指标: 样本量加权 RMSE (w=n_total), overall=6 行为均值\n")
    lines.append("## 总表\n")
    lines.append("| model | overall | stable组 | drift组 | mask | stay_home | avoid_rest | avoid_crowds | avoid_shops | avoid_work |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    order = ["persistence", "base", "base_ar1", "llm_full", "llm_no_inertia",
             "llm_no_policy", "llm_no_burden"]
    for m in order:
        r = ag.loc[m]
        lines.append(f"| {m} | {r['overall']:.4f} | {r['stable']:.4f} | {r['drift']:.4f} "
                     f"| {r['mask']:.4f} | {r['stay_home']:.4f} | {r['avoid_restaurants']:.4f} "
                     f"| {r['avoid_crowds']:.4f} | {r['avoid_shops']:.4f} | {r['avoid_work']:.4f} |")
    lines.append("")
    lines.append("## H1 主判定\n")
    lines.append(f"- base_ar1 overall = {base_o:.4f}, llm_full overall = {llm_o:.4f}, "
                 f"改善 = {improve * 100:.1f}% (阈值 >=15%)")
    lines.append(f"- 稳定组: llm_full {ag.loc['llm_full','stable']:.4f} vs base_ar1 "
                 f"{ag.loc['base_ar1','stable']:.4f} -> {'不输' if stable_ok else '输'}")
    if h1_pass:
        lines.append(f"- **判定: H1 支持** (LLM 行为层整体改善 {improve * 100:.1f}% >= 15% 且稳定组不输)")
    elif improve >= 0.15:
        lines.append("- **判定: H1 否证** (整体达标但稳定组输 -> 改善依赖漂移组)")
    else:
        lines.append(f"- **判定: H1 否证** (整体改善 {improve * 100:.1f}% < 15%)")
    lines.append("")
    lines.append("## H1c 消融归因\n")
    lines.append("| ablation | overall | 相对 full 恶化 % |")
    lines.append("|---|---|---|")
    for _, r in abl.iterrows():
        lines.append(f"| {r['ablation']} | {r['overall']:.4f} | {r['deterioration_pct']:+.1f}% |")
    lines.append("")
    if h1c_pass:
        lines.append("- **判定: H1c 支持** (去惯性伤害最大 -> LLM 预测主要由行为惯性驱动, 消融非平凡)")
    else:
        lines.append(f"- **判定: H1c 否证** (伤害最大的是 {worst}, 非去惯性 -> LLM 依赖的信息结构与预期不符)")
    lines.append("")
    lines.append("## 参照系读数\n")
    lines.append(f"- persistence overall = {ag.loc['persistence','overall']:.4f}: "
                 f"行为惯性很强 (YouGov 行为周际自相关高), 所有模型都应先跟它比")
    lines.append(f"- LLM vs persistence: "
                 f"{(ag.loc['persistence','overall'] - llm_o) / ag.loc['persistence','overall'] * 100:.1f}% "
                 f"{'优于' if llm_o < ag.loc['persistence','overall'] else '劣于'}")
    md = "\n".join(lines)
    with open(PANEL + r"\h1_eval.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print("\n--- ag table ---")
    print(ag.to_string())


if __name__ == "__main__":
    main()
