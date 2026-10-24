# -*- coding: utf-8 -*-
"""M9: T2 SEIR 轨道三配置 —— 行为反馈对疫情轨迹预测的增益检验.

日步长 Euler SEIR (sigma=1/3, gamma=1/5), beta(t)=beta0*max(0.05, 1-eps*b/4),
b=6 行为均值 (0-4). 三配置:
  no_feedback: b=0 全周期 (恒 beta0)
  rule_driven: 训练期 b=观测, 保留期 b=base_ar1 预测
  llm_driven:  训练期 b=观测, 保留期 b=llm_full 预测
训练期网格拟合 (beta0, eps, infl, lag) + kappa 解析最小二乘; 保留期外推
test RMSE + 峰值周误差. 输出 seir_fits.csv / seir_traj.csv / seir_eval.md.
否证条件: 保留期行为驱动不优于无反馈.
"""
import itertools

import numpy as np
import pandas as pd

from panel_prep import BEHAV6, N_TEST, ROOT, load_df

PANEL = ROOT + r"\data\panel"
SIGMA, GAMMA = 1 / 3, 1 / 5
BETA0_GRID = list(np.geomspace(0.1, 0.9, 16))
EPS_GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]
INFL_GRID = [2.0, 4.0]
LAG_GRID = [0, 1, 2]


def load_b_series(df):
    """返回 dict: 观测/规则预测/LLM预测 的周级行为均值序列 (长度=52)."""
    obs = df[BEHAV6].mean(axis=1)  # 0-4
    bp = pd.read_csv(PANEL + r"\behavior_baseline_ar1_pred.csv", parse_dates=["week_start"])
    bdf = df[["week_start"]].merge(bp[["week_start"] + [b + "_pred" for b in BEHAV6]],
                                   on="week_start", how="left")
    rule = bdf[[b + "_pred" for b in BEHAV6]].mean(axis=1)
    llm = pd.read_csv(PANEL + r"\llm_pred_long.csv", parse_dates=["week_start"])
    lf = llm[llm["config"] == "full"].pivot(index="week_start", columns="behavior",
                                            values="pred").reindex(df["week_start"])
    llm_b = lf.mean(axis=1)
    return {"obs": obs.to_numpy(), "rule": rule.to_numpy(), "llm": llm_b.to_numpy()}


def simulate(b_week, beta0, eps, infl, w0, n_pop, n_weeks):
    """日步长 Euler SEIR, 返回周真实新感染数组 (长度=n_weeks)."""
    n_days = n_weeks * 7
    b_daily = np.repeat(np.nan_to_num(b_week, nan=0.0), 7)[:n_days]
    S = n_pop
    E = infl * w0 * 3 / 7
    I = infl * w0 * 5 / 7
    daily = np.empty(n_days)
    for d in range(n_days):
        beta = beta0 * max(0.05, 1 - eps * b_daily[d] / 4)
        new_inf = beta * S * I / n_pop
        e2i, i2r = SIGMA * E, GAMMA * I
        S -= new_inf
        E += new_inf - e2i
        I += e2i - i2r
        daily[d] = new_inf
    return daily.reshape(n_weeks, 7).sum(axis=1)


def fit_eval(name, b_train, b_test, obs_week, w0, n_pop, n_train, n_weeks,
             eps_grid=EPS_GRID):
    """网格拟合训练期 + 保留期外推. 返回最优参数与轨迹."""
    best = None
    b_full_cfg = np.concatenate([b_train, b_test])
    for beta0, eps, infl, lag in itertools.product(BETA0_GRID, eps_grid, INFL_GRID, LAG_GRID):
        m_week = simulate(b_full_cfg, beta0, eps, infl, w0, n_pop, n_weeks)
        mr = m_week / infl  # kappa 前报告病例形状
        # 训练期对齐: model(t-lag) vs obs(t)
        tr_slice = slice(lag, n_train)
        m_tr = mr[lag - lag:n_train - lag] if lag > 0 else mr[:n_train]
        o_tr = obs_week[lag:n_train] if lag > 0 else obs_week[:n_train]
        # 上面简化: obs 窗口 [lag, n_train), model 窗口 [0, n_train-lag)
        m_tr = mr[:n_train - lag]
        denom = np.sum(m_tr * m_tr)
        if denom <= 0:
            continue
        kappa = max(0.0, np.sum(m_tr * o_tr) / denom)
        fit_tr = kappa * m_tr
        rmse_tr = float(np.sqrt(np.mean((fit_tr - o_tr) ** 2)))
        key = (rmse_tr, beta0, eps, infl, lag, kappa)
        if best is None or key[0] < best[0]:
            best = key
    rmse_tr, beta0, eps, infl, lag, kappa = best
    # 最终轨迹
    m_week = simulate(b_full_cfg, beta0, eps, infl, w0, n_pop, n_weeks)
    mr = m_week / infl
    m_tr = mr[:n_train - lag]
    o_tr = obs_week[lag:n_train]
    pred_week = kappa * mr
    # test: obs 保留期窗口与 model 对齐 (同 lag 平移)
    te_m = pred_week[n_train - lag:n_weeks - lag]
    te_o = obs_week[n_train:n_weeks]
    rmse_te = float(np.sqrt(np.mean((te_m - te_o) ** 2)))
    peak_err = int(abs(int(np.argmax(te_m)) - int(np.argmax(te_o))))
    return {"config": name, "beta0": round(beta0, 4), "eps": eps, "infl": infl,
            "lag_weeks": lag, "kappa": round(kappa, 4), "train_rmse": round(rmse_tr, 1),
            "test_rmse": round(rmse_te, 1), "peak_week_err": peak_err,
            "b_test_mean": round(float(np.nanmean(b_test)), 3)}, pred_week


def main():
    df, burden = load_df()
    obs_cases = df[["week_start"]].merge(burden[["week_start", "cases"]], on="week_start",
                                         how="left")["cases"].to_numpy()
    n_weeks = len(df)
    n_train = n_weeks - N_TEST
    w0 = float(burden.set_index("week_start").loc[df["week_start"].iloc[0], "cases"])
    n_pop = 331_000_000.0
    bser = load_b_series(df)
    b_train = bser["obs"][:n_train]  # 训练期一律用观测
    print(f"n_weeks={n_weeks}, train={n_train}, w0={w0:.0f}, "
          f"obs cases train mean={obs_cases[:n_train].mean():.0f}")

    fits, trajs = [], []
    # no_feedback: b=0 全周期, eps 无效 -> 仅 beta0 x infl x lag
    res, traj = fit_eval("no_feedback", np.zeros(n_train), np.zeros(N_TEST),
                         obs_cases, w0, n_pop, n_train, n_weeks, eps_grid=[0.0])
    fits.append(res)
    trajs.append(("no_feedback", traj))
    print("no_feedback:", res)

    for name in ["rule_driven", "llm_driven"]:
        b_test = bser["rule" if name == "rule_driven" else "llm"][n_train:]
        res, traj = fit_eval(name, b_train, b_test, obs_cases, w0, n_pop, n_train, n_weeks)
        fits.append(res)
        trajs.append((name, traj))
        print(f"{name}:", res)

    fdf = pd.DataFrame(fits)
    fdf.to_csv(PANEL + r"\seir_fits.csv", index=False)

    tdf = pd.DataFrame({"week_start": df["week_start"].dt.date, "obs_cases": obs_cases})
    for name, traj in trajs:
        tdf[f"pred_{name}"] = np.round(traj).astype(int)
    tdf.to_csv(PANEL + r"\seir_traj.csv", index=False)

    # 判定
    nf = fdf.set_index("config")
    base_te = nf.loc["no_feedback", "test_rmse"]
    lines = ["# T2 SEIR 轨道: 行为反馈的轨迹预测增益\n"]
    lines.append(f"- 训练 {n_train} 周 (至 {df['week_start'].iloc[n_train-1].date()}), "
                 f"保留 {N_TEST} 周 Delta 波 ({df['week_start'].iloc[n_train].date()} .. "
                 f"{df['week_start'].iloc[-1].date()})")
    lines.append(f"- 网格: beta0 {len(BETA0_GRID)} x eps {len(EPS_GRID)} x infl {INFL_GRID} "
                 f"x lag {LAG_GRID}; kappa 训练期解析最小二乘\n")
    lines.append("| config | beta0 | eps | infl | lag | kappa | train RMSE | test RMSE | 峰值周误差 | 保留期 b 均值 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in fdf.iterrows():
        lines.append(f"| {r['config']} | {r['beta0']} | {r['eps']} | {r['infl']} | {r['lag_weeks']} "
                     f"| {r['kappa']} | {r['train_rmse']:.0f} | {r['test_rmse']:.0f} "
                     f"| {r['peak_week_err']} 周 | {r['b_test_mean']} |")
    lines.append("")
    for name in ["rule_driven", "llm_driven"]:
        te = nf.loc[name, "test_rmse"]
        d = (base_te - te) / base_te * 100
        verdict = "优于无反馈 (有增益)" if te < base_te else "不优于无反馈 (否证)"
        lines.append(f"- **{name}** test RMSE {te:.0f} vs 无反馈 {base_te:.0f} "
                     f"({d:+.1f}%) -> {verdict}")
    lines.append("")
    lines.append("## T2 判定\n")
    gain_rule = base_te - nf.loc["rule_driven", "test_rmse"]
    gain_llm = base_te - nf.loc["llm_driven", "test_rmse"]
    if gain_rule > 0 or gain_llm > 0:
        lines.append(f"- 部分支持: 行为驱动配置中 {'规则' if gain_rule > gain_llm else 'LLM'} 驱动 "
                     f"优于无反馈; 增益规则 {gain_rule:+.0f} / LLM {gain_llm:+.0f} (周病例 RMSE)")
    else:
        lines.append("- **T2 否证**: 保留期行为驱动均不优于无反馈 -> 行为反馈未转化为轨迹预测增益")
    md = "\n".join(lines)
    with open(PANEL + r"\seir_eval.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
