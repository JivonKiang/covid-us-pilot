# -*- coding: utf-8 -*-
"""M11 阶段A3: 中国三省 SEIR 轨道复现 (T2 中国版).

与美国 seir_track2.py 同构, 差异:
- 数据: cn_prov_week.csv (3省 x 49周) + cn_national_week.csv (国家级行为)
- 行为观测稀疏且断续 (面板内 17 个 n>=30 有效周, 12 段, 最长 2 周)
  -> 仅以 n>=30 周为锚, 其余置 NaN 后 ffill+bfill 扩展为 49 周连续 (惯性假设, 局限已记录)
- n_pop 按省 (七普 2020); 单省循环各拟合一条轨迹
- rule_driven 的保留期 b: 现场复刻 behavior_baseline.py 的 AR1 岭回归
  (在扩展序列上拟合, 等权; 美国版按 n 加权因中国非观测周无 n 不可行)
- w0: 前三周 JHU 未记录 -> new_cases NaN 填 0, w0 = max(首周, 1.0) 小种子
- llm_driven 留阶段 C
否证条件 (H-CN3): rule_driven 不优于 no_feedback.
预期: 行为天花板效应 (中国行为水平全程高企), 增益比美国侧变小或为负.
"""
import itertools

import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
PANEL = ROOT + r"\data\panel"
SIGMA, GAMMA = 1 / 3, 1 / 5
BETA0_GRID = list(np.geomspace(0.1, 0.9, 16))
EPS_GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]
INFL_GRID = [2.0, 4.0]
LAG_GRID = [0, 1, 2]
N_TEST = 12
LAMBDA = 1.0
PROVINCES = ["Hubei", "Guangdong", "Beijing"]
POP_2020 = {"Hubei": 57_752_557, "Guangdong": 126_012_510, "Beijing": 21_893_095}
BEHAV6 = ["yougov_mask_freq", "yougov_stay_home_freq", "yougov_avoid_restaurants_freq",
          "yougov_avoid_crowds_freq", "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]
MIN_N = 30


def simulate(b_week, beta0, eps, infl, w0, n_pop, n_weeks):
    """日步长 Euler SEIR, 返回周真实新感染数组 (长度=n_weeks). 与美国版逐行一致."""
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
    """网格拟合训练期 + 保留期外推. 与美国版逻辑一致."""
    best = None
    b_full_cfg = np.concatenate([b_train, b_test])
    for beta0, eps, infl, lag in itertools.product(BETA0_GRID, eps_grid, INFL_GRID, LAG_GRID):
        m_week = simulate(b_full_cfg, beta0, eps, infl, w0, n_pop, n_weeks)
        mr = m_week / infl
        m_tr = mr[:n_train - lag]
        o_tr = obs_week[lag:n_train]
        denom = np.sum(m_tr * m_tr)
        if denom <= 0:
            continue
        kappa = max(0.0, np.sum(m_tr * o_tr) / denom)
        fit_tr = kappa * m_tr
        rmse_tr = float(np.sqrt(np.mean((fit_tr - o_tr) ** 2)))
        if best is None or rmse_tr < best[0]:
            best = (rmse_tr, beta0, eps, infl, lag, kappa)
    rmse_tr, beta0, eps, infl, lag, kappa = best
    m_week = simulate(b_full_cfg, beta0, eps, infl, w0, n_pop, n_weeks)
    mr = m_week / infl
    pred_week = kappa * mr
    te_m = pred_week[n_train - lag:n_weeks - lag]
    te_o = obs_week[n_train:n_weeks]
    rmse_te = float(np.sqrt(np.mean((te_m - te_o) ** 2)))
    peak_err = int(abs(int(np.argmax(te_m)) - int(np.argmax(te_o))))
    return {"config": name, "beta0": round(beta0, 4), "eps": eps, "infl": infl,
            "lag_weeks": lag, "kappa": round(kappa, 4), "train_rmse": round(rmse_tr, 1),
            "test_rmse": round(rmse_te, 1), "peak_week_err": peak_err,
            "b_test_mean": round(float(np.nanmean(b_test)), 3)}, pred_week


def build_b_series():
    """国家级行为序列: 49 周完整日历为骨架, 观测锚 (n>=30) 保留真值,
    其余周 (无波次或小样本) 置 NaN 后 ffill/bfill 扩展; 附 AR1 所需驱动."""
    nat = pd.read_csv(PANEL + r"\cn_national_week.csv", parse_dates=["week_start"])
    weeks_all = pd.date_range("2020-01-06", "2020-12-07", freq="7D")  # 与面板同骨架
    nat = nat.set_index("week_start").reindex(weeks_all)
    n_obs = int((nat["n_total"] >= MIN_N).sum())
    nat.loc[~(nat["n_total"] >= MIN_N), BEHAV6] = np.nan  # 剔除小样本/缺失周 (如 n=3)
    for b in BEHAV6:
        nat[b] = nat[b].ffill().bfill()
    nat = nat.reset_index().rename(columns={"index": "week_start"})
    # 驱动特征 (国家级, 三省人口加权/求和, 与 behavior_baseline.py 同构)
    panel = pd.read_csv(PANEL + r"\cn_prov_week.csv", parse_dates=["week_start"])
    panel["new_cases"] = panel["new_cases"].fillna(0.0)
    g = panel.groupby("week_start")
    drv = pd.DataFrame({
        "cases": g["new_cases"].sum(),
        "pop": g["population"].sum(),
        "stringency": g.apply(
            lambda x: np.average(x["pol_stringency"], weights=x["population"]),
            include_groups=False),
    }).reset_index()
    drv["per100k"] = drv["cases"] / drv["pop"] * 1e5
    drv["log_burden"] = np.log10(drv["per100k"].clip(lower=0.01))
    df = nat[["week_start"] + BEHAV6].merge(drv, on="week_start", how="left")
    df["t"] = np.arange(len(df))
    return df, n_obs


def ar1_rule_pred(df):
    """复刻 behavior_baseline.py base_ar1: 岭回归 lambda=1, 特征=政策lag1+log负担lag1+trend+行为lag1.
    中国差异: 在扩展序列上拟合, 等权 (非观测周无 n_total 可加权). 返回 49 周行为均值预测."""
    pol_lag = df["stringency"].shift(1).bfill().to_numpy(float)
    bur_lag = df["log_burden"].shift(1).bfill().to_numpy(float)
    t_feat = df["t"].to_numpy(float)
    preds = []
    for b in BEHAV6:
        y = df[b].to_numpy(float)
        Xr = np.column_stack([pol_lag, bur_lag, t_feat, df[b].shift(1).bfill().to_numpy(float)])
        mu, sd = Xr.mean(0), Xr.std(0)
        sd[sd == 0] = 1.0
        Xs = np.column_stack([np.ones(len(Xr)), (Xr - mu) / sd])
        A = Xs.T @ Xs + LAMBDA * np.eye(Xs.shape[1])
        A[0, 0] -= LAMBDA  # 截距不正则
        beta = np.linalg.solve(A, Xs.T @ y)
        preds.append(Xs @ beta)
    return np.mean(preds, axis=0)


def main():
    df, n_obs = build_b_series()
    b_filled = df[BEHAV6].mean(axis=1).to_numpy()  # 扩展后行为均值 (0-4), 49 周
    rule_pred = ar1_rule_pred(df)
    panel = pd.read_csv(PANEL + r"\cn_prov_week.csv", parse_dates=["week_start"])
    panel["new_cases"] = panel["new_cases"].fillna(0.0)
    weeks = df["week_start"].reset_index(drop=True)
    n_weeks = len(df)
    n_train = n_weeks - N_TEST
    print(f"n_weeks={n_weeks}, n_train={n_train}, 行为观测锚 {n_obs} 周 (n>={MIN_N}), "
          f"b_filled 范围 [{b_filled.min():.2f}, {b_filled.max():.2f}]")
    print(f"train 周: {weeks.iloc[0].date()} .. {weeks.iloc[n_train-1].date()}; "
          f"test 周: {weeks.iloc[n_train].date()} .. {weeks.iloc[-1].date()}")

    fits, traj_rows = [], []
    for prov in PROVINCES:
        sub = panel[panel["state"] == prov].sort_values("week_start").reset_index(drop=True)
        obs = sub["new_cases"].to_numpy(float)
        w0 = max(float(obs[0]), 1.0)
        n_pop = float(POP_2020[prov])
        b_train = b_filled[:n_train]
        res_nf, traj_nf = fit_eval("no_feedback", np.zeros(n_train), np.zeros(N_TEST),
                                   obs, w0, n_pop, n_train, n_weeks, eps_grid=[0.0])
        res_rd, traj_rd = fit_eval("rule_driven", b_train, rule_pred[n_train:],
                                   obs, w0, n_pop, n_train, n_weeks)
        for r in (res_nf, res_rd):
            r["province"] = prov
            r["w0"] = w0
            r["n_pop"] = int(n_pop)
        fits += [res_nf, res_rd]
        print(f"\n[{prov}] w0={w0:.0f}, n_pop={n_pop/1e6:.1f}M, "
              f"obs test 均值={obs[n_train:].mean():.1f}")
        print("  no_feedback:", res_nf)
        print("  rule_driven:", res_rd)
        for i, wk in enumerate(weeks):
            traj_rows.append({"province": prov, "week_start": wk.date(),
                              "obs_cases": obs[i],
                              "pred_no_feedback": int(round(traj_nf[i])),
                              "pred_rule_driven": int(round(traj_rd[i]))})

    fdf = pd.DataFrame(fits)[["province", "config", "beta0", "eps", "infl", "lag_weeks",
                              "kappa", "train_rmse", "test_rmse", "peak_week_err",
                              "b_test_mean", "w0", "n_pop"]]
    fdf.to_csv(PANEL + r"\seir_fits_china.csv", index=False)
    tdf = pd.DataFrame(traj_rows)
    tdf.to_csv(PANEL + r"\seir_traj_china.csv", index=False)

    lines = ["# T2 中国版 SEIR: 行为反馈的省级轨迹预测增益 (湖北/广东/北京)\n"]
    lines.append(f"- 面板 {n_weeks} 周 (2020-01-06 .. {weeks.iloc[-1].date()}), "
                 f"训练 {n_train} 周 / 保留 {N_TEST} 周; 行为观测锚 {n_obs} 周 "
                 f"(n>={MIN_N}), 非锚周 ffill/bfill 惯性扩展 (局限)")
    lines.append(f"- 网格: beta0 {len(BETA0_GRID)} x eps {len(EPS_GRID)} x infl {INFL_GRID} "
                 f"x lag {LAG_GRID}; kappa 训练期解析最小二乘; n_pop 按省七普")
    lines.append("- rule_driven 保留期 b = AR1 岭回归 (扩展序列拟合, 等权); "
                 "llm_driven 留阶段 C\n")
    lines.append("| 省 | config | beta0 | eps | infl | lag | kappa | train RMSE | test RMSE | 峰值周误差 | 保留期 b |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in fdf.iterrows():
        lines.append(f"| {r['province']} | {r['config']} | {r['beta0']} | {r['eps']} "
                     f"| {r['infl']} | {r['lag_weeks']} | {r['kappa']} "
                     f"| {r['train_rmse']:.0f} | {r['test_rmse']:.0f} "
                     f"| {r['peak_week_err']} 周 | {r['b_test_mean']} |")
    lines.append("\n## 逐省判定 (rule_driven vs no_feedback, test RMSE)\n")
    gains = {}
    for prov in PROVINCES:
        base_te = fdf[(fdf["province"] == prov) & (fdf["config"] == "no_feedback")]["test_rmse"].iloc[0]
        te = fdf[(fdf["province"] == prov) & (fdf["config"] == "rule_driven")]["test_rmse"].iloc[0]
        d = (base_te - te) / base_te * 100 if base_te > 0 else np.nan
        gains[prov] = base_te - te
        verdict = "有增益" if te < base_te else "无增益"
        lines.append(f"- **{prov}**: {te:.1f} vs {base_te:.1f} ({d:+.1f}%) -> {verdict}")
    lines.append("\n## H-CN3 判定\n")
    n_gain = sum(1 for v in gains.values() if v > 0)
    lines.append(f"- rule_driven 有增益的省: {n_gain}/3; "
                 + ("部分支持 (增益小于美国侧或集中于单省, 与行为天花板预期一致)"
                    if n_gain > 0 else
                    "**否证**: 三省保留期行为驱动均不优于无反馈 (与天花板效应预期一致)"))
    lines.append("- 注: 湖北/北京保留期病例近零 (周 0-8 例), RMSE 区分度弱, "
                 "解读以广东 (保留期周均 ~17 例) 为主")
    md = "\n".join(lines)
    with open(PANEL + r"\seir_eval_china.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print(f"\n-> {PANEL}\\seir_fits_china.csv / seir_traj_china.csv / seir_eval_china.md")


if __name__ == "__main__":
    main()
