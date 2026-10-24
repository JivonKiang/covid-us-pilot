# -*- coding: utf-8 -*-
"""M13: H3 放开波盲测 — 标定 + 模拟 (预注册 M13-放开波盲测-预注册.md 的机械执行).

预注册约束:
- 标定: JHU 美国周度新增, 2021-12-06 ~ 2022-01-03 (BA.1 指数期), log OLS -> r(每周) -> r_d, T2, beta_omicron
- 模拟: SEIR 日步长 Euler, sigma=1/3, gamma=1/5, N=14.117e8, 起始 2022-11-28, 14 周
- 行为: 11-28~12-06 b_pre (2020 面板末4周六项均值, 1-4 尺度); 12-07~2023-01-31 b_resid; 之后 0
- 网格: I0 {1e5,1e6,5e6} x S0_frac {1.0,0.8} x b_resid {0,0.5,1.0} = 18 组合, 主情景 I0=1e6/S0=N/b=0
- 输出: P1 峰周起始日, P2 峰值周新增感染, P3 累计感染率@2023-03-05, P4 @2023-01-31
本脚本不得读取任何 2022-11 之后的中国负担数据 (预注册 §2.2).
"""
import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
OUT_PANEL = ROOT + r"\data\panel"
SIGMA, GAMMA = 1 / 3, 1 / 5
N_CN = 1_411_780_000
START = pd.Timestamp("2022-11-28")
N_WEEKS = 14
N_DAYS = N_WEEKS * 7
BEHAV6 = ["yougov_mask_freq", "yougov_stay_home_freq", "yougov_avoid_restaurants_freq",
          "yougov_avoid_crowds_freq", "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]
EPSILON = 1.0  # beta = beta_omicron * max(0.05, 1 - eps*b/4), b 1-4 尺度


# ---------- 1. 标定: 美国 BA.1 波 (预注册 v1.1 机械扫描窗口) ----------
def calibrate():
    df = pd.read_csv(ROOT + r"\data\raw\jhu_ts_confirmed_us.csv")
    date_cols = [c for c in df.columns if "/" in c]
    ts = df[date_cols].sum(axis=0)
    ts.index = pd.to_datetime(ts.index, format="%m/%d/%y")
    ts = ts.sort_index()
    cum = ts
    # 候选窗口: 周一起始, 连续 4 个自然周, 覆盖 2021-11-22 至 2022-01-17 结尾
    starts = pd.date_range("2021-11-22", "2021-12-20", freq="7D")
    best = None
    for s0 in starts:
        weekly = []
        for k in range(4):
            a = s0 + pd.Timedelta(days=7 * k)
            b = a + pd.Timedelta(days=6)
            val = cum.loc[b] - cum.loc[a - pd.Timedelta(days=1)]
            weekly.append(max(val, 1.0))
        y = np.log(np.array(weekly))
        x = np.arange(4, dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        if best is None or slope > best[0]:
            best = (slope, s0, weekly)
    r_week, win_start, weekly = best
    win_end = win_start + pd.Timedelta(days=27)
    slope, intercept = np.polyfit(np.arange(4, dtype=float), np.log(np.array(weekly)), 1)
    resid = np.log(np.array(weekly)) - (slope * np.arange(4) + intercept)
    se = np.sqrt(np.sum(resid ** 2) / 2 / np.sum((np.arange(4) - np.arange(4).mean()) ** 2))
    ci = 1.96 * se
    r_week_lo, r_week_hi = slope - ci, slope + ci
    r_d = r_week / 7
    beta = r_d + GAMMA
    beta_lo = (r_week_lo / 7) + GAMMA
    beta_hi = (r_week_hi / 7) + GAMMA
    t2 = np.log(2) / r_d if r_d > 0 else np.nan
    t2_lo = np.log(2) / r_week_hi / 7 if r_week_hi > 0 else np.nan
    t2_hi = np.log(2) / r_week_lo / 7 if r_week_lo > 0 else np.nan
    return dict(window=(win_start.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")),
                weekly=weekly, r_week=r_week, r_week_ci=(r_week_lo, r_week_hi),
                r_d=r_d, T2=t2, T2_ci=(t2_lo, t2_hi),
                beta=beta, beta_ci=(beta_lo, beta_hi))


# ---------- 2. 行为基线 b_pre ----------
def get_b_pre():
    df = pd.read_csv(OUT_PANEL + r"\cn_prov_week.csv", parse_dates=["week_start"])
    tail = df[df["week_start"] >= df["week_start"].max() - pd.Timedelta(weeks=3)]
    vals = []
    for c in BEHAV6:
        v = tail[c].dropna()
        if len(v):
            vals.append(v.mean())
    b_pre = float(np.mean(vals))
    return b_pre, [f"{c}={v:.3f}" for c, v in zip(BEHAV6, vals)]


# ---------- 3. SEIR ----------
def run_seir(beta_omicron, b_pre, b_resid, S0, I0, E0, n_days=N_DAYS,
             resid_start_day=9, resid_end_day=65):
    """resid_start_day: 12-07 距 11-28 = 9 天; resid_end_day: 02-01 距 11-28 = 65 天."""
    S, E, I = float(S0), float(E0), float(I0)
    daily = np.empty(n_days)
    for d in range(n_days):
        if d < resid_start_day:
            b = b_pre
        elif d < resid_end_day:
            b = b_resid
        else:
            b = 0.0
        beta = beta_omicron * max(0.05, 1 - EPSILON * b / 4)
        new_inf = beta * S * I / N_CN
        e2i, i2r = SIGMA * E, GAMMA * I
        S -= new_inf
        E += new_inf - e2i
        I += e2i - i2r
        daily[d] = new_inf
    return daily


def summarize(daily):
    wk = daily.reshape(N_WEEKS, 7).sum(axis=1)
    pk = int(np.argmax(wk))
    p1 = (START + pd.Timedelta(days=7 * pk)).strftime("%Y-%m-%d")
    p2 = float(wk[pk])
    cum = np.cumsum(daily)
    p3 = cum[-1] / N_CN
    p4 = cum[64] / N_CN  # 2023-01-31 = 11-28 + 64 天
    return p1, p2, p3, p4


def main():
    cal = calibrate()
    print("== 标定 (JHU 美国 BA.1 指数期, 预注册 v1.1 机械扫描窗口) ==")
    print(f"选中窗口: {cal['window'][0]} ~ {cal['window'][1]}")
    print(f"周度新增: {[f'{w:.0f}' for w in cal['weekly']]}")
    print(f"r(每周) = {cal['r_week']:.4f}  CI95 [{cal['r_week_ci'][0]:.4f}, {cal['r_week_ci'][1]:.4f}]")
    print(f"r_d = {cal['r_d']:.5f}  倍增时间 T2 = {cal['T2']:.1f} 天  CI95 [{cal['T2_ci'][0]:.1f}, {cal['T2_ci'][1]:.1f}]")
    print(f"beta_omicron = {cal['beta']:.5f}  CI95 [{cal['beta_ci'][0]:.5f}, {cal['beta_ci'][1]:.5f}]")
    beta = cal["beta"]

    b_pre, b_detail = get_b_pre()
    print(f"\nb_pre (2020 面板末4周) = {b_pre:.3f}  ({'; '.join(b_detail)})")

    print("\n== 主情景 (I0=1e6, S0=N, b_resid=0) ==")
    daily_main = run_seir(beta, b_pre, 0.0, N_CN, 1e6, 1.5e6)
    p1, p2, p3, p4 = summarize(daily_main)
    print(f"P1 感染峰周起始: {p1}")
    print(f"P2 峰值周新增感染: {p2/1e8:.3f} 亿 ({p2:.3e})")
    print(f"P3 累计感染率@2023-03-05: {p3*100:.1f}%")
    print(f"P4 累计感染率@2023-01-31: {p4*100:.1f}%")

    print("\n== 敏感性网格 (18 组合) ==")
    rows = []
    for i0 in [1e5, 1e6, 5e6]:
        for s0f in [1.0, 0.8]:
            for br in [0.0, 0.5, 1.0]:
                is_main = (i0 == 1e6 and s0f == 1.0 and br == 0.0)
                d = run_seir(beta, b_pre, br, N_CN * s0f, i0, 1.5 * i0)
                a, bb, c, dd = summarize(d)
                rows.append(dict(I0=i0, S0_frac=s0f, b_resid=br, is_main=is_main,
                                 P1_peak_week=a, P2_peak_weekly_cases=bb,
                                 P3_cum_inf_rate_0305=c, P4_cum_inf_rate_0131=dd))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_PANEL + r"\m13_blindtest_grid.csv", index=False)
    print(grid.to_string(index=False,
                         formatters={"P2_peak_weekly_cases": lambda v: f"{v:.2e}",
                                     "P3_cum_inf_rate_0305": lambda v: f"{v*100:.1f}%",
                                     "P4_cum_inf_rate_0131": lambda v: f"{v*100:.1f}%"}))
    print("\n网格已存:", OUT_PANEL + r"\m13_blindtest_grid.csv")

    # 主情景日度曲线 (供解盲时画轨迹对照)
    traj = pd.DataFrame({"date": pd.date_range(START, periods=N_DAYS),
                         "daily_new_inf": daily_main,
                         "cum_inf": np.cumsum(daily_main)})
    traj.to_csv(OUT_PANEL + r"\m13_blindtest_traj_main.csv", index=False)
    print("主情景轨迹已存:", OUT_PANEL + r"\m13_blindtest_traj_main.csv")


if __name__ == "__main__":
    main()
