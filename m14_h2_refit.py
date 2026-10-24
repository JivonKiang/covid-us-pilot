# -*- coding: utf-8 -*-
"""M14: H2 要素 C — 中国放开波事后回溯拟合 (预注册 M14-双国复现H2-预注册.md v1.1 的机械执行).

预注册约束 (§3.3 + §6 偏差预声明):
- 管线: SEIR 日步长 Euler, sigma=1/3, gamma=1/5, N=14.117e8, b=0, E0=I0, S0=N, t0=2022-12-01
- 自由度仅 2: beta_omicron_fit x I0 (网格扫描, log 空间最小二乘)
- 拟合目标 (偏差 D1): 发热门诊 3 日观测点 log 最小二乘
    观测 (万人次/日 -> 人/日): 2022-12-09=42.5e4, 2022-12-23=286.7e4, 2023-02-06=13.7e4
    模型侧: 当日模拟日新增感染 (就诊率常数 log 空间被尺度吸收)
- 判定 (主情景单值 = argmin loss):
    C1 峰周起始日 in [2022-11-30, 2022-12-27] (偏差 D3: t0 起每 7 天分周 argmax, 另报日峰)
    C2 累计感染率@2023-02-05 in [20.1%, 100%]
    C3 模拟周新增感染 vs ILI 新冠阳性率周序列 (9 点) Spearman rho >= 0.8, lag {0,7} 取最优 (偏差 D2)
- 敏感性包络: 网格全组合 C1/C2/C3 命中比例 (仅报告, 不判定)
- 辅助校验: 发热门诊 3 点定性形态 (起点低/峰高/末点低)
"""
import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
OUT_PANEL = ROOT + r"\data\panel"
SIGMA, GAMMA = 1 / 3, 1 / 5
N_CN = 1_411_780_000
T0 = pd.Timestamp("2022-12-01")
N_DAYS = 68  # d0=12-01 ... d67=2023-02-06

FEVER_OBS = {  # 人/日 (万人次 x 1e4)
    "2022-12-09": 42.5e4,
    "2022-12-23": 286.7e4,
    "2023-02-06": 13.7e4,
}
FEVER_IDX = {(pd.Timestamp(k) - T0).days: v for k, v in FEVER_OBS.items()}

ILI_END = ["2022-12-11", "2022-12-18", "2022-12-25", "2023-01-01", "2023-01-08",
           "2023-01-15", "2023-01-22", "2023-01-29", "2023-02-05"]
ILI_POS = [7.8, 33.2, 59.5, 60.0, 45.3, 24.2, 12.3, 7.8, 5.7]  # %, 图 3-1 文本 9 点

C1_LO, C1_HI = "2022-11-30", "2022-12-27"
C2_LO, C2_HI = 0.201, 1.00
C3_MIN_RHO = 0.8
LAGS = [0, 7]  # 偏差 D2: [0,7] 天带离散化为两档


def run_seir(beta, I0):
    S, E, I = float(N_CN), float(I0), float(I0)
    daily = np.empty(N_DAYS)
    for d in range(N_DAYS):
        new_inf = beta * S * I / N_CN
        e2i, i2r = SIGMA * E, GAMMA * I
        S -= new_inf
        E += new_inf - e2i
        I += e2i - i2r
        daily[d] = new_inf
    return daily


def loss_fn(daily):
    s = 0.0
    for d, obs in FEVER_IDX.items():
        s += (np.log(max(daily[d], 1.0)) - np.log(obs)) ** 2
    return s


def weekly_model(daily, lag):
    """周窗口 [end-6-lag, end-lag], 起点早于 t0 截断; 返回 9 点周新增感染."""
    out = []
    for e in ILI_END:
        end_i = (pd.Timestamp(e) - T0).days
        lo = max(end_i - 6 - lag, 0)
        out.append(float(daily[lo:end_i + 1].sum()))
    return np.array(out)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def c3_eval(daily):
    best_rho, best_lag = -2.0, None
    for lag in LAGS:
        rho = spearman(weekly_model(daily, lag), np.array(ILI_POS))
        if rho > best_rho:
            best_rho, best_lag = rho, lag
    return best_rho, best_lag


def c1_eval(daily):
    wk = daily[:63].reshape(9, 7).sum(axis=1)  # 12-01 起 9 个整周 (至 02-01)
    pk = int(np.argmax(wk))
    return (T0 + pd.Timedelta(days=7 * pk)).strftime("%Y-%m-%d"), \
           (T0 + pd.Timedelta(days=int(np.argmax(daily)))).strftime("%Y-%m-%d")


def c2_eval(daily):
    return float(np.cumsum(daily)[66]) / N_CN  # d0..d66 含 02-05 (cumsum 到 d66)


def main():
    betas = np.round(np.arange(0.30, 1.26, 0.01), 4)
    i0s = np.logspace(4, 7.5, 36)
    rows = []
    for beta in betas:
        for i0 in i0s:
            daily = run_seir(float(beta), float(i0))
            pw, dp = c1_eval(daily)
            c2 = c2_eval(daily)
            rho, lag = c3_eval(daily)
            rows.append(dict(beta=float(beta), I0=float(i0),
                             loss=loss_fn(daily),
                             peak_week_start=pw, daily_peak_date=dp,
                             cum_rate_0205=c2, rho_best=rho, lag_best=lag,
                             C1_hit=C1_LO <= pw <= C1_HI,
                             C2_hit=C2_LO <= c2 <= C2_HI,
                             C3_hit=rho >= C3_MIN_RHO))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_PANEL + r"\m14_h2_grid.csv", index=False)

    main_row = grid.loc[grid["loss"].idxmin()]
    beta_m, i0_m = float(main_row["beta"]), float(main_row["I0"])
    daily_main = run_seir(beta_m, i0_m)
    traj = pd.DataFrame({"date": pd.date_range(T0, periods=N_DAYS),
                         "daily_new_inf": daily_main,
                         "cum_inf": np.cumsum(daily_main)})
    traj.to_csv(OUT_PANEL + r"\m14_h2_traj_main.csv", index=False)

    print("== M14 H2 要素C 拟合 (预注册 v1.1 机械执行) ==")
    print(f"网格: {len(betas)} beta x {len(i0s)} I0 = {len(grid)} 组合, 全部存 m14_h2_grid.csv")
    print(f"\n== 主情景 (argmin loss) ==")
    print(f"beta_omicron_fit = {beta_m:.4f}   I0 = {i0_m:.4e}")
    print(f"loss (3点 log 最小二乘) = {main_row['loss']:.4f}")
    print(f"主情景三点对照 (模型 vs 发热门诊观测, 人/日):")
    for d, obs in FEVER_IDX.items():
        print(f"  {(T0 + pd.Timedelta(days=d)).date()}: "
              f"model={daily_main[d]:.3e}  obs={obs:.3e}  log_ratio={np.log(max(daily_main[d],1.0))-np.log(obs):+.2f}")
    pw, dp = c1_eval(daily_main)
    print(f"\nC1 峰周起始日 = {pw}  (日峰 = {dp})  窗口 [{C1_LO}, {C1_HI}]  -> {'达标' if C1_LO <= pw <= C1_HI else '失手'}")
    c2 = c2_eval(daily_main)
    print(f"C2 累计感染率@2023-02-05 = {c2*100:.1f}%  窗口 [20.1%, 100%]  -> {'达标' if C2_LO <= c2 <= C2_HI else '失手'}")
    rho, lag = c3_eval(daily_main)
    print(f"C3 Spearman rho = {rho:.3f} (最优 lag = {lag} 天)  阈值 >= {C3_MIN_RHO}  -> {'达标' if rho >= C3_MIN_RHO else '失手'}")
    print(f"  两档 rho: " + ", ".join(f"lag{L}d={spearman(weekly_model(daily_main, L), np.array(ILI_POS)):.3f}" for L in LAGS))

    print(f"\n== 敏感性包络 (仅报告) ==")
    for c in ["C1_hit", "C2_hit", "C3_hit"]:
        print(f"{c}: {grid[c].mean()*100:.1f}% 命中 ({int(grid[c].sum())}/{len(grid)})")
    all_hit = grid[grid[["C1_hit", "C2_hit", "C3_hit"]].all(axis=1)]
    print(f"三合一命中: {len(all_hit)}/{len(grid)}")
    if len(all_hit):
        print(f"  命中区 beta [{all_hit['beta'].min():.2f}, {all_hit['beta'].max():.2f}] "
              f"I0 [{all_hit['I0'].min():.2e}, {all_hit['I0'].max():.2e}]")

    # 对照 (仅报告, 不判定): C1 窗口内组合中 loss 最小者
    in_win = grid[grid["C1_hit"]]
    if len(in_win):
        w = in_win.loc[in_win["loss"].idxmin()]
        dw = run_seir(float(w["beta"]), float(w["I0"]))
        print(f"\n== 对照: C1 窗口内组合 argmin loss (beta={w['beta']:.2f}, I0={w['I0']:.3e}, loss={w['loss']:.3f}) ==")
        for d, obs in FEVER_IDX.items():
            print(f"  {(T0 + pd.Timedelta(days=d)).date()}: model={dw[d]:.3e}  obs={obs:.3e}  "
                  f"log_ratio={np.log(max(dw[d],1.0))-np.log(obs):+.2f}")
        print(f"  C2={w['cum_rate_0205']*100:.1f}%  C3 rho={w['rho_best']:.3f}(lag {int(w['lag_best'])}d)")

    print("\n主情景轨迹已存:", OUT_PANEL + r"\m14_h2_traj_main.csv")


if __name__ == "__main__":
    main()
