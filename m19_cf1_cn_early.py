# -*- coding: utf-8 -*-
"""M19 Q1: 中国提前 3 个月放开 (预注册 M19-反事实四问-预注册.md v1.0 §3 机械执行).

基线: M14 主情景 (m14_h2_grid.csv argmin loss 复核, t0=2022-12-01, 窗 68 天)
Q1a: t0=2022-09-01, beta/I0 原值锁死 (纯时间平移)
Q1b: t0=2022-09-01 + I0=0.1*I0_m (9 月放开清零更彻底 -> 免疫空白更深)
指标 (相对比较, 三层边界): dPeak (日峰差, 天) / dPeakH (峰高相对 %) /
    dCum (窗末累计相对 %) / rho (同窗日新增序列 Spearman, 无 lag)
判定: 预注册 §1 H4a 预期区间与意外判据
"""
import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
PANEL = ROOT + r"\data\panel"
SIGMA, GAMMA = 1 / 3, 1 / 5
N_CN = 1_411_780_000
N_DAYS = 68

# --- 基线复核: grid argmin ---
grid = pd.read_csv(PANEL + r"\m14_h2_grid.csv")
row = grid.loc[grid["loss"].idxmin()]
BETA_M, I0_M = float(row["beta"]), float(row["I0"])
print(f"[基线复核] grid argmin: beta={BETA_M:.4f} I0={I0_M:.4e} loss={row['loss']:.4f}")
print(f"[预注册记录] beta=0.83 I0=5.0119e4 -> {'一致' if abs(BETA_M-0.83)<1e-9 and abs(I0_M-5.0119e4)/5.0119e4<0.01 else '不一致, 以 grid 为准 (留痕)'}")

T0_BASE = pd.Timestamp("2022-12-01")
T0_EARLY = pd.Timestamp("2022-09-01")


def run_seir(beta, I0, S0=None, n_days=N_DAYS):
    S0 = N_CN if S0 is None else S0
    S, E, I = float(S0), float(I0), float(I0)
    daily = np.empty(n_days)
    for d in range(n_days):
        new_inf = beta * S * I / N_CN
        e2i, i2r = SIGMA * E, GAMMA * I
        S -= new_inf
        E += new_inf - e2i
        I += e2i - i2r
        daily[d] = new_inf
    return daily


def metrics(d_base, d_cf):
    pk_b, pk_c = int(np.argmax(d_base)), int(np.argmax(d_cf))
    return dict(
        dPeak=pk_c - pk_b,
        dPeakH=(d_cf[pk_c] / d_base[pk_b] - 1) * 100,
        dCum=(d_cf.sum() / d_base.sum() - 1) * 100,
        rho=float(pd.Series(d_base).corr(pd.Series(d_cf), method="spearman")),
    )


# --- 三情景 ---
base = run_seir(BETA_M, I0_M)
q1a = run_seir(BETA_M, I0_M)            # 与基线逐日同值 (平移验证)
q1b = run_seir(BETA_M, I0_M * 0.1)

m_a, m_b = metrics(base, q1a), metrics(base, q1b)

print("\n== Q1a 纯平移 (t0=2022-09-01, 参数原值) ==")
for k, v in m_a.items():
    print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v:+d}")
print(f"  判定 H4a: |dPeakH|<5% -> {'支持' if abs(m_a['dPeakH']) < 5 else '意外'}, "
      f"|dCum|<5% -> {'支持' if abs(m_a['dCum']) < 5 else '意外'}")

print("\n== Q1b 免疫空白更深 (t0=2022-09-01, I0=0.1*I0_m) ==")
for k, v in m_b.items():
    print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v:+d}")
ok_peak = 1 <= m_b["dPeak"] <= 4
ok_cum = abs(m_b["dCum"]) < 15
ok_delay = m_b["dPeak"] <= 7
print(f"  判定 H4b-预期: 峰推迟 1-4 天 -> {'支持' if ok_peak else '意外'}; "
      f"|dCum|<15% -> {'支持' if ok_cum else '意外'}; 意外判据 dPeak>7天 -> "
      f"{'未触发' if ok_delay else '触发'}")

# --- 落盘 ---
out = pd.DataFrame({
    "date_base": pd.date_range(T0_BASE, periods=N_DAYS),
    "base_daily": base,
    "date_q1a": pd.date_range(T0_EARLY, periods=N_DAYS),
    "q1a_daily": q1a,
    "date_q1b": pd.date_range(T0_EARLY, periods=N_DAYS),
    "q1b_daily": q1b,
})
out.to_csv(PANEL + r"\m19_cf1_cn_early_traj.csv", index=False)
pd.DataFrame([
    {"scenario": "base(t0=12-01)", "beta": BETA_M, "I0": I0_M, **metrics(base, base)},
    {"scenario": "Q1a(t0=09-01)", "beta": BETA_M, "I0": I0_M, **m_a},
    {"scenario": "Q1b(t0=09-01,I0x0.1)", "beta": BETA_M, "I0": I0_M * 0.1, **m_b},
]).to_csv(PANEL + r"\m19_cf1_metrics.csv", index=False)
print("\n落盘: m19_cf1_cn_early_traj.csv / m19_cf1_metrics.csv")
