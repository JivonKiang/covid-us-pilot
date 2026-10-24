# -*- coding: utf-8 -*-
"""M19 Q2: 中国渐进沟通 (预注册 M19-反事实四问-预注册.md v1.0 §3 机械执行).

基线: M14 主情景 (beta=0.83 常数 = 静默切换即阶跃, t0=2022-12-01, I0=5.0119e4)
Q2: beta(d) = beta_m * max(0.3, min(1, 0.3 + 0.7*d/42))  (42 天线性缓坡)
    beta(t) 逐日更新, 步进公式不变 (预注册 §2.3 显式扩展 D5)
指标: dPeak / dPeakH / dCum / rho (同基线日新增序列)
判定: 预注册 §1 H4b (峰推迟>=7天; 峰高降幅 10-40%; |dCum|<10%;
      意外: 峰高降幅<5% 或 >60%, 或 |dCum|>10%)
"""
import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
PANEL = ROOT + r"\data\panel"
SIGMA, GAMMA = 1 / 3, 1 / 5
N_CN = 1_411_780_000
N_DAYS = 68
T0 = pd.Timestamp("2022-12-01")

grid = pd.read_csv(PANEL + r"\m14_h2_grid.csv")
row = grid.loc[grid["loss"].idxmin()]
BETA_M, I0_M = float(row["beta"]), float(row["I0"])
print(f"[基线复核] beta={BETA_M:.4f} I0={I0_M:.4e}")


def run_seir(beta_path, I0):
    S, E, I = float(N_CN), float(I0), float(I0)
    daily = np.empty(N_DAYS)
    for d in range(N_DAYS):
        new_inf = beta_path(d) * S * I / N_CN
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


base = run_seir(lambda d: BETA_M, I0_M)
q2 = run_seir(lambda d: BETA_M * max(0.3, min(1.0, 0.3 + 0.7 * d / 42.0)), I0_M)
m = metrics(base, q2)

print("\n== Q2 渐进沟通 (beta 缓坡 42 天, 0.3x -> 1.0x) ==")
print(f"  dPeak: {m['dPeak']:+d} 天  (基线日峰 d{int(np.argmax(base))}, 缓坡日峰 d{int(np.argmax(q2))})")
print(f"  dPeakH: {m['dPeakH']:+.2f}%")
print(f"  dCum: {m['dCum']:+.2f}%")
print(f"  rho: {m['rho']:+.4f}")
print(f"\n判定 H4b: 峰推迟>=7天 -> {'支持' if m['dPeak'] >= 7 else '否证'}; "
      f"峰高降幅 10-40% -> {'支持' if 10 <= -m['dPeakH'] <= 40 else '意外'}; "
      f"|dCum|<10% -> {'支持' if abs(m['dCum']) < 10 else '意外'}")
if m["dPeakH"] > -5:
    print(f"  意外判据触发: 峰高降幅<5% (渐进通道无效)")
elif m["dPeakH"] < -60:
    print(f"  意外判据触发: 峰高降幅>60% (单波假设破裂)")
elif abs(m["dCum"]) >= 10:
    print(f"  意外判据触发: |dCum|>=10%")

pd.DataFrame({
    "date": pd.date_range(T0, periods=N_DAYS),
    "base_daily": base, "q2_daily": q2,
    "q2_beta_factor": [BETA_M * max(0.3, min(1.0, 0.3 + 0.7 * d / 42.0)) / BETA_M for d in range(N_DAYS)],
}).to_csv(PANEL + r"\m19_cf2_cn_gradual_traj.csv", index=False)
pd.DataFrame([{"scenario": "base(beta_step)", **metrics(base, base)},
              {"scenario": "Q2(beta_ramp42d)", **m}]).to_csv(PANEL + r"\m19_cf2_metrics.csv", index=False)
print("\n落盘: m19_cf2_cn_gradual_traj.csv / m19_cf2_metrics.csv")
