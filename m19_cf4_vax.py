# -*- coding: utf-8 -*-
"""M19 Q4: 放开前免疫提升 (预注册 M19-反事实四问-预注册.md v1.0 §3 机械执行).

基线: M14 主情景 (grid argmin 复核, t0=2022-12-01, beta/I0 锁死, 窗 68 天)
Q4: 只改 S0 = N x (1-delta), delta = 60+人口占比 x 覆盖提升 x 防感染效力 (H4d 降级: 全人群免疫墙代理)
三因子公开核实 (执行时落盘引用, 预注册 D6/D2):
    60+全程接种率(2022-11-28) = 86.42%  来源: 国家疾控局(健康报转述) jkb.com.cn/news/depth/2022/1207/487313.html
        注: 预注册草案 76% 与核实值不符, 以核实值为准 (H4d 授权执行时核实, 留痕)
    60+人口占比(2022末) = 19.8%  来源: 国家统计局2022统计公报 stats.gov.cn/sj/zxfb/202302/t20230203_1901709.html
    灭活疫苗基础免疫对 Omicron 感染效力 = 16.4% (95%CI 9.5-22.8)
        来源: 真实世界 meta 回归 pubmed.ncbi.nlm.nih.gov/37106390;
        佐证: 全疫苗 Omicron 感染 VE 23.5% (PMC9542696), CoronaVac 3月内 37.8% (Vaccines 11-00224)
    覆盖提升 = 0.90 - 0.8642 = 0.0358 (0.90 为预注册情景目标, 非经验值)
敏感性 (D6): 效力取 CI 上下界各跑一次, 仅报告不判定
判定: H4d 预期 |dPeakH|<5% 且 |dCum|<5%; 意外判据任一 >10%
"""
import numpy as np
import pandas as pd

PANEL = r"e:\ai agent\research\covid-us-pilot\data\panel"
SIGMA, GAMMA = 1 / 3, 1 / 5
N_CN = 1_411_780_000
N_DAYS = 68
T0_BASE = pd.Timestamp("2022-12-01")

# --- 三因子 (公开核实值, 见 docstring 来源) ---
COV60_BASE, TARGET = 0.8642, 0.90
SHARE60 = 0.198
VE_MID, VE_LO, VE_HI = 0.164, 0.095, 0.228
UPLIFT = TARGET - COV60_BASE
delta_of = lambda ve: SHARE60 * UPLIFT * ve
D_MID, D_LO, D_HI = delta_of(VE_MID), delta_of(VE_LO), delta_of(VE_HI)
print(f"[三因子] 占比={SHARE60} 提升={UPLIFT:.4f} 效力中值={VE_MID} (CI {VE_LO}-{VE_HI})")
print(f"[delta] 中值={D_MID:.6f} ({D_MID*100:.4f}%) 低界={D_LO:.6f} 高界={D_HI:.6f}")

# --- 基线复核: grid argmin (同 Q1) ---
grid = pd.read_csv(PANEL + r"\m14_h2_grid.csv")
row = grid.loc[grid["loss"].idxmin()]
BETA_M, I0_M = float(row["beta"]), float(row["I0"])
print(f"[基线复核] grid argmin: beta={BETA_M:.4f} I0={I0_M:.4e} loss={row['loss']:.4f}")


def run_seir(I0, S0):
    S, E, I = float(S0), float(I0), float(I0)
    daily = np.empty(N_DAYS)
    for d in range(N_DAYS):
        new_inf = BETA_M * S * I / N_CN
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


base = run_seir(I0_M, N_CN)
q4 = run_seir(I0_M, N_CN * (1 - D_MID))
q4_lo = run_seir(I0_M, N_CN * (1 - D_LO))
q4_hi = run_seir(I0_M, N_CN * (1 - D_HI))
# 复核: 敏感性仅报告
m, m_lo, m_hi = metrics(base, q4), metrics(base, q4_lo), metrics(base, q4_hi)

print("\n== Q4 免疫墙代理 (S0=N x (1-delta_mid), delta=%.6f) ==" % D_MID)
for k, v in m.items():
    print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v:+d}")
print(f"  判定 H4d: |dPeakH|<5% -> {'支持' if abs(m['dPeakH']) < 5 else '意外'}; "
      f"|dCum|<5% -> {'支持' if abs(m['dCum']) < 5 else '意外'}; "
      f"意外判据任一>10% -> {'未触发' if max(abs(m['dPeakH']), abs(m['dCum'])) <= 10 else '触发'}")
print(f"\n[敏感性 D6 仅报告] 效力低界(VE={VE_LO}, delta={D_LO:.6f}): "
      f"dPeak={m_lo['dPeak']:+d} dPeakH={m_lo['dPeakH']:+.4f}% dCum={m_lo['dCum']:+.4f}%")
print(f"[敏感性 D6 仅报告] 效力高界(VE={VE_HI}, delta={D_HI:.6f}): "
      f"dPeak={m_hi['dPeak']:+d} dPeakH={m_hi['dPeakH']:+.4f}% dCum={m_hi['dCum']:+.4f}%")

# --- 落盘 ---
pd.DataFrame({
    "date": pd.date_range(T0_BASE, periods=N_DAYS),
    "base_daily": base, "q4_daily": q4,
}).to_csv(PANEL + r"\m19_cf4_vax_traj.csv", index=False)
pd.DataFrame([
    {"scenario": "base(S0=N)", "delta": 0.0, **metrics(base, base)},
    {"scenario": f"Q4(VE中值{VE_MID})", "delta": D_MID, **m},
    {"scenario": f"Q4敏感性(VE低界{VE_LO}, 仅报告)", "delta": D_LO, **m_lo},
    {"scenario": f"Q4敏感性(VE高界{VE_HI}, 仅报告)", "delta": D_HI, **m_hi},
]).to_csv(PANEL + r"\m19_cf4_metrics.csv", index=False)
print("\n落盘: m19_cf4_vax_traj.csv / m19_cf4_metrics.csv")
