# -*- coding: utf-8 -*-
"""M19 Q3: 美国更强早期 NPI (预注册 M19-反事实四问-预注册.md v1.0 §3 机械执行 + 窗口修正).

基线: M9 no_feedback (seir_fits.csv: beta0=0.2408, infl=2.0, kappa=0.1855, b=0 恒 beta),
     52 周面板窗 (2020-03-30 起), 日步长 Euler SEIR (sigma=1/3, gamma=1/5), n_pop=3.31e8
窗口修正 (预注册 §2.2 授权执行时锁定, 汇报补记 D7/D8):
     预注册原折减窗 2020-02-17 起在面板窗外 (首周 2020-03-30) -> 修正为窗内前 5 周
     (2020-03-30 .. 2020-05-02, d0..d34) beta x 0.6; 「更早行动」语义收窄为「窗内更强管制」
     多轮波动在 no_feedback (b=0) 内不可表达 -> H4c 后半 (后续波/免疫债) 不可判定, D8 登记
锁定: kappa/w0/infl/beta0 全锁基线值 (kappa 不重拟合, 同一观测尺度); 面板窗/初值锁死
复核修正 (本轮): seir_fits.csv 的 beta0/kappa 为 4 位舍入值, 直接复算周差 528;
     改为 (A) BETA0_GRID 唯一反解全精度 beta0 (round4==0.2408) + (B) kappa 按原解析公式重算
     (round4==0.1855 校验), 基线周病例 vs seir_traj.csv 逐位一致; Q3 另以原引擎
     simulate(eps=1.6, b=前5周=1) 等价参数化交叉复核 (数学上 = 前35天 beta x 0.6)
指标: dPeak / dPeakH / dCum / rho (日新增序列, 日峰主口径)
判定: H4c 可判定部分 (峰高降幅 15-50%; 峰时推迟 0-3 周)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"e:\ai agent\research\covid-us-pilot")
from panel_prep import N_TEST, load_df  # noqa: E402
import seir_track2 as st  # noqa: E402

PANEL = r"e:\ai agent\research\covid-us-pilot\data\panel"
SIGMA, GAMMA = 1 / 3, 1 / 5
N_POP = 331_000_000.0
N_WEEKS_REDUCE = 5  # 修正折减窗: 前 5 周
REDUCE = 0.6

df, burden = load_df()
n_weeks = len(df)
n_days = n_weeks * 7
w0 = float(burden.set_index("week_start").loc[df["week_start"].iloc[0], "cases"])
fit = pd.read_csv(PANEL + r"\seir_fits.csv").set_index("config").loc["no_feedback"]
INFL, LAG = float(fit["infl"]), int(fit["lag_weeks"])
obs_cases = df[["week_start"]].merge(burden[["week_start", "cases"]], on="week_start",
                                     how="left")["cases"].to_numpy()
print(f"[锁定] n_weeks={n_weeks} 首周={df['week_start'].iloc[0].date()} "
      f"w0={w0:.0f} infl={INFL} lag={LAG}")

# (A) 网格唯一反解全精度 beta0
cand = [float(v) for v in st.BETA0_GRID if round(float(v), 4) == float(fit["beta0"])]
if len(cand) != 1:
    print(f"[反解] beta0 候选 {len(cand)} 个, 停")
    sys.exit(1)
BETA0 = cand[0]
# (B) kappa 按原解析公式重算 (m_tr = mr[:n_train-lag], o_tr = obs[lag:n_train])
n_train = n_weeks - N_TEST
m_week0 = st.simulate(np.zeros(n_weeks), BETA0, 0.0, INFL, w0, N_POP, n_weeks)
mr0 = m_week0 / INFL
KAPPA = max(0.0, float(np.sum(mr0[:n_train - LAG] * obs_cases[LAG:n_train])
                       / np.sum(mr0[:n_train - LAG] ** 2)))
print(f"[精度恢复] beta0={BETA0:.10f} (round4={round(BETA0,4)} 唯一匹配) "
      f"kappa={KAPPA:.10f} (round4={round(KAPPA,4)} "
      f"{'==csv 一致' if round(KAPPA,4)==float(fit['kappa']) else '与 csv 不一致, 停'})")
if round(KAPPA, 4) != float(fit["kappa"]):
    sys.exit(1)


def run(beta_factor_fn):
    S, E, I = N_POP, INFL * w0 * 3 / 7, INFL * w0 * 5 / 7
    daily = np.empty(n_days)
    for d in range(n_days):
        beta = BETA0 * beta_factor_fn(d)
        new_inf = beta * S * I / N_POP
        e2i, i2r = SIGMA * E, GAMMA * I
        S -= new_inf
        E += new_inf - e2i
        I += e2i - i2r
        daily[d] = new_inf
    return daily


base = run(lambda d: 1.0)
q3 = run(lambda d: REDUCE if d < N_WEEKS_REDUCE * 7 else 1.0)

# 复核A: 基线周病例 (kappa*weekly/infl) vs seir_traj.csv pred_no_feedback 逐位一致
traj_ref = pd.read_csv(PANEL + r"\seir_traj.csv", parse_dates=["week_start"])
ref = traj_ref["pred_no_feedback"].to_numpy()[:n_weeks]
wk_base = np.round(KAPPA * base.reshape(n_weeks, 7).sum(axis=1) / INFL)
diff_a = np.abs(wk_base - ref).max()
print(f"[复核A] 基线周病例 vs seir_traj.csv 最大差 = {diff_a:.1f} "
      f"-> {'一致' if diff_a == 0 else '不一致, 停'}")
if diff_a != 0:
    sys.exit(1)
# 复核B: Q3 以原引擎等价参数化 (eps=1.6, b=前5周=1 -> beta*max(0.05,1-1.6/4)=beta*0.6) 交叉复核
b_q3_week = np.concatenate([np.ones(N_WEEKS_REDUCE), np.zeros(n_weeks - N_WEEKS_REDUCE)])
q3_ref_week = st.simulate(b_q3_week, BETA0, (1 - REDUCE) * 4, INFL, w0, N_POP, n_weeks)
diff_b = np.abs(q3.reshape(n_weeks, 7).sum(axis=1) - q3_ref_week).max()
print(f"[复核B] Q3 周病例 vs 原引擎等价参数化 (eps={ (1-REDUCE)*4 }) 最大差 = {diff_b:.2e} "
      f"-> {'一致' if diff_b < 1e-6 else '不一致, 停'}")
if diff_b >= 1e-6:
    sys.exit(1)


def metrics(d_base, d_cf):
    pk_b, pk_c = int(np.argmax(d_base)), int(np.argmax(d_cf))
    return dict(
        dPeak=pk_c - pk_b,
        dPeakH=(d_cf[pk_c] / d_base[pk_b] - 1) * 100,
        dCum=(d_cf.sum() / d_base.sum() - 1) * 100,
        rho=float(pd.Series(d_base).corr(pd.Series(d_cf), method="spearman")),
    )


m = metrics(base, q3)
print("\n== Q3 更强早期 NPI (前 5 周 beta x 0.6, 窗内修正版) ==")
print(f"  dPeak: {m['dPeak']:+d} 天 (基线日峰 d{int(np.argmax(base))}, 折减日峰 d{int(np.argmax(q3))})")
print(f"  dPeakH: {m['dPeakH']:+.2f}%")
print(f"  dCum: {m['dCum']:+.2f}%")
print(f"  rho: {m['rho']:+.4f}")
ok_h = 15 <= -m["dPeakH"] <= 50
ok_t = 0 <= m["dPeak"] <= 21
print(f"\n判定 H4c (可判定部分): 峰高降幅 15-50% -> {'支持' if ok_h else '意外'}; "
      f"峰时推迟 0-3 周 -> {'支持' if ok_t else '意外'}")
print("  (后续波/免疫债: no_feedback 无多波机制, D8 不可判定)")

pd.DataFrame({
    "date": pd.date_range(df["week_start"].iloc[0], periods=n_days),
    "base_daily": base, "q3_daily": q3,
}).to_csv(PANEL + r"\m19_cf3_us_early_npi_traj.csv", index=False)
pd.DataFrame([{"scenario": "base(no_feedback)", **metrics(base, base)},
              {"scenario": "Q3(5wk beta x0.6)", **m}]).to_csv(PANEL + r"\m19_cf3_metrics.csv", index=False)
print("\n落盘: m19_cf3_us_early_npi_traj.csv / m19_cf3_metrics.csv")
