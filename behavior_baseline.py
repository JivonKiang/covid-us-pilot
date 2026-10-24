# -*- coding: utf-8 -*-
"""M3-6 行为层 · 固定规则基线（W1 可拟合性 / W3 保留集外推稳定）
两个变体：
- base     ：行为 ~ 政策指数(lag1) + log负担(lag1) + 周序趋势（方案口径主判定）
- base_ar1 ：base + 该行为自身 lag1（行为惯性对照，回答「崩是结构问题还是特征不够」）
- 训练期：前 40 个可用周；保留集：最后 12 周（约 2021-04 -> 2021-10，Delta 波）
- 按周样本量加权（均值归一）；特征按各自训练有效行标准化
W1 判定：多数指标训练 R² >= 0.5（可拟合）
W3 判定：保留 RMSE / 训练 RMSE <= 2（外推不崩）
"""
import numpy as np
import pandas as pd

ROOT = r"e:\ai agent\research\covid-us-pilot"
PANEL = ROOT + r"\data\panel"
LAMBDA = 1.0

nat = pd.read_csv(PANEL + r"\us_national_week.csv", parse_dates=["week_start"])
panel = pd.read_csv(PANEL + r"\us_state_week.csv", parse_dates=["week_start"])

BEHAV = ["yougov_mask_freq", "yougov_hands_freq", "yougov_stay_home_freq",
         "yougov_avoid_restaurants_freq", "yougov_avoid_crowds_freq",
         "yougov_avoid_shops_freq", "yougov_avoid_work_freq"]

# 国家级驱动（与 EDA 同口径）
pop_sum = panel.groupby("week_start").apply(
    lambda g: np.average(g["pol_stringency"], weights=g["population"])
    if g["pol_stringency"].notna().all() else g["pol_stringency"].mean(),
    include_groups=False)
burden = panel.groupby("week_start").agg(cases=("new_cases", "sum"),
                                         pop=("population", "sum")).reset_index()
burden["per100k"] = burden["cases"] / burden["pop"] * 1e5
drv = pd.DataFrame({"week_start": pop_sum.index,
                    "stringency": pop_sum.to_numpy()}).merge(
    burden[["week_start", "per100k"]], on="week_start")
drv["log_burden"] = np.log10(drv["per100k"].clip(lower=0.01))

df = nat.merge(drv, on="week_start", how="left")
df = df[df["n_total"] >= 50].reset_index(drop=True)
df["t"] = np.arange(len(df))

w = df["n_total"].to_numpy(float)
w = w / w.mean()  # 均值归一（和=n_train），保留相对样本量差异且量级匹配 λ

n_train = len(df) - 12
tr, te = np.arange(n_train), np.arange(n_train, len(df))

# 公共特征：lag1（上周政策/负担驱动本周行为）
pol_lag = df["stringency"].shift(1).bfill().to_numpy(float)
bur_lag = df["log_burden"].shift(1).bfill().to_numpy(float)
t_feat = df["t"].to_numpy(float)


def ridge(X, y, wt, lam):
    A = X.T @ (wt[:, None] * X) + lam * np.eye(X.shape[1])
    A[0, 0] -= lam  # 截距在第 0 列，不正则
    return np.linalg.solve(A, X.T @ (wt * y))


def rmse(y, p, wt):
    return float(np.sqrt(np.average((y - p) ** 2, weights=wt)))


def r2(y, p, wt):
    ss = np.average((y - np.average(y, weights=wt)) ** 2, weights=wt)
    return 1 - np.average((y - p) ** 2, weights=wt) / ss if ss > 0 else np.nan


def fit_variant(use_ar):
    rows, pred_out = [], pd.DataFrame({"week_start": df["week_start"]})
    for b in BEHAV:
        y = df[b].to_numpy(float)
        cols = [pol_lag, bur_lag, t_feat]
        if use_ar:
            cols.append(df[b].shift(1).to_numpy(float))  # 行为自身惯性
        Xr = np.column_stack(cols)
        ok = np.isfinite(y) & np.isfinite(Xr).all(1)  # 逐指标有效样本过滤
        trv, tev = tr[ok[tr]], te[ok[te]]
        row = {"behavior": b.replace("yougov_", "").replace("_freq", ""),
               "n_train": int(ok[tr].sum()), "n_test": int(ok[te].sum()),
               "train_R2": np.nan, "train_RMSE": np.nan,
               "test_R2": np.nan, "test_RMSE": np.nan, "RMSE_ratio": np.nan}
        if len(trv) >= 5:
            mu, sd = Xr[trv].mean(0), Xr[trv].std(0)
            sd[sd == 0] = 1.0
            Xs = (Xr - mu) / sd
            Xs = np.column_stack([np.ones(len(Xs)), Xs])
            beta = ridge(Xs[trv], y[trv], w[trv], LAMBDA)
            ptr = Xs[trv] @ beta
            row["train_R2"] = round(r2(y[trv], ptr, w[trv]), 3)
            row["train_RMSE"] = round(rmse(y[trv], ptr, w[trv]), 3)
            pred_out[b] = y
            pred_out[b + "_pred"] = Xs @ beta
            if len(tev) > 0:
                pte = Xs[tev] @ beta
                rtr = rmse(y[trv], ptr, w[trv])
                row["test_R2"] = round(r2(y[tev], pte, w[tev]), 3)
                row["test_RMSE"] = round(rmse(y[tev], pte, w[tev]), 3)
                row["RMSE_ratio"] = round(
                    rmse(y[tev], pte, w[tev]) / rtr, 2) if rtr > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), pred_out


res_base, pred_base = fit_variant(False)
res_ar, pred_ar = fit_variant(True)

for tag, res in [("base", res_base), ("base_ar1", res_ar)]:
    print(f"== {tag} ==")
    print(res.to_string(index=False))
w1_d, w3_d = int(res_base["train_R2"].notna().sum()), int(res_base["RMSE_ratio"].notna().sum())
w1 = int((res_base["train_R2"] >= 0.5).sum())
w3 = int((res_base["RMSE_ratio"] <= 2).sum())
w1a_d, w3a_d = int(res_ar["train_R2"].notna().sum()), int(res_ar["RMSE_ratio"].notna().sum())
w1a, w3a = int((res_ar["train_R2"] >= 0.5).sum()), int((res_ar["RMSE_ratio"] <= 2).sum())
print(f"\nbase     W1: {w1}/{w1_d}   W3: {w3}/{w3_d}")
print(f"base_ar1 W1: {w1a}/{w1a_d}   W3: {w3a}/{w3a_d}")

out_md = PANEL + r"\behavior_baseline.md"
pred_base.to_csv(PANEL + r"\behavior_baseline_pred.csv", index=False)
pred_ar.to_csv(PANEL + r"\behavior_baseline_ar1_pred.csv", index=False)
with open(out_md, "w", encoding="utf-8") as f:
    f.write("# 固定规则行为基线（M3–6 效度预检）\n\n"
            f"- 模型：岭回归 λ={LAMBDA}，按周样本量加权（均值归一），"
            "特征按各自训练有效行标准化；逐指标过滤无效周\n"
            f"- 切分：前 {n_train} 个可用周训练 / 最后 12 周保留"
            f"（{df['week_start'].iloc[n_train].date()} -> {df['week_start'].iloc[-1].date()}，Delta 波）\n\n"
            "## 变体 base（政策lag1 + log负担lag1 + 周序趋势；方案口径主判定）\n\n"
            + res_base.to_string(index=False) + "\n\n"
            "## 变体 base_ar1（base + 该行为自身 lag1，行为惯性对照）\n\n"
            + res_ar.to_string(index=False) + "\n\n"
            f"- **W1 判定（base）**：训练 R²≥0.5 的指标 {w1}/{w1_d}\n"
            f"- **W3 判定（base）**：保留集 RMSE 恶化 ≤2× 的指标 {w3}/{w3_d}\n"
            f"- **对照（base_ar1）**：W1 {w1a}/{w1a_d}；W3 {w3a}/{w3a_d}\n"
            "- 注：保留集恰为 Delta 变异期，属「分布漂移下的最严格外推」；"
            "n_test=0 的指标（问卷停题，如 hands）不参与 W3 计数。\n")
print(f"-> {out_md}\n-> behavior_baseline_pred.csv / behavior_baseline_ar1_pred.csv")
