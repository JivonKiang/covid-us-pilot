# -*- coding: utf-8 -*-
"""临时探针: base_ar1 在锚周的预测 vs 真值, 定位 overall=2.74 异常."""
import llm_behavior_layer_china as L

df, n_obs = L.build_df()
n = len(df)
test_idx = list(range(n - L.N_TEST, n))
test = df.iloc[test_idx].set_index("week_start")
anchors = test.index[test["n_total"] >= L.MIN_N]
bp = L.base_ar1_pred(df, n - L.N_TEST)
print("锚周 base_ar1 预测 vs 真值:")
for a in anchors:
    row_p = bp.loc[a]
    row_o = df.loc[df["week_start"] == a, L.BEHAV6].iloc[0]
    print(str(a.date()),
          {L.SHORT[b]: (round(float(row_p[b]), 2), round(float(row_o[b]), 2)) for b in L.BEHAV6})
print("训练段(0-36)行为均值:", round(float(df[L.BEHAV6][:37].mean().mean()), 2),
      "锚周真值均值:", round(float(df.loc[df["week_start"].isin(anchors), L.BEHAV6].mean().mean()), 2))
print("t=30..40 复合均值预测:", [round(float(bp.iloc[i][L.BEHAV6].mean()), 2) for i in range(30, 41)])
print("训练段各行为均值:", {L.SHORT[b]: round(float(df[b][:37].mean()), 2) for b in L.BEHAV6})
