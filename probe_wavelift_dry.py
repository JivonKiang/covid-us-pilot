# -*- coding: utf-8 -*-
"""dry-run: 验证 prompt 生成 + base_ar1 数值合理性 (不调 LLM)."""
import llm_behavior_wavelift as m

units = m.build_units()
print("--- full prompt (national, W4) ---")
print(m.build_prompt(units["national"], 3, "full", "national"))
print()
print("--- no_inertia prompt tail (Guangdong, W8) ---")
print(m.build_prompt(units["Guangdong"], 7, "no_inertia", "Guangdong")[-400:])
print()
for u in ["national", "Beijing"]:
    du = units[u]
    pred = m.base_ar1_pred(du, use_t=True)
    pred_nt = m.base_ar1_pred(du, use_t=False)
    test = du.iloc[3:8].set_index("week_start")
    pers = du.set_index("week_start")[m.TARGETS].shift(1).iloc[3:8]
    print(f"--- {u}: truth vs base_ar1 vs base_ar1_notrend vs persistence (W4-W8) ---")
    cmp = (test[m.TARGETS].round(2).join(pred.round(2), rsuffix="_ar1")
           .join(pred_nt.round(2), rsuffix="_nt").join(pers, rsuffix="_pers"))
    print(cmp.to_string())
    for t in m.TARGETS:
        print(f"  {t}: rmse_ar1={m.rmse(test[t], pred.loc[test.index, t]):.3f} "
              f"rmse_notrend={m.rmse(test[t], pred_nt.loc[test.index, t]):.3f} "
              f"rmse_pers={m.rmse(test[t], pers[t]):.3f}")
