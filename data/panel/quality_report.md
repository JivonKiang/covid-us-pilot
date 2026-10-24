# us_state_week 数据质量报告

- 生成时间：2026-10-22 02:21
- 维度：51 辖区 × 137 周 = 6987 行
- 周定义：周一锚点，2020-03-02 → 2022-10-10（最后一周截至今可用数据，Mobility 止于 2022-10-15）
- 州人口合计：328.7 百万（含 DC）

## 列覆盖率

| 列组 | 覆盖率 |
|---|---|
| 负担 (new_cases) | new_cases 100%, new_deaths 100% |
| 率 (per100k) | new_cases_per100k 100%, new_deaths_per100k 100% |
| 政策 (11 列均值) | pol_stringency 100%, pol_containment 100%, pol_govresp 100%, pol_c1_school 100%, pol_c2_workplace 100%, pol_c3_events 100%, pol_c4_gather 100%, pol_c6_stayhome 100%, pol_c7_internal 100%, pol_h6_masks 100%, pol_e1_income 100% |
| Mobility (6 列均值) | mob_retail 100%, mob_grocery 100%, mob_parks 99%, mob_transit 100%, mob_workplaces 100%, mob_residential 100% |
| YouGov 州样本数 | n_yougov_state 35% |

## 负担数据

- 周新增为负的格子：确诊 6 / 6987，死亡 21（JHU 数据回溯修正，未改数）
- 2022-03 后 JHU 确诊报告趋弱（居家自测不再上报），解读末期确诊需谨慎

## 政策数据（OxCGRT 州级，原始日度缺失率）

| 列 | 缺失率 |
|---|---|
| pol_stringency | 1.9% |
| pol_containment | 1.9% |
| pol_govresp | 1.9% |
| pol_c1_school | 0.0% |
| pol_c2_workplace | 0.0% |
| pol_c3_events | 0.0% |
| pol_c4_gather | 0.0% |
| pol_c6_stayhome | 0.0% |
| pol_c7_internal | 0.0% |
| pol_h6_masks | 0.0% |
| pol_e1_income | 0.0% |

## 行为数据

- Mobility 州级行缺失率最高：parks_percent_change_from_baseline 1.1%
- YouGov 国家级周样本量：中位 811，最小 1，最大 1879
- 样本量 <50 的周（9 个，散布于 2020-06~2021-07，与 YouGov 样本轮换相关，见数据边界说明）：
  - 2020-06-15: n=7
  - 2020-07-13: n=41
  - 2020-07-27: n=37
  - 2020-08-10: n=3
  - 2020-08-24: n=1
  - 2020-09-07: n=33
  - 2021-03-01: n=46
  - 2021-06-21: n=29
  - 2021-07-05: n=44
- 首周（2020-03-30）样本：口罩题 n=1879，洗手题 n=1879，总样本 n=1879
- YouGov 州×周平均样本量：中位 9.6，仅作种子参考（校准主口径用国家级）

## 抽样检查（加州 2020-04 与 2021-04 各一周）

     state week_start  new_cases  new_deaths  new_cases_per100k  new_deaths_per100k  population  pol_stringency  pol_containment  pol_govresp  pol_c1_school  pol_c2_workplace  pol_c3_events  pol_c4_gather  pol_c6_stayhome  pol_c7_internal  pol_h6_masks  pol_e1_income  mob_retail  mob_grocery  mob_parks  mob_transit  mob_workplaces  mob_residential  n_yougov_state
California 2020-04-06     7715.0       289.0          19.525604            0.731419  39512223.0           82.41            70.54        72.66            3.0               3.0            2.0            4.0              2.0              2.0           3.0            2.0  -49.428571   -22.714286 -48.142857   -58.142857      -51.428571        22.571429           120.0
California 2021-04-05    24481.0       691.0          61.958043            1.748826  39512223.0           59.72            64.70        64.43            2.0               2.0            2.0            4.0              2.0              1.0           3.0            1.0  -18.428571    -9.000000  -0.428571   -36.857143      -33.285714         8.714286            62.0

