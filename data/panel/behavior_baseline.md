# 固定规则行为基线（M3–6 效度预检）

- 模型：岭回归 λ=1.0，按周样本量加权（均值归一），特征按各自训练有效行标准化；逐指标过滤无效周
- 切分：前 40 个可用周训练 / 最后 12 周保留（2021-07-26 -> 2021-10-11，Delta 波）

## 变体 base（政策lag1 + log负担lag1 + 周序趋势；方案口径主判定）

         behavior  n_train  n_test  train_R2  train_RMSE  test_R2  test_RMSE  RMSE_ratio
             mask       40      12     0.433       0.463  -59.910      1.652        3.57
            hands       21       0     0.940       0.041      NaN        NaN         NaN
        stay_home       40      12     0.939       0.092   -2.265      0.225        2.46
avoid_restaurants       40      12     0.905       0.117   -7.659      0.388        3.31
     avoid_crowds       40      12     0.880       0.103    0.040      0.106        1.03
      avoid_shops       40      12     0.902       0.123   -1.326      0.219        1.78
       avoid_work       40      12     0.916       0.117   -0.102      0.168        1.43

## 变体 base_ar1（base + 该行为自身 lag1，行为惯性对照）

         behavior  n_train  n_test  train_R2  train_RMSE  test_R2  test_RMSE  RMSE_ratio
             mask       39      12     0.870       0.160   -7.800      0.628        3.93
            hands       20       0     0.926       0.044      NaN        NaN         NaN
        stay_home       39      12     0.941       0.086   -0.716      0.163        1.91
avoid_restaurants       39      12     0.906       0.110   -3.509      0.280        2.56
     avoid_crowds       39      12     0.867       0.106   -0.004      0.109        1.03
      avoid_shops       39      12     0.905       0.114   -0.022      0.145        1.27
       avoid_work       39      12     0.926       0.103    0.139      0.148        1.44

- **W1 判定（base）**：训练 R²≥0.5 的指标 6/7
- **W3 判定（base）**：保留集 RMSE 恶化 ≤2× 的指标 3/6
- **对照（base_ar1）**：W1 7/7；W3 4/6
- 注：保留集恰为 Delta 变异期，属「分布漂移下的最严格外推」；n_test=0 的指标（问卷停题，如 hands）不参与 W3 计数。
