# DeepMicelle

胶束处方性质预测系统。用基团贡献法得到的 Flory-Huggins 参数 (χ) 作为物理先验特征，
结合处方/工艺参数预测载药胶束的关键性能指标，并提供 Streamlit 交互界面。

## 方法

1. **χ 引擎**：由药物与疏水嵌段的溶解度参数差计算 `chi_dc`、`chi_cw`，并导出缔合常数
   `Kam`，作为“能否成胶”的物理判据（`chi_dc < 2` 视为相容）。
2. **特征**：嵌段比、亲/疏水嵌段分子量及其交互项、药物描述符 (MolWt/LogP/TPSA)、
   χ 三件套、制备方法、嵌段家族。
3. **模型**：XGBoost 回归，`Size` 在 log1p 空间训练。
4. **适用域**：`chi_dc > 2` 提示域外；未知嵌段家族区间放大；EE 仅作趋势参考。

## 预测目标

|目标|模型|留一验证 MAE|
|---|---|---|
|包封率 EE %|`models/v6_EE.pkl`|4.83|
|载药量 DL %|`models/v6_DL.pkl`|0.59|
|粒径 Size nm|`models/v6_Size.pkl`|14.85|
|载药 CMC (log)|`models/v4_logCMC.pkl`|0.379（5折CV R² 0.489）|

Zeta 电位、粒径变化率、药物泄漏率、PDI 变化因数据量不足（n<10）暂不提供预测。

## 运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

界面按“介质条件 → 处方 → 水化参数 → 药物”四步录入，输出预测值及 90% 区间。


## 目录

- `train_v6.py`：训练脚本，产出 `models/v6_*.pkl` 与 `reports/v6_loo.json`
- `predict.py`：预测接口（含区间与适用域提示）
- `calibration.py`：可选的单锚点迁移校准，默认关闭
- `active_learn.py`：主动学习，生成高信息量实验推荐清单
- `reports/`：验证指标与留一结果
- `data/cleaned/`：清洗后的训练数据

## 数据来源

`data/cleaned/formulation_train_v3final_new.xlsx` 共 578 行，来自文献整理与本实验室
实验（`data_source = user_exp`）。`cmc_train_v2.xlsx` 为表面活性剂本体 CMC 数据，
与载药胶束 CMC 口径不同，仅作参考。
