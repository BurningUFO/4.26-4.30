# B 组建模计算区

本目录用于完成 `方案.md` 中阶段 4-6 的建模与结果输出。

## 当前已完成

### 阶段 4：第 2 问寿命预测

- `outputs/B1_寿命预测结果.xlsx`
- `outputs/B1_寿命预测模型说明.md`
- `figures_B/fig_B1_lifetime_prediction.png`
- `figures_B/fig_B1_remaining_life_bar.png`

### 阶段 5：第 3 问维护策略优化

- `outputs/B2_维护策略优化结果.xlsx`
- `outputs/B2_维护优化模型说明.md`
- `figures_B/fig_B2_maintenance_strategy_comparison.png`
- `figures_B/fig_B2_policy_timeline.png`

### 阶段 6：第 4 问成本敏感性分析

- `outputs/B3_成本敏感性分析结果.xlsx`
- `outputs/B3_敏感性分析说明.md`
- `figures_B/fig_B3_cost_sensitivity.png`
- `figures_B/fig_B3_sensitivity_tornado.png`

## 附加交付

- `outputs/B_to_C_第2-4问论文素材交接说明.md`
- `outputs/B_终稿前检查清单.md`

## 运行方式

```bash
python3 04_代码/B_prediction_model.py
python3 04_代码/B_optimization.py
python3 04_代码/B_sensitivity_analysis.py
```
