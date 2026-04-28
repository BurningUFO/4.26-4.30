# A 侧图表说明

本目录用于存放成员 A 在第 1 问分析中生成的论文可用图表。

## 当前图表清单

- `fig_01_all_filters_time_series.png`
  - 作用：展示 10 台过滤器的日均透水率时间序列总体变化。
  - 使用建议：信息量较大，适合作为总览图或附录图，不建议作为正文唯一趋势图。
- `fig_01a_filter_time_series_facets.png`
  - 作用：按过滤器拆分为小多图，并叠加中维护、大维护节点。
  - 使用建议：优先作为正文主趋势图，用于展示单台设备衰减-维护-恢复过程。
- `fig_01b_normalized_permeability_trend.png`
  - 作用：将每台过滤器按自身 90 分位透水率标准化为高位基准 100，展示总体相对变化。
  - 使用建议：作为 `fig_01a` 的补充图，用于说明去除设备量级差异后的共同波动趋势。
- `fig_02_filter_trend_lines.png`
  - 作用：展示各过滤器在维护周期内的日均透水率局部线性下降趋势。
- `fig_03_monthly_average_permeability.png`
  - 作用：展示总体月平均透水率变化，用于季节性判断。
- `fig_04_seasonal_average_permeability.png`
  - 作用：展示四季平均透水率差异。
- `fig_05_maintenance_effect_boxplot.png`
  - 作用：比较中维护和大维护的恢复量分布。
- `fig_06_before_after_maintenance.png`
  - 作用：比较维护前最近透水率、维护后最近透水率和维护后 7/15 天均值。
- `fig_07_filter_decline_rate.png`
  - 作用：比较各过滤器维护周期年化下降量，单位为透水率值/年。
- `fig_08_jump_points_and_maintenance.png`
  - 作用：展示日尺度跳变点与维护记录的邻近关系，并比较维护恢复量与维护后 3 天跳升幅度。
  - 使用建议：用于回应异常跳变和数据联动分析，建议放在维护效果分析之后。

## 同步规则

- 本目录中的 10 张图会同步复制到 `03_论文_C/图表汇总/`。
- 图表由 `04_代码/A_data_process.py` 自动生成，不建议手工覆盖。

## 复现方式

- 运行命令：`python 04_代码/A_data_process.py`
- 生成范围：阶段 1、阶段 2、阶段 3 全部结果，其中本目录会重生成 10 张图。
