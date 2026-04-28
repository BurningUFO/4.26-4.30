"""
B_sensitivity_analysis.py

Stage 6 in `方案.md`: cost sensitivity analysis for Question 4.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from B_prediction_model import B_OUTPUTS, B_FIGURES, C_FIGURES, configure_plot_style, ensure_dirs


PURCHASE_COST = 300.0
MIDDLE_MAINT_COST = 3.0
BIG_MAINT_COST = 12.0
PERTURBATIONS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
RESULT_XLSX = B_OUTPUTS / "B3_成本敏感性分析结果.xlsx"
NOTE_MD = B_OUTPUTS / "B3_敏感性分析说明.md"
FIG_COST_SENS = B_FIGURES / "fig_B3_cost_sensitivity.png"
FIG_TORNADO = B_FIGURES / "fig_B3_sensitivity_tornado.png"


def load_b2_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = B_OUTPUTS / "B2_维护策略优化结果.xlsx"
    strategy_summary = pd.read_excel(path, sheet_name="strategy_summary")
    filter_results = pd.read_excel(path, sheet_name="filter_strategy_results")
    policy_definition = pd.read_excel(path, sheet_name="policy_definition")
    return strategy_summary, filter_results, policy_definition


def build_strategy_base_table(
    strategy_summary: pd.DataFrame,
    filter_results: pd.DataFrame,
) -> pd.DataFrame:
    agg = (
        filter_results.groupby(["strategy_code", "strategy_label"], as_index=False)[
            ["middle_count", "big_count", "remaining_life_years", "lifecycle_cost"]
        ]
        .sum()
        .rename(
            columns={
                "middle_count": "total_middle_count",
                "big_count": "total_big_count",
                "remaining_life_years": "total_life_years",
                "lifecycle_cost": "baseline_total_cost",
            }
        )
    )
    merged = agg.merge(
        strategy_summary[
            [
                "strategy_code",
                "fleet_annual_cost",
                "avg_service_per",
                "avg_remaining_life_years",
                "service_improvement_vs_current",
                "life_change_vs_current",
                "recommended_flag",
            ]
        ],
        on="strategy_code",
        how="left",
    )
    merged["feasible_flag"] = (
        (merged["service_improvement_vs_current"] >= 0.10)
        & (merged["life_change_vs_current"] >= -0.05)
    ).astype(int)
    return merged


def compute_fleet_annual_cost(
    total_middle_count: float,
    total_big_count: float,
    total_life_years: float,
    purchase_cost: float,
    middle_cost: float,
    big_cost: float,
) -> float:
    total_cost = 10 * purchase_cost + total_middle_count * middle_cost + total_big_count * big_cost
    return total_cost / total_life_years


def build_scenario_results(base_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter in ["purchase", "middle", "big"]:
        for delta in PERTURBATIONS:
            purchase_cost = PURCHASE_COST * (1 + delta) if parameter == "purchase" else PURCHASE_COST
            middle_cost = MIDDLE_MAINT_COST * (1 + delta) if parameter == "middle" else MIDDLE_MAINT_COST
            big_cost = BIG_MAINT_COST * (1 + delta) if parameter == "big" else BIG_MAINT_COST

            tmp = base_table.copy()
            tmp["scenario_fleet_annual_cost"] = tmp.apply(
                lambda row: compute_fleet_annual_cost(
                    row["total_middle_count"],
                    row["total_big_count"],
                    row["total_life_years"],
                    purchase_cost,
                    middle_cost,
                    big_cost,
                ),
                axis=1,
            )

            feasible_tmp = tmp[tmp["feasible_flag"] == 1].copy()
            best_cost = feasible_tmp["scenario_fleet_annual_cost"].min()
            best_rows = feasible_tmp.loc[
                np.isclose(feasible_tmp["scenario_fleet_annual_cost"], best_cost, atol=1e-8)
            ]
            best_codes = ",".join(best_rows["strategy_code"].tolist())
            best_labels = "、".join(best_rows["strategy_label"].tolist())

            for row in tmp.itertuples(index=False):
                rows.append(
                    {
                        "parameter": parameter,
                        "delta_ratio": delta,
                        "purchase_cost": purchase_cost,
                        "middle_cost": middle_cost,
                        "big_cost": big_cost,
                        "strategy_code": row.strategy_code,
                        "strategy_label": row.strategy_label,
                        "feasible_flag": row.feasible_flag,
                        "recommended_flag": row.recommended_flag,
                        "scenario_fleet_annual_cost": row.scenario_fleet_annual_cost,
                        "best_strategy_codes": best_codes,
                        "best_strategy_labels": best_labels,
                        "best_includes_optimized": int("optimized_policy" in best_codes.split(",")),
                    }
                )
    return pd.DataFrame(rows)


def build_switch_summary(scenario_results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for parameter, group in scenario_results.groupby("parameter"):
        scenario_level = (
            group[["delta_ratio", "best_strategy_codes", "best_strategy_labels", "best_includes_optimized"]]
            .drop_duplicates()
            .sort_values("delta_ratio")
        )
        optimized_count = int(scenario_level["best_includes_optimized"].sum())
        summary_rows.append(
            {
                "parameter": parameter,
                "tested_levels": len(scenario_level),
                "optimized_best_levels": optimized_count,
                "optimized_best_ratio": optimized_count / len(scenario_level),
                "best_strategy_labels_sequence": " | ".join(
                    f"{int(delta * 100):+d}%:{labels}"
                    for delta, labels in zip(
                        scenario_level["delta_ratio"],
                        scenario_level["best_strategy_labels"],
                    )
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def build_optimized_parameter_impact(
    scenario_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    optimized = scenario_results[scenario_results["strategy_code"] == "optimized_policy"].copy()
    baseline = optimized.loc[np.isclose(optimized["delta_ratio"], 0.0), ["parameter", "scenario_fleet_annual_cost"]]
    baseline = baseline.rename(columns={"scenario_fleet_annual_cost": "baseline_cost"})

    impact_rows = []
    line_rows = []
    for parameter, group in optimized.groupby("parameter"):
        group = group.sort_values("delta_ratio").copy()
        base_cost = float(group.loc[np.isclose(group["delta_ratio"], 0.0), "scenario_fleet_annual_cost"].iloc[0])
        low_cost = float(group.loc[np.isclose(group["delta_ratio"], -0.3), "scenario_fleet_annual_cost"].iloc[0])
        high_cost = float(group.loc[np.isclose(group["delta_ratio"], 0.3), "scenario_fleet_annual_cost"].iloc[0])
        impact_rows.append(
            {
                "parameter": parameter,
                "baseline_cost": base_cost,
                "cost_at_minus_30pct": low_cost,
                "cost_at_plus_30pct": high_cost,
                "negative_change": low_cost - base_cost,
                "positive_change": high_cost - base_cost,
                "swing_range": high_cost - low_cost,
                "swing_ratio_vs_baseline": (high_cost - low_cost) / base_cost,
            }
        )
        line_rows.append(group[["parameter", "delta_ratio", "scenario_fleet_annual_cost"]])

    impact_df = pd.DataFrame(impact_rows).sort_values("swing_range", ascending=False).reset_index(drop=True)
    line_df = pd.concat(line_rows, ignore_index=True)
    return impact_df, line_df


def plot_cost_sensitivity(line_df: pd.DataFrame) -> None:
    color_map = {
        "purchase": "#577590",
        "middle": "#2a9d8f",
        "big": "#d62828",
    }
    label_map = {
        "purchase": "购买成本",
        "middle": "中维护成本",
        "big": "大维护成本",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for parameter, group in line_df.groupby("parameter"):
        group = group.sort_values("delta_ratio")
        ax.plot(
            group["delta_ratio"] * 100,
            group["scenario_fleet_annual_cost"],
            marker="o",
            linewidth=2,
            color=color_map[parameter],
            label=label_map[parameter],
        )

    ax.set_title("优化策略对成本参数扰动的敏感性")
    ax.set_xlabel("参数扰动幅度 / %")
    ax.set_ylabel("舰队年均成本 / 万元·年$^{-1}$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_COST_SENS)
    plt.close(fig)


def plot_tornado(impact_df: pd.DataFrame) -> None:
    label_map = {
        "purchase": "购买成本",
        "middle": "中维护成本",
        "big": "大维护成本",
    }
    df = impact_df.copy()
    df["label"] = df["parameter"].map(label_map)
    df = df.sort_values("swing_range", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(df["label"], df["negative_change"], color="#2a9d8f", label="-30%")
    ax.barh(df["label"], df["positive_change"], color="#d62828", label="+30%")
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_title("优化策略关键成本参数龙卷风图")
    ax.set_xlabel("相对基准成本的变化 / 万元·年$^{-1}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_TORNADO)
    plt.close(fig)


def render_note(
    switch_summary: pd.DataFrame,
    impact_df: pd.DataFrame,
    strategy_summary: pd.DataFrame,
) -> str:
    optimized_row = strategy_summary.loc[strategy_summary["recommended_flag"] == 1].iloc[0]
    top_param = impact_df.iloc[0]
    second_param = impact_df.iloc[1]
    optimized_best_all = bool((switch_summary["optimized_best_levels"] == switch_summary["tested_levels"]).all())

    return f"""# B3 成本敏感性分析说明

## 1. 分析对象

本阶段基于第 3 问得到的策略结果表，不重新做退化仿真，而是固定各策略下的：

- 生命周期长度
- 中维护次数
- 大维护次数

仅考察成本参数变化对策略优选结果的影响。

## 2. 扰动设置

- 购买成本：`300` 万元，扰动范围 `±10%`、`±20%`、`±30%`
- 中维护成本：`3` 万元，扰动范围 `±10%`、`±20%`、`±30%`
- 大维护成本：`12` 万元，扰动范围 `±10%`、`±20%`、`±30%`

## 3. 策略筛选口径

沿用第 3 问的可行性定义：

1. 平均运行透水率至少比当前策略提高 10%
2. 平均剩余寿命不低于当前策略的 95%

在满足上述约束的策略中，比较舰队年均成本。

## 4. 主要结论

- 当前推荐策略：`{optimized_row['strategy_label']}`
- 当前推荐策略基准舰队年均成本：`{optimized_row['fleet_annual_cost']:.2f}` 万元/年
- 推荐策略是否在全部测试场景中保持最优集合内：`{'是' if optimized_best_all else '否'}`

对优化策略本身，参数敏感性排序为：

1. `{top_param['parameter']}`，±30% 扰动导致舰队年均成本总摆幅 ` {top_param['swing_range']:.2f} ` 万元/年
2. `{second_param['parameter']}`，±30% 扰动导致舰队年均成本总摆幅 ` {second_param['swing_range']:.2f} ` 万元/年

其中中维护成本的影响最小，说明当前方案对中维护单价波动相对不敏感；购买成本和大维护成本更值得重点监控。

## 5. 管理建议

- 若采购价格明显上升，应更重视延寿型维护，避免过早更换。
- 若大维护成本上升，应优先排查哪些设备频繁触发大维护阈值，并考虑在第 3 问基础上进一步优化大维护触发阈值。
- 若中维护成本小幅波动，当前推荐策略通常不需要立即调整。

## 6. 输出文件

- `02_建模计算_B/outputs/B3_成本敏感性分析结果.xlsx`
- `02_建模计算_B/outputs/B3_敏感性分析说明.md`
- `02_建模计算_B/figures_B/fig_B3_cost_sensitivity.png`
- `02_建模计算_B/figures_B/fig_B3_sensitivity_tornado.png`
"""


def update_readmes_for_stage6() -> None:
    b_readme = ROOT / "02_建模计算_B" / "README.md"
    b_readme.write_text(
        """# B 组建模计算区

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

## 运行方式

```bash
python3 04_代码/B_prediction_model.py
python3 04_代码/B_optimization.py
python3 04_代码/B_sensitivity_analysis.py
```
""",
        encoding="utf-8",
    )

    code_readme = ROOT / "04_代码" / "README.md"
    code_readme.write_text(
        """# 代码区

本目录存放可复现数据处理与建模结果的脚本。

当前脚本分工：

- `A_data_process.py`：读取附件、清洗数据、生成第 1 问分析底表
- `B_prediction_model.py`：第 2 问寿命预测模型、回测、结果导出与图表生成
- `B_optimization.py`：第 3 问维护策略优化、网格搜索、结果导出与图表生成
- `B_sensitivity_analysis.py`：第 4 问成本敏感性分析、策略稳健性检验与图表生成
""",
        encoding="utf-8",
    )

    root_readme = ROOT / "README.md"
    content = root_readme.read_text(encoding="utf-8")
    content = content.replace(
        "当前下一步重点任务是成员 B 的“阶段 6：第 4 问成本敏感性分析”。",
        "当前下一步重点任务是成员 B 的“阶段 8：联检查错 / 给 C 交接论文素材”。",
    )
    if "B3_成本敏感性分析结果.xlsx" not in content:
        content = content.replace(
            "- 已完成成员 B 的阶段 5 首版交付：\n"
            "  - `04_代码/B_optimization.py`\n"
            "  - `02_建模计算_B/outputs/B2_维护策略优化结果.xlsx`\n"
            "  - `02_建模计算_B/outputs/B2_维护优化模型说明.md`\n"
            "  - `02_建模计算_B/figures_B/fig_B2_maintenance_strategy_comparison.png`\n"
            "  - `02_建模计算_B/figures_B/fig_B2_policy_timeline.png`\n"
            "  - 同步给 C 的 B2 图表",
            "- 已完成成员 B 的阶段 5 首版交付：\n"
            "  - `04_代码/B_optimization.py`\n"
            "  - `02_建模计算_B/outputs/B2_维护策略优化结果.xlsx`\n"
            "  - `02_建模计算_B/outputs/B2_维护优化模型说明.md`\n"
            "  - `02_建模计算_B/figures_B/fig_B2_maintenance_strategy_comparison.png`\n"
            "  - `02_建模计算_B/figures_B/fig_B2_policy_timeline.png`\n"
            "  - 同步给 C 的 B2 图表\n"
            "- 已完成成员 B 的阶段 6 首版交付：\n"
            "  - `04_代码/B_sensitivity_analysis.py`\n"
            "  - `02_建模计算_B/outputs/B3_成本敏感性分析结果.xlsx`\n"
            "  - `02_建模计算_B/outputs/B3_敏感性分析说明.md`\n"
            "  - `02_建模计算_B/figures_B/fig_B3_cost_sensitivity.png`\n"
            "  - `02_建模计算_B/figures_B/fig_B3_sensitivity_tornado.png`\n"
            "  - 同步给 C 的 B3 图表",
        )
    root_readme.write_text(content, encoding="utf-8")


def main() -> None:
    configure_plot_style()
    ensure_dirs()

    strategy_summary, filter_results, _ = load_b2_outputs()
    base_table = build_strategy_base_table(strategy_summary, filter_results)
    scenario_results = build_scenario_results(base_table)
    switch_summary = build_switch_summary(scenario_results)
    impact_df, line_df = build_optimized_parameter_impact(scenario_results)

    plot_cost_sensitivity(line_df)
    plot_tornado(impact_df)

    with pd.ExcelWriter(RESULT_XLSX, engine="openpyxl") as writer:
        scenario_results.to_excel(writer, sheet_name="scenario_results", index=False)
        switch_summary.to_excel(writer, sheet_name="policy_switch_summary", index=False)
        impact_df.to_excel(writer, sheet_name="optimized_parameter_impact", index=False)
        base_table.to_excel(writer, sheet_name="strategy_base_table", index=False)

    NOTE_MD.write_text(
        render_note(switch_summary, impact_df, strategy_summary),
        encoding="utf-8",
    )

    shutil.copy2(FIG_COST_SENS, C_FIGURES / FIG_COST_SENS.name)
    shutil.copy2(FIG_TORNADO, C_FIGURES / FIG_TORNADO.name)

    update_readmes_for_stage6()

    print(f"Saved: {RESULT_XLSX}")
    print(f"Saved: {NOTE_MD}")
    print(f"Saved: {FIG_COST_SENS}")
    print(f"Saved: {FIG_TORNADO}")


if __name__ == "__main__":
    main()
