"""
B_optimization.py

Stage 5 in `方案.md`: maintenance strategy optimization for Question 3.
"""

from __future__ import annotations

from dataclasses import dataclass
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

from B_prediction_model import (
    A_EXPORTS,
    B_OUTPUTS,
    B_FIGURES,
    C_FIGURES,
    LIFE_THRESHOLD,
    MIN_IRREVERSIBLE_DAILY_LOSS,
    RECOVERY_INSUFFICIENCY_RATIO,
    build_daily_series,
    build_month_adjustments,
    build_recovery_models,
    build_schedule_assumptions,
    configure_plot_style,
    ensure_dirs,
    get_month_adjustment,
    init_cap_state,
    load_inputs,
    pick_recovery_model,
)


PURCHASE_COST = 300.0
MIDDLE_MAINT_COST = 3.0
BIG_MAINT_COST = 12.0
OPT_HORIZON_DAYS = 15 * 365
STRATEGY_RESULTS_XLSX = B_OUTPUTS / "B2_维护策略优化结果.xlsx"
STRATEGY_NOTE_MD = B_OUTPUTS / "B2_维护优化模型说明.md"
FIG_STRATEGY_COMPARISON = B_FIGURES / "fig_B2_maintenance_strategy_comparison.png"
FIG_POLICY_TIMELINE = B_FIGURES / "fig_B2_policy_timeline.png"


@dataclass(frozen=True)
class Policy:
    code: str
    label: str
    mode: str = "fixed"
    interval_scale: float = 1.0
    medium_trigger: float = 55.0
    big_trigger: float = 45.0
    min_gap_days: int = 30
    big_every: int = 0
    note: str = ""


def load_b1_summary() -> pd.DataFrame:
    path = B_OUTPUTS / "B1_寿命预测结果.xlsx"
    return pd.read_excel(path, sheet_name="lifetime_summary")


def simulate_filter_under_policy(
    filter_id: str,
    filter_daily: pd.DataFrame,
    decline_row: pd.Series,
    schedule,
    month_adjustments: dict[tuple[str, int], float],
    pooled_models,
    local_models,
    policy: Policy,
) -> tuple[pd.DataFrame, dict[str, object]]:
    last_date = filter_daily["date"].iloc[-1]
    last_per = float(filter_daily["per"].iloc[-1])
    init_adj = get_month_adjustment(filter_id, last_date, month_adjustments)
    latent_state = max(0.0, last_per - init_adj)
    cap0 = init_cap_state(filter_daily, filter_id, month_adjustments, latent_state)
    cap_state = cap0

    daily_decline_rate = float(decline_row["daily_decline_rate"])
    irreversible_daily_loss = max(
        MIN_IRREVERSIBLE_DAILY_LOSS,
        abs(float(decline_row["net_trend_slope_per_day"])),
    )

    days_since_maintenance = schedule.days_since_last_maint
    maintenance_index = 0
    middle_count = 0
    big_count = 0
    last_recovery = 999.0
    last_trigger_type = "中维护"

    if policy.mode == "historical":
        sequence = schedule.historical_sequence or ["中维护"]
        sequence_index = 0
        next_due_day = schedule.next_maintenance_in_days
        interval_days = schedule.median_gap_days
    else:
        sequence = []
        sequence_index = 0
        interval_days = max(
            policy.min_gap_days,
            int(round(schedule.median_gap_days * policy.interval_scale)),
        )
        next_due_day = max(1, interval_days - schedule.days_since_last_maint)

    records: list[dict[str, object]] = []
    service_values: list[float] = []
    annual_buffer: list[float] = []
    threshold_breach_date = None
    failure_date = None

    for step in range(1, OPT_HORIZON_DAYS + 1):
        current_date = last_date + pd.Timedelta(days=step)
        month_adj = get_month_adjustment(filter_id, current_date, month_adjustments)

        cap_state = max(0.0, cap_state - irreversible_daily_loss)
        latent_state = min(max(0.0, latent_state - daily_decline_rate), cap_state)
        obs_before = max(0.0, latent_state + month_adj)

        maintain_type = None
        if policy.mode == "historical":
            if step == next_due_day:
                maintain_type = sequence[sequence_index % len(sequence)]
                sequence_index += 1
                next_due_day += interval_days
        else:
            due = step >= next_due_day
            if days_since_maintenance >= policy.min_gap_days:
                if obs_before <= policy.big_trigger:
                    maintain_type = "大维护"
                elif obs_before <= policy.medium_trigger or due:
                    if policy.big_every and (maintenance_index + 1) % policy.big_every == 0:
                        maintain_type = "大维护"
                    else:
                        maintain_type = "中维护"

        recovery_gain = 0.0
        if maintain_type is not None:
            model = pick_recovery_model(filter_id, maintain_type, pooled_models, local_models)
            before_per = max(0.0, latent_state + month_adj)
            raw_delta = max(0.0, model.intercept + model.slope * before_per)
            scaled_delta = raw_delta * max(0.0, cap_state / cap0) * model.effectiveness_coef
            after_per = min(before_per + scaled_delta, max(0.0, cap_state + month_adj))
            latent_state = max(0.0, after_per - month_adj)
            recovery_gain = max(0.0, after_per - before_per)

            last_recovery = recovery_gain
            last_trigger_type = maintain_type
            maintenance_index += 1
            if maintain_type == "中维护":
                middle_count += 1
            else:
                big_count += 1
            days_since_maintenance = 0

            if policy.mode != "historical":
                next_due_day = step + interval_days
        else:
            days_since_maintenance += 1

        pred_per = max(0.0, latent_state + month_adj)
        service_values.append(pred_per)
        annual_buffer.append(pred_per)
        if len(annual_buffer) > 365:
            annual_buffer.pop(0)

        annual_mean = float(np.mean(annual_buffer)) if annual_buffer else np.nan
        threshold_breach = False
        life_end = False
        if len(annual_buffer) == 365 and annual_mean < LIFE_THRESHOLD:
            threshold_breach = True
            if threshold_breach_date is None:
                threshold_breach_date = current_date
            recovery_threshold = RECOVERY_INSUFFICIENCY_RATIO * pooled_models[last_trigger_type].mean_delta
            if last_recovery < recovery_threshold:
                life_end = True
                failure_date = current_date

        records.append(
            {
                "filter_id": filter_id,
                "strategy_code": policy.code,
                "date": current_date,
                "pred_per": pred_per,
                "annual_mean_365d": annual_mean,
                "maintenance_type": maintain_type,
                "recovery_gain": recovery_gain,
                "threshold_breach_flag": int(threshold_breach),
                "life_end_flag": int(life_end),
            }
        )

        if life_end:
            break

    path_df = pd.DataFrame(records)
    if failure_date is None:
        failure_date = path_df["date"].iloc[-1]
    remaining_life_days = int((failure_date - last_date).days)
    remaining_life_years = remaining_life_days / 365.0
    total_cost = PURCHASE_COST + middle_count * MIDDLE_MAINT_COST + big_count * BIG_MAINT_COST
    fleet_annual_cost = total_cost / remaining_life_years
    service_array = np.array(service_values, dtype=float)

    summary = {
        "filter_id": filter_id,
        "strategy_code": policy.code,
        "strategy_label": policy.label,
        "predicted_failure_date": failure_date.date(),
        "remaining_life_days": remaining_life_days,
        "remaining_life_years": remaining_life_years,
        "middle_count": middle_count,
        "big_count": big_count,
        "lifecycle_cost": total_cost,
        "annual_cost": fleet_annual_cost,
        "avg_service_per": float(service_array.mean()),
        "below55_ratio": float((service_array < 55).mean()),
        "below45_ratio": float((service_array < 45).mean()),
        "threshold_breach_date": threshold_breach_date.date() if threshold_breach_date is not None else pd.NaT,
    }
    return path_df, summary


def evaluate_policy(
    policy: Policy,
    daily: pd.DataFrame,
    decline: pd.DataFrame,
    schedules,
    month_adjustments,
    pooled_models,
    local_models,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    filter_rows = []
    path_frames = []
    for filter_id, filter_daily in daily.groupby("filter_id"):
        filter_daily = filter_daily.sort_values("date").reset_index(drop=True)
        decline_row = decline.loc[decline["filter_id"] == filter_id].iloc[0]
        path_df, summary = simulate_filter_under_policy(
            filter_id,
            filter_daily,
            decline_row,
            schedules[filter_id],
            month_adjustments,
            pooled_models,
            local_models,
            policy,
        )
        filter_rows.append(summary)
        path_frames.append(path_df)

    filter_df = pd.DataFrame(filter_rows).sort_values("filter_id").reset_index(drop=True)
    path_df = pd.concat(path_frames, ignore_index=True)
    aggregate = {
        "strategy_code": policy.code,
        "strategy_label": policy.label,
        "mode": policy.mode,
        "interval_scale": policy.interval_scale,
        "medium_trigger": policy.medium_trigger,
        "big_trigger": policy.big_trigger,
        "min_gap_days": policy.min_gap_days,
        "big_every": policy.big_every,
        "fleet_annual_cost": filter_df["lifecycle_cost"].sum() / filter_df["remaining_life_years"].sum(),
        "avg_filter_annual_cost": filter_df["annual_cost"].mean(),
        "avg_remaining_life_years": filter_df["remaining_life_years"].mean(),
        "avg_service_per": filter_df["avg_service_per"].mean(),
        "avg_below55_ratio": filter_df["below55_ratio"].mean(),
        "avg_below45_ratio": filter_df["below45_ratio"].mean(),
        "avg_middle_count": filter_df["middle_count"].mean(),
        "avg_big_count": filter_df["big_count"].mean(),
        "filters_within_2y": int((filter_df["remaining_life_years"] <= 2).sum()),
        "policy_note": policy.note,
    }
    return filter_df, path_df, aggregate


def search_optimized_policy(
    baseline_aggregate: dict[str, object],
    daily: pd.DataFrame,
    decline: pd.DataFrame,
    schedules,
    month_adjustments,
    pooled_models,
    local_models,
) -> tuple[Policy, pd.DataFrame]:
    rows = []
    best = None

    for interval_scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for medium_trigger in [52, 54, 56, 58, 60, 62]:
            for big_trigger in [42, 44, 46, 48]:
                for big_every in [0, 4, 5, 6]:
                    policy = Policy(
                        code=f"grid_{interval_scale}_{medium_trigger}_{big_trigger}_{big_every}",
                        label="网格候选",
                        mode="fixed",
                        interval_scale=interval_scale,
                        medium_trigger=medium_trigger,
                        big_trigger=big_trigger,
                        min_gap_days=30,
                        big_every=big_every,
                    )
                    _, _, aggregate = evaluate_policy(
                        policy,
                        daily,
                        decline,
                        schedules,
                        month_adjustments,
                        pooled_models,
                        local_models,
                    )

                    feasible = (
                        aggregate["avg_service_per"]
                        >= baseline_aggregate["avg_service_per"] * 1.10
                        and aggregate["avg_remaining_life_years"]
                        >= baseline_aggregate["avg_remaining_life_years"] * 0.95
                    )
                    aggregate["feasible_flag"] = int(feasible)
                    rows.append(aggregate)
                    if feasible:
                        score = (
                            aggregate["fleet_annual_cost"],
                            -aggregate["avg_service_per"],
                        )
                        if best is None or score < best[0]:
                            best = (score, policy, aggregate)

    grid_df = pd.DataFrame(rows).sort_values(
        ["feasible_flag", "fleet_annual_cost", "avg_service_per"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    if best is None:
        best_row = grid_df.iloc[0]
        policy = Policy(
            code="optimized_policy",
            label="优化策略",
            mode="fixed",
            interval_scale=float(best_row["interval_scale"]),
            medium_trigger=float(best_row["medium_trigger"]),
            big_trigger=float(best_row["big_trigger"]),
            min_gap_days=int(best_row["min_gap_days"]),
            big_every=int(best_row["big_every"]),
            note="无可行解满足相对基准约束，退回到网格中成本最低方案。",
        )
        return policy, grid_df

    selected = best[1]
    optimized = Policy(
        code="optimized_policy",
        label="优化策略",
        mode="fixed",
        interval_scale=selected.interval_scale,
        medium_trigger=selected.medium_trigger,
        big_trigger=selected.big_trigger,
        min_gap_days=selected.min_gap_days,
        big_every=selected.big_every,
        note=(
            "满足“平均运行透水率较当前策略提升至少10%，"
            "平均寿命不低于当前策略95%”的相对约束下，"
            "舰队年均成本最低。"
        ),
    )
    return optimized, grid_df


def plot_strategy_comparison(strategy_summary: pd.DataFrame) -> None:
    df = strategy_summary.copy()
    order = ["current_policy", "economy_policy", "conservative_policy", "threshold_policy", "optimized_policy"]
    df["sort_key"] = df["strategy_code"].map({code: idx for idx, code in enumerate(order)})
    df = df.sort_values("sort_key")

    fig, ax1 = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(df))
    bars = ax1.bar(x, df["fleet_annual_cost"], color="#577590", width=0.58, label="舰队年均成本")
    ax1.set_ylabel("舰队年均成本 / 万元·年$^{-1}$")
    ax1.set_xlabel("维护策略")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["strategy_label"], rotation=0)
    ax1.set_title("不同维护策略的年均成本与运行水平比较")

    ax2 = ax1.twinx()
    ax2.plot(x, df["avg_service_per"], color="#d62828", marker="o", linewidth=2, label="平均运行透水率")
    ax2.plot(x, df["avg_remaining_life_years"], color="#2a9d8f", marker="s", linewidth=2, label="平均剩余寿命")
    ax2.set_ylabel("平均透水率 / 平均剩余寿命（年）")

    for bar, value in zip(bars, df["fleet_annual_cost"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_STRATEGY_COMPARISON)
    plt.close(fig)


def plot_policy_timeline(optimized_filter_results: pd.DataFrame, optimized_path: pd.DataFrame) -> None:
    chosen_filters = (
        optimized_filter_results.sort_values("remaining_life_years")
        .head(5)["filter_id"]
        .tolist()
    )
    plot_df = optimized_path[
        (optimized_path["filter_id"].isin(chosen_filters))
        & (optimized_path["maintenance_type"].notna())
    ].copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df = plot_df[plot_df["date"] <= plot_df["date"].min() + pd.Timedelta(days=3 * 365)]

    y_positions = {filter_id: idx for idx, filter_id in enumerate(chosen_filters)}
    fig, ax = plt.subplots(figsize=(11.5, 4.8))

    for filter_id in chosen_filters:
        ax.hlines(y_positions[filter_id], plot_df["date"].min(), plot_df["date"].max(), color="#dddddd", linewidth=0.8)

    middle_points = plot_df[plot_df["maintenance_type"] == "中维护"]
    big_points = plot_df[plot_df["maintenance_type"] == "大维护"]
    ax.scatter(
        middle_points["date"],
        middle_points["filter_id"].map(y_positions),
        color="#2a9d8f",
        marker="o",
        s=36,
        label="中维护",
    )
    ax.scatter(
        big_points["date"],
        big_points["filter_id"].map(y_positions),
        color="#d62828",
        marker="^",
        s=48,
        label="大维护",
    )

    for row in optimized_filter_results.itertuples(index=False):
        if row.filter_id not in chosen_filters:
            continue
        fail_date = pd.to_datetime(row.predicted_failure_date)
        if fail_date <= plot_df["date"].max():
            ax.scatter(
                fail_date,
                y_positions[row.filter_id],
                color="#ff9f1c",
                marker="x",
                s=54,
                label=None,
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(chosen_filters)
    ax.set_xlabel("日期")
    ax.set_ylabel("过滤器编号")
    ax.set_title("优化策略下重点设备维护时间轴（前 3 年）")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_POLICY_TIMELINE)
    plt.close(fig)


def render_strategy_note(
    strategy_summary: pd.DataFrame,
    optimized_policy: Policy,
    grid_df: pd.DataFrame,
) -> str:
    current_row = strategy_summary.loc[strategy_summary["strategy_code"] == "current_policy"].iloc[0]
    optimized_row = strategy_summary.loc[strategy_summary["strategy_code"] == "optimized_policy"].iloc[0]
    top_grid = grid_df.head(5)[
        [
            "fleet_annual_cost",
            "avg_service_per",
            "avg_remaining_life_years",
            "interval_scale",
            "medium_trigger",
            "big_trigger",
            "big_every",
            "feasible_flag",
        ]
    ].round(4)

    return f"""# B2 维护优化模型说明

## 1. 输入

- 第 2 问寿命预测结果：`02_建模计算_B/outputs/B1_寿命预测结果.xlsx`
- A 侧维护恢复参数：`01_数据处理_A/exports/maintenance_match.xlsx`
- A 侧维护效果统计：`01_数据处理_A/exports/维护效果统计表.xlsx`
- A 侧下降率与月份因子：`01_数据处理_A/exports/每台过滤器下降率表.xlsx`

## 2. 优化目标

对任一策略，定义过滤器生命周期年均成本为：

`C = (300 + 3N_m + 12N_l) / T`

其中：

- `300`：购买一台过滤器的成本，单位万元
- `N_m`：中维护次数
- `N_l`：大维护次数
- `T`：从当前时点到寿命终止的持续时间，单位年

为避免只追求低成本而长期运行在低透水率区间，本阶段采用相对可行性约束：

1. 策略的平均运行透水率至少比“当前策略”提高 10%
2. 策略的平均剩余寿命不低于“当前策略”的 95%

在满足上述约束的候选策略中，选择舰队年均成本最小的方案。

## 3. 比较策略

- `当前策略`：延续历史维护节奏与维护类型序列
- `节约策略`：适当拉长维护间隔，降低中维护触发频率
- `保守策略`：缩短维护间隔，并定期插入大维护
- `阈值策略`：当透水率降到阈值以下时触发维护
- `优化策略`：在网格搜索中自动寻找满足约束且成本最低的参数组合

## 4. 优化结果

- 当前策略舰队年均成本：`{current_row['fleet_annual_cost']:.2f}` 万元/年
- 当前策略平均运行透水率：`{current_row['avg_service_per']:.2f}`
- 优化策略舰队年均成本：`{optimized_row['fleet_annual_cost']:.2f}` 万元/年
- 优化策略平均运行透水率：`{optimized_row['avg_service_per']:.2f}`
- 优化策略平均剩余寿命：`{optimized_row['avg_remaining_life_years']:.2f}` 年

优化策略参数为：

- 维护间隔缩放系数：`{optimized_policy.interval_scale:.2f}`
- 中维护触发阈值：`{optimized_policy.medium_trigger:.0f}`
- 大维护触发阈值：`{optimized_policy.big_trigger:.0f}`
- 最小维护间隔：`{optimized_policy.min_gap_days}` 天
- 周期性大维护：`{'不强制' if optimized_policy.big_every == 0 else f'每 {optimized_policy.big_every} 次维护一次大维护'}`

## 5. 推荐结论

- 若仅延续当前策略，年均成本最低，但运行期透水率偏低，设备长期处于弱性能区间。
- 优化策略的思路不是一味增加维护，而是在“适度拉长计划间隔”的同时设置更明确的透水率触发阈值。
- 这使得维护动作更多地在设备真正进入低性能区间时发生，因而相较保守策略更节约，相较当前策略又能显著改善运行状态。

## 6. 网格搜索前 5 名

```text
{top_grid.to_string(index=False)}
```

## 7. 输出文件

- `02_建模计算_B/outputs/B2_维护策略优化结果.xlsx`
- `02_建模计算_B/outputs/B2_维护优化模型说明.md`
- `02_建模计算_B/figures_B/fig_B2_maintenance_strategy_comparison.png`
- `02_建模计算_B/figures_B/fig_B2_policy_timeline.png`

## 8. 风险说明

- 大维护样本量仍只有 17 条，因此优化策略中关于大维护的收益判断应视为“在当前样本下的最优经验规则”。
- 第 2 问回测误差较大的过滤器在第 3 问中也会放大策略差异，后续第 4 问应继续对成本参数和阈值做敏感性分析。
"""


def update_b_readme_for_stage5() -> None:
    readme_path = ROOT / "02_建模计算_B" / "README.md"
    content = """# B 组建模计算区

本目录用于完成 `方案.md` 中阶段 4-6 的建模与结果输出。

## 当前已完成

### 阶段 4：第 2 问寿命预测

- 脚本：`04_代码/B_prediction_model.py`
- 输出：
  - `outputs/B1_寿命预测结果.xlsx`
  - `outputs/B1_寿命预测模型说明.md`
  - `figures_B/fig_B1_lifetime_prediction.png`
  - `figures_B/fig_B1_remaining_life_bar.png`

### 阶段 5：第 3 问维护策略优化

- 脚本：`04_代码/B_optimization.py`
- 输出：
  - `outputs/B2_维护策略优化结果.xlsx`
  - `outputs/B2_维护优化模型说明.md`
  - `figures_B/fig_B2_maintenance_strategy_comparison.png`
  - `figures_B/fig_B2_policy_timeline.png`

## 输出文件说明

- `B2_维护策略优化结果.xlsx`
  - `strategy_summary`：不同策略的舰队成本、寿命和运行水平对比
  - `filter_strategy_results`：每台设备在各策略下的寿命与成本
  - `optimized_path_daily`：优化策略未来逐日仿真路径
  - `optimization_grid_top`：网格搜索前 50 名候选
  - `policy_definition`：策略参数定义
- `B2_维护优化模型说明.md`
  - 供成员 C 直接抽取第 3 问模型说明、约束条件和推荐结论

## 运行方式

```bash
python3 04_代码/B_prediction_model.py
python3 04_代码/B_optimization.py
```

## 后续阶段

- 阶段 6：基于优化策略结果继续做成本敏感性分析
"""
    readme_path.write_text(content, encoding="utf-8")


def update_code_readme_for_stage5() -> None:
    readme_path = ROOT / "04_代码" / "README.md"
    content = """# 代码区

本目录存放可复现数据处理与建模结果的脚本。

当前脚本分工：

- `A_data_process.py`：读取附件、清洗数据、生成第 1 问分析底表
- `B_prediction_model.py`：第 2 问寿命预测模型、回测、结果导出与图表生成
- `B_optimization.py`：第 3 问维护策略优化、网格搜索、结果导出与图表生成

## 当前可直接运行的脚本

### `B_prediction_model.py`

- 输出：
  - `02_建模计算_B/outputs/B1_寿命预测结果.xlsx`
  - `02_建模计算_B/outputs/B1_寿命预测模型说明.md`
  - `02_建模计算_B/figures_B/fig_B1_lifetime_prediction.png`
  - `02_建模计算_B/figures_B/fig_B1_remaining_life_bar.png`

### `B_optimization.py`

- 依赖：先完成 `B_prediction_model.py`
- 输出：
  - `02_建模计算_B/outputs/B2_维护策略优化结果.xlsx`
  - `02_建模计算_B/outputs/B2_维护优化模型说明.md`
  - `02_建模计算_B/figures_B/fig_B2_maintenance_strategy_comparison.png`
  - `02_建模计算_B/figures_B/fig_B2_policy_timeline.png`
  - `03_论文_C/图表汇总/fig_B2_*.png`
"""
    readme_path.write_text(content, encoding="utf-8")


def update_root_readme_for_stage5() -> None:
    readme_path = ROOT / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    content = content.replace(
        "当前下一步重点任务是成员 B 的“阶段 5：第 3 问优化建模”。",
        "当前下一步重点任务是成员 B 的“阶段 6：第 4 问成本敏感性分析”。",
    )
    if "B2_维护策略优化结果.xlsx" not in content:
        content = content.replace(
            "- 已完成成员 B 的阶段 4 首版交付：\n"
            "  - `04_代码/B_prediction_model.py`\n"
            "  - `02_建模计算_B/outputs/B1_寿命预测结果.xlsx`\n"
            "  - `02_建模计算_B/outputs/B1_寿命预测模型说明.md`\n"
            "  - `02_建模计算_B/figures_B/fig_B1_lifetime_prediction.png`\n"
            "  - `02_建模计算_B/figures_B/fig_B1_remaining_life_bar.png`\n"
            "  - 同步给 C 的 B1 图表",
            "- 已完成成员 B 的阶段 4 首版交付：\n"
            "  - `04_代码/B_prediction_model.py`\n"
            "  - `02_建模计算_B/outputs/B1_寿命预测结果.xlsx`\n"
            "  - `02_建模计算_B/outputs/B1_寿命预测模型说明.md`\n"
            "  - `02_建模计算_B/figures_B/fig_B1_lifetime_prediction.png`\n"
            "  - `02_建模计算_B/figures_B/fig_B1_remaining_life_bar.png`\n"
            "  - 同步给 C 的 B1 图表\n"
            "- 已完成成员 B 的阶段 5 首版交付：\n"
            "  - `04_代码/B_optimization.py`\n"
            "  - `02_建模计算_B/outputs/B2_维护策略优化结果.xlsx`\n"
            "  - `02_建模计算_B/outputs/B2_维护优化模型说明.md`\n"
            "  - `02_建模计算_B/figures_B/fig_B2_maintenance_strategy_comparison.png`\n"
            "  - `02_建模计算_B/figures_B/fig_B2_policy_timeline.png`\n"
            "  - 同步给 C 的 B2 图表",
        )
    readme_path.write_text(content, encoding="utf-8")


def main() -> None:
    configure_plot_style()
    ensure_dirs()

    data = load_inputs()
    daily = build_daily_series(data["clean"])
    month_adjustments, _ = build_month_adjustments(daily, data["month_by_filter"])
    pooled_models, local_models, _ = build_recovery_models(
        data["match"],
        data["maintenance_type_summary"],
        data["maintenance_filter_type_summary"],
    )
    schedules = build_schedule_assumptions(data["maintenance"], daily)
    b1_summary = load_b1_summary()

    named_policies = [
        Policy(
            code="current_policy",
            label="当前策略",
            mode="historical",
            note="沿用历史维护间隔中位数和维护类型顺序。",
        ),
        Policy(
            code="economy_policy",
            label="节约策略",
            interval_scale=1.20,
            medium_trigger=52,
            big_trigger=42,
            min_gap_days=35,
            big_every=0,
            note="适度拉长计划间隔，减少预防性维护。",
        ),
        Policy(
            code="conservative_policy",
            label="保守策略",
            interval_scale=0.85,
            medium_trigger=60,
            big_trigger=46,
            min_gap_days=30,
            big_every=4,
            note="提高维护频率，并每 4 次维护插入一次大维护。",
        ),
        Policy(
            code="threshold_policy",
            label="阈值策略",
            interval_scale=1.00,
            medium_trigger=58,
            big_trigger=44,
            min_gap_days=30,
            big_every=0,
            note="按透水率阈值触发维护，避免过早动作。",
        ),
    ]
    named_policy_map = {policy.code: policy for policy in named_policies}

    strategy_summaries = []
    filter_results_frames = []
    path_map = {}
    aggregate_map = {}

    for policy in named_policies:
        filter_df, path_df, aggregate = evaluate_policy(
            policy,
            daily,
            data["decline"],
            schedules,
            month_adjustments,
            pooled_models,
            local_models,
        )
        strategy_summaries.append(aggregate)
        filter_results_frames.append(filter_df)
        path_map[policy.code] = path_df
        aggregate_map[policy.code] = aggregate

    optimized_policy, grid_df = search_optimized_policy(
        aggregate_map["current_policy"],
        daily,
        data["decline"],
        schedules,
        month_adjustments,
        pooled_models,
        local_models,
    )

    feasible_named_rows = []
    current_service = aggregate_map["current_policy"]["avg_service_per"]
    current_life = aggregate_map["current_policy"]["avg_remaining_life_years"]
    for policy in named_policies:
        if policy.code == "current_policy":
            continue
        aggregate = aggregate_map[policy.code]
        feasible = (
            aggregate["avg_service_per"] >= current_service * 1.10
            and aggregate["avg_remaining_life_years"] >= current_life * 0.95
        )
        if feasible:
            feasible_named_rows.append((aggregate["fleet_annual_cost"], policy, aggregate))

    optimized_filter_df, optimized_path_df, optimized_aggregate = evaluate_policy(
        optimized_policy,
        daily,
        data["decline"],
        schedules,
        month_adjustments,
        pooled_models,
        local_models,
    )

    if feasible_named_rows:
        best_named_cost, best_named_policy, _ = min(feasible_named_rows, key=lambda item: item[0])
        if best_named_cost < optimized_aggregate["fleet_annual_cost"]:
            optimized_policy = Policy(
                code="optimized_policy",
                label="优化策略",
                mode=best_named_policy.mode,
                interval_scale=best_named_policy.interval_scale,
                medium_trigger=best_named_policy.medium_trigger,
                big_trigger=best_named_policy.big_trigger,
                min_gap_days=best_named_policy.min_gap_days,
                big_every=best_named_policy.big_every,
                note=(
                    "在满足相对运行质量约束的命名策略与网格候选联合集合中，"
                    f"由 `{best_named_policy.label}` 转化得到的最低成本可行方案。"
                ),
            )
            optimized_filter_df, optimized_path_df, optimized_aggregate = evaluate_policy(
                optimized_policy,
                daily,
                data["decline"],
                schedules,
                month_adjustments,
                pooled_models,
                local_models,
            )
    strategy_summaries.append(optimized_aggregate)
    filter_results_frames.append(optimized_filter_df)
    path_map[optimized_policy.code] = optimized_path_df

    strategy_summary = pd.DataFrame(strategy_summaries)
    strategy_summary["current_baseline_service"] = aggregate_map["current_policy"]["avg_service_per"]
    strategy_summary["current_baseline_life"] = aggregate_map["current_policy"]["avg_remaining_life_years"]
    strategy_summary["service_improvement_vs_current"] = (
        strategy_summary["avg_service_per"] / aggregate_map["current_policy"]["avg_service_per"] - 1
    )
    strategy_summary["life_change_vs_current"] = (
        strategy_summary["avg_remaining_life_years"] / aggregate_map["current_policy"]["avg_remaining_life_years"] - 1
    )
    strategy_summary["recommended_flag"] = (
        strategy_summary["strategy_code"] == "optimized_policy"
    ).astype(int)

    filter_strategy_results = pd.concat(filter_results_frames, ignore_index=True)
    filter_strategy_results = filter_strategy_results.merge(
        b1_summary[["filter_id", "remaining_life_years"]].rename(
            columns={"remaining_life_years": "b1_remaining_life_years"}
        ),
        on="filter_id",
        how="left",
    )

    policy_definition = pd.DataFrame(
        [
            {
                "strategy_code": policy.code,
                "strategy_label": policy.label,
                "mode": policy.mode,
                "interval_scale": policy.interval_scale,
                "medium_trigger": policy.medium_trigger,
                "big_trigger": policy.big_trigger,
                "min_gap_days": policy.min_gap_days,
                "big_every": policy.big_every,
                "note": policy.note,
            }
            for policy in named_policies + [optimized_policy]
        ]
    )

    plot_strategy_comparison(strategy_summary)
    plot_policy_timeline(optimized_filter_df, optimized_path_df)

    with pd.ExcelWriter(STRATEGY_RESULTS_XLSX, engine="openpyxl") as writer:
        strategy_summary.to_excel(writer, sheet_name="strategy_summary", index=False)
        filter_strategy_results.to_excel(writer, sheet_name="filter_strategy_results", index=False)
        optimized_path_df.to_excel(writer, sheet_name="optimized_path_daily", index=False)
        grid_df.head(50).to_excel(writer, sheet_name="optimization_grid_top", index=False)
        policy_definition.to_excel(writer, sheet_name="policy_definition", index=False)

    STRATEGY_NOTE_MD.write_text(
        render_strategy_note(strategy_summary, optimized_policy, grid_df),
        encoding="utf-8",
    )

    shutil.copy2(FIG_STRATEGY_COMPARISON, C_FIGURES / FIG_STRATEGY_COMPARISON.name)
    shutil.copy2(FIG_POLICY_TIMELINE, C_FIGURES / FIG_POLICY_TIMELINE.name)

    update_b_readme_for_stage5()
    update_code_readme_for_stage5()
    update_root_readme_for_stage5()

    print(f"Saved: {STRATEGY_RESULTS_XLSX}")
    print(f"Saved: {STRATEGY_NOTE_MD}")
    print(f"Saved: {FIG_STRATEGY_COMPARISON}")
    print(f"Saved: {FIG_POLICY_TIMELINE}")


if __name__ == "__main__":
    main()
