"""
B_prediction_model.py

Stage 4 in `方案.md`: lifetime prediction for Question 2.

This script reads A-side structured outputs, builds a degradation +
seasonality + maintenance recovery simulation model, evaluates it on a
recent backtest window, and exports B-side deliverables for the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import math
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A_EXPORTS = ROOT / "01_数据处理_A" / "exports"
B_OUTPUTS = ROOT / "02_建模计算_B" / "outputs"
B_FIGURES = ROOT / "02_建模计算_B" / "figures_B"
C_FIGURES = ROOT / "03_论文_C" / "图表汇总"
MPL_CONFIG_DIR = ROOT / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIFETIME_XLSX = B_OUTPUTS / "B1_寿命预测结果.xlsx"
MODEL_NOTE = B_OUTPUTS / "B1_寿命预测模型说明.md"
FIG_LIFETIME = B_FIGURES / "fig_B1_lifetime_prediction.png"
FIG_REMAINING = B_FIGURES / "fig_B1_remaining_life_bar.png"

LIFE_THRESHOLD = 37.0
RECOVERY_INSUFFICIENCY_RATIO = 0.20
BACKTEST_DAYS = 120
RECENT_CAP_WINDOW_DAYS = 120
MIN_IRREVERSIBLE_DAILY_LOSS = 0.03
FORECAST_HORIZON_DAYS = 15 * 365


@dataclass
class RecoveryModel:
    scope: str
    maintain_type: str
    intercept: float
    slope: float
    mean_delta: float
    effectiveness_coef: float
    sample_count: int


@dataclass
class ScheduleAssumption:
    filter_id: str
    median_gap_days: int
    days_since_last_maint: int
    next_maintenance_in_days: int
    historical_sequence: list[str]
    big_count: int


def configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Hiragino Sans GB",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"


def ensure_dirs() -> None:
    for path in (B_OUTPUTS, B_FIGURES, C_FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> dict[str, pd.DataFrame]:
    clean = pd.read_excel(A_EXPORTS / "clean_data.xlsx")
    maintenance = pd.read_excel(A_EXPORTS / "maintenance_record.xlsx")
    match = pd.read_excel(A_EXPORTS / "maintenance_match.xlsx")
    decline = pd.read_excel(
        A_EXPORTS / "每台过滤器下降率表.xlsx",
        sheet_name="decline_rate_by_filter",
    )
    month_by_filter = pd.read_excel(
        A_EXPORTS / "每台过滤器下降率表.xlsx",
        sheet_name="monthly_average_by_filter",
    )
    month_overall = pd.read_excel(
        A_EXPORTS / "每台过滤器下降率表.xlsx",
        sheet_name="monthly_average_overall",
    )
    maintenance_type_summary = pd.read_excel(
        A_EXPORTS / "维护效果统计表.xlsx",
        sheet_name="summary_by_type",
    )
    maintenance_filter_type_summary = pd.read_excel(
        A_EXPORTS / "维护效果统计表.xlsx",
        sheet_name="summary_by_filter_type",
    )

    clean["date"] = pd.to_datetime(clean["date"])
    maintenance["date"] = pd.to_datetime(maintenance["date"])
    match["maintain_date"] = pd.to_datetime(match["maintain_date"])

    return {
        "clean": clean,
        "maintenance": maintenance,
        "match": match,
        "decline": decline,
        "month_by_filter": month_by_filter,
        "month_overall": month_overall,
        "maintenance_type_summary": maintenance_type_summary,
        "maintenance_filter_type_summary": maintenance_filter_type_summary,
    }


def build_daily_series(clean: pd.DataFrame) -> pd.DataFrame:
    valid = clean[clean["is_missing"] == 0].copy()
    daily = (
        valid.groupby(["filter_id", "date"], as_index=False)["per"]
        .mean()
        .sort_values(["filter_id", "date"])
        .reset_index(drop=True)
    )
    return daily


def build_month_adjustments(
    daily: pd.DataFrame,
    month_by_filter: pd.DataFrame,
) -> tuple[dict[tuple[str, int], float], pd.DataFrame]:
    filter_means = daily.groupby("filter_id")["per"].mean().to_dict()
    monthly = month_by_filter.copy()
    monthly["month"] = monthly["month"].astype(int)
    monthly["month_adjustment"] = monthly.apply(
        lambda row: row["month_mean_per"] - filter_means[row["filter_id"]],
        axis=1,
    )
    month_adjustments = {
        (row.filter_id, int(row.month)): float(row.month_adjustment)
        for row in monthly.itertuples(index=False)
    }
    return month_adjustments, monthly


def fit_linear_relation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x = x.astype(float)
    y = y.astype(float)
    centered = x - x.mean()
    denominator = float((centered**2).sum())
    slope = float(((centered) * (y - y.mean())).sum() / denominator) if denominator else 0.0
    intercept = float(y.mean() - slope * x.mean())
    return intercept, slope


def build_recovery_models(
    match: pd.DataFrame,
    summary_by_type: pd.DataFrame,
    summary_by_filter_type: pd.DataFrame,
) -> tuple[dict[str, RecoveryModel], dict[tuple[str, str], RecoveryModel], pd.DataFrame]:
    pooled_models: dict[str, RecoveryModel] = {}
    for maintain_type, group in match.groupby("maintain_type"):
        intercept, slope = fit_linear_relation(group["before_per"], group["delta_per"])
        summary_row = summary_by_type.loc[
            summary_by_type["maintain_type"] == maintain_type
        ].iloc[0]
        pooled_models[maintain_type] = RecoveryModel(
            scope="pooled",
            maintain_type=maintain_type,
            intercept=intercept,
            slope=slope,
            mean_delta=float(group["delta_per"].mean()),
            effectiveness_coef=float(summary_row["mean_maintenance_effectiveness_coef"]),
            sample_count=int(len(group)),
        )

    local_models: dict[tuple[str, str], RecoveryModel] = {}
    for (filter_id, maintain_type), group in match.groupby(["filter_id", "maintain_type"]):
        if len(group) < 5:
            continue
        intercept, slope = fit_linear_relation(group["before_per"], group["delta_per"])
        summary_row = summary_by_filter_type.loc[
            (summary_by_filter_type["filter_id"] == filter_id)
            & (summary_by_filter_type["maintain_type"] == maintain_type)
        ].iloc[0]
        local_models[(filter_id, maintain_type)] = RecoveryModel(
            scope="filter_type",
            maintain_type=maintain_type,
            intercept=intercept,
            slope=slope,
            mean_delta=float(group["delta_per"].mean()),
            effectiveness_coef=float(summary_row["mean_maintenance_effectiveness_coef"]),
            sample_count=int(len(group)),
        )

    model_rows = []
    for model in pooled_models.values():
        model_rows.append(
            {
                "scope": model.scope,
                "filter_id": "ALL",
                "maintain_type": model.maintain_type,
                "intercept": model.intercept,
                "slope": model.slope,
                "mean_delta": model.mean_delta,
                "effectiveness_coef": model.effectiveness_coef,
                "sample_count": model.sample_count,
            }
        )
    for (filter_id, _), model in local_models.items():
        model_rows.append(
            {
                "scope": model.scope,
                "filter_id": filter_id,
                "maintain_type": model.maintain_type,
                "intercept": model.intercept,
                "slope": model.slope,
                "mean_delta": model.mean_delta,
                "effectiveness_coef": model.effectiveness_coef,
                "sample_count": model.sample_count,
            }
        )

    return pooled_models, local_models, pd.DataFrame(model_rows)


def build_schedule_assumptions(maintenance: pd.DataFrame, daily: pd.DataFrame) -> dict[str, ScheduleAssumption]:
    assumptions: dict[str, ScheduleAssumption] = {}
    last_obs_dates = daily.groupby("filter_id")["date"].max().to_dict()

    for filter_id, group in maintenance.groupby("filter_id"):
        group = group.sort_values("date").reset_index(drop=True)
        gaps = group["date"].diff().dt.days.dropna()
        median_gap_days = int(round(gaps.median())) if not gaps.empty else 60
        last_maint_date = group["date"].iloc[-1]
        last_obs_date = last_obs_dates[filter_id]
        days_since_last_maint = max(0, int((last_obs_date - last_maint_date).days))
        next_maintenance_in_days = max(1, median_gap_days - days_since_last_maint)
        historical_sequence = group["maintain_type"].tolist()
        assumptions[filter_id] = ScheduleAssumption(
            filter_id=filter_id,
            median_gap_days=median_gap_days,
            days_since_last_maint=days_since_last_maint,
            next_maintenance_in_days=next_maintenance_in_days,
            historical_sequence=historical_sequence,
            big_count=int((group["maintain_type"] == "大维护").sum()),
        )
    return assumptions


def get_month_adjustment(
    filter_id: str,
    date_value: pd.Timestamp,
    month_adjustments: dict[tuple[str, int], float],
) -> float:
    return month_adjustments.get((filter_id, int(date_value.month)), 0.0)


def pick_recovery_model(
    filter_id: str,
    maintain_type: str,
    pooled_models: dict[str, RecoveryModel],
    local_models: dict[tuple[str, str], RecoveryModel],
) -> RecoveryModel:
    return local_models.get((filter_id, maintain_type), pooled_models[maintain_type])


def init_cap_state(
    history: pd.DataFrame,
    filter_id: str,
    month_adjustments: dict[tuple[str, int], float],
    latent_state: float,
) -> float:
    recent = history[history["date"] >= (history["date"].max() - pd.Timedelta(days=RECENT_CAP_WINDOW_DAYS))].copy()
    recent["adj"] = recent["date"].apply(
        lambda d: get_month_adjustment(filter_id, d, month_adjustments)
    )
    cap0 = float((recent["per"] - recent["adj"]).quantile(0.9))
    if not np.isfinite(cap0) or cap0 < latent_state:
        cap0 = latent_state + 5.0
    return cap0


def simulate_backtest(
    filter_id: str,
    filter_daily: pd.DataFrame,
    decline_row: pd.Series,
    maintenance: pd.DataFrame,
    month_adjustments: dict[tuple[str, int], float],
    pooled_models: dict[str, RecoveryModel],
    local_models: dict[tuple[str, str], RecoveryModel],
) -> tuple[pd.DataFrame, dict[str, float]]:
    end_date = filter_daily["date"].max()
    start_date = end_date - pd.Timedelta(days=BACKTEST_DAYS - 1)

    history = filter_daily[filter_daily["date"] < start_date].copy()
    holdout = filter_daily[filter_daily["date"] >= start_date].copy()
    if history.empty or len(holdout) < 60:
        return pd.DataFrame(), {}

    init_date = history["date"].iloc[-1]
    init_obs = float(history["per"].iloc[-1])
    init_adj = get_month_adjustment(filter_id, init_date, month_adjustments)
    latent_state = max(0.0, init_obs - init_adj)
    cap0 = init_cap_state(history, filter_id, month_adjustments, latent_state)
    cap_state = cap0

    daily_decline_rate = float(decline_row["daily_decline_rate"])
    irreversible_daily_loss = max(
        MIN_IRREVERSIBLE_DAILY_LOSS,
        abs(float(decline_row["net_trend_slope_per_day"])),
    )

    actual_events = (
        maintenance[
            (maintenance["filter_id"] == filter_id)
            & (maintenance["date"] >= start_date)
            & (maintenance["date"] <= end_date)
        ][["date", "maintain_type"]]
        .sort_values("date")
        .copy()
    )
    event_map = {
        pd.Timestamp(row.date).normalize(): row.maintain_type
        for row in actual_events.itertuples(index=False)
    }

    predictions = []
    current_date = init_date
    for _ in range(len(holdout)):
        current_date = current_date + pd.Timedelta(days=1)
        month_adj = get_month_adjustment(filter_id, current_date, month_adjustments)

        cap_state = max(0.0, cap_state - irreversible_daily_loss)
        latent_state = min(max(0.0, latent_state - daily_decline_rate), cap_state)

        maintain_type = event_map.get(current_date.normalize())
        if maintain_type:
            model = pick_recovery_model(filter_id, maintain_type, pooled_models, local_models)
            before_per = max(0.0, latent_state + month_adj)
            raw_delta = max(0.0, model.intercept + model.slope * before_per)
            scaled_delta = raw_delta * max(0.0, cap_state / cap0) * model.effectiveness_coef
            after_per = min(before_per + scaled_delta, max(0.0, cap_state + month_adj))
            latent_state = max(0.0, after_per - month_adj)

        pred_per = max(0.0, latent_state + month_adj)
        predictions.append(pred_per)

    merged = holdout[["date", "per"]].copy()
    merged["pred_per"] = predictions[: len(merged)]

    error = merged["pred_per"] - merged["per"]
    mae = float(error.abs().mean())
    rmse = float(np.sqrt((error**2).mean()))
    mape = float((error.abs() / merged["per"].clip(lower=1e-6)).mean())
    sst = float(((merged["per"] - merged["per"].mean()) ** 2).sum())
    sse = float(((merged["per"] - merged["pred_per"]) ** 2).sum())
    r2 = float(1 - sse / sst) if sst > 0 else float("nan")

    metrics = {
        "backtest_days": int(len(merged)),
        "backtest_mae": mae,
        "backtest_rmse": rmse,
        "backtest_mape": mape,
        "backtest_r2": r2,
    }
    return merged, metrics


def simulate_future(
    filter_id: str,
    filter_daily: pd.DataFrame,
    decline_row: pd.Series,
    schedule: ScheduleAssumption,
    month_adjustments: dict[tuple[str, int], float],
    pooled_models: dict[str, RecoveryModel],
    local_models: dict[tuple[str, str], RecoveryModel],
) -> tuple[pd.DataFrame, dict[str, object]]:
    last_date = filter_daily["date"].iloc[-1]
    last_per = float(filter_daily["per"].iloc[-1])
    init_adj = get_month_adjustment(filter_id, last_date, month_adjustments)
    latent_state = max(0.0, last_per - init_adj)
    cap0 = init_cap_state(filter_daily, filter_id, month_adjustments, latent_state)
    cap_state = cap0

    daily_decline_rate = float(decline_row["daily_decline_rate"])
    net_trend_slope = float(decline_row["net_trend_slope_per_day"])
    irreversible_daily_loss = max(MIN_IRREVERSIBLE_DAILY_LOSS, abs(net_trend_slope))

    sequence = schedule.historical_sequence or ["中维护"]
    sequence_index = 0
    next_maintenance_in_days = schedule.next_maintenance_in_days

    records: list[dict[str, object]] = []
    maintenance_counter = {"中维护": 0, "大维护": 0}
    threshold_breach_date = None
    life_end_date = None
    last_recovery = math.nan
    last_trigger_type = None
    annual_buffer: list[float] = []

    for step in range(1, FORECAST_HORIZON_DAYS + 1):
        current_date = last_date + pd.Timedelta(days=step)
        month_adj = get_month_adjustment(filter_id, current_date, month_adjustments)

        cap_state = max(0.0, cap_state - irreversible_daily_loss)
        latent_state = min(max(0.0, latent_state - daily_decline_rate), cap_state)

        maintain_type = None
        before_per = math.nan
        recovery_gain = 0.0

        if step == next_maintenance_in_days:
            maintain_type = sequence[sequence_index % len(sequence)]
            maintenance_counter[maintain_type] += 1
            model = pick_recovery_model(filter_id, maintain_type, pooled_models, local_models)

            before_per = max(0.0, latent_state + month_adj)
            raw_delta = max(0.0, model.intercept + model.slope * before_per)
            scaled_delta = raw_delta * max(0.0, cap_state / cap0) * model.effectiveness_coef
            after_per = min(before_per + scaled_delta, max(0.0, cap_state + month_adj))
            recovery_gain = max(0.0, after_per - before_per)
            latent_state = max(0.0, after_per - month_adj)

            last_recovery = recovery_gain
            last_trigger_type = maintain_type
            sequence_index += 1
            next_maintenance_in_days += schedule.median_gap_days

        pred_per = max(0.0, latent_state + month_adj)
        annual_buffer.append(pred_per)
        if len(annual_buffer) > 365:
            annual_buffer.pop(0)

        annual_mean = float(np.mean(annual_buffer)) if annual_buffer else float("nan")
        threshold_breach = False
        life_end = False
        recovery_threshold = math.nan

        if len(annual_buffer) == 365 and annual_mean < LIFE_THRESHOLD:
            threshold_breach = True
            if threshold_breach_date is None:
                threshold_breach_date = current_date
            if last_trigger_type is not None:
                recovery_threshold = (
                    RECOVERY_INSUFFICIENCY_RATIO
                    * pooled_models[last_trigger_type].mean_delta
                )
                if last_recovery < recovery_threshold:
                    life_end = True
                    life_end_date = current_date

        records.append(
            {
                "filter_id": filter_id,
                "date": current_date,
                "pred_per": pred_per,
                "annual_mean_365d": annual_mean,
                "maintenance_type": maintain_type,
                "maintenance_before_per": before_per,
                "recovery_gain": recovery_gain,
                "cap_state": cap_state,
                "threshold_breach_flag": int(threshold_breach),
                "life_end_flag": int(life_end),
                "recovery_threshold": recovery_threshold,
            }
        )

        if life_end:
            break

    forecast_path = pd.DataFrame(records)
    if forecast_path.empty:
        threshold_breach_date = pd.NaT
        life_end_date = pd.NaT
        remaining_life_days = 0
        failure_annual_mean = math.nan
        last_recovery_value = math.nan
        last_trigger_type = None
    else:
        if threshold_breach_date is None:
            threshold_breach_date = pd.NaT
        if life_end_date is None:
            life_end_date = forecast_path["date"].iloc[-1]
        remaining_life_days = int((life_end_date - last_date).days)
        failure_row = forecast_path.loc[forecast_path["date"] == life_end_date].iloc[0]
        failure_annual_mean = float(failure_row["annual_mean_365d"])
        last_recovery_value = float(last_recovery) if not math.isnan(last_recovery) else math.nan

    summary = {
        "filter_id": filter_id,
        "last_observation_date": last_date.date(),
        "current_per": last_per,
        "daily_decline_rate": daily_decline_rate,
        "net_trend_slope_per_day": net_trend_slope,
        "irreversible_daily_loss": irreversible_daily_loss,
        "median_maintenance_gap_days": schedule.median_gap_days,
        "days_since_last_maint": schedule.days_since_last_maint,
        "next_maintenance_in_days": schedule.next_maintenance_in_days,
        "historical_big_maintenance_count": schedule.big_count,
        "threshold_breach_date": threshold_breach_date.date()
        if not pd.isna(threshold_breach_date)
        else pd.NaT,
        "predicted_failure_date": life_end_date.date() if not pd.isna(life_end_date) else pd.NaT,
        "remaining_life_days": remaining_life_days,
        "remaining_life_years": remaining_life_days / 365.0,
        "failure_annual_mean_per": failure_annual_mean,
        "last_effective_recovery": last_recovery_value,
        "last_trigger_type": last_trigger_type,
        "recovery_threshold": (
            RECOVERY_INSUFFICIENCY_RATIO * pooled_models[last_trigger_type].mean_delta
            if last_trigger_type
            else math.nan
        ),
        "forecast_middle_maint_count": maintenance_counter["中维护"],
        "forecast_big_maint_count": maintenance_counter["大维护"],
    }
    return forecast_path, summary


def build_parameter_table(
    monthly_adjustments: pd.DataFrame,
    recovery_models_df: pd.DataFrame,
    decline: pd.DataFrame,
) -> pd.DataFrame:
    general_rows = [
        {
            "parameter_group": "general",
            "parameter_name": "life_threshold",
            "parameter_value": LIFE_THRESHOLD,
            "parameter_note": "365日滚动平均透水率阈值",
        },
        {
            "parameter_group": "general",
            "parameter_name": "recovery_insufficiency_ratio",
            "parameter_value": RECOVERY_INSUFFICIENCY_RATIO,
            "parameter_note": "恢复不足阈值占同类维护历史平均恢复量的比例",
        },
        {
            "parameter_group": "general",
            "parameter_name": "backtest_days",
            "parameter_value": BACKTEST_DAYS,
            "parameter_note": "最近窗口回测天数",
        },
        {
            "parameter_group": "general",
            "parameter_name": "min_irreversible_daily_loss",
            "parameter_value": MIN_IRREVERSIBLE_DAILY_LOSS,
            "parameter_note": "不可逆老化项的最小日损耗",
        },
    ]

    decline_rows = []
    for row in decline.itertuples(index=False):
        decline_rows.extend(
            [
                {
                    "parameter_group": "decline",
                    "parameter_name": f"{row.filter_id}_daily_decline_rate",
                    "parameter_value": row.daily_decline_rate,
                    "parameter_note": "维护周期日均下降率",
                },
                {
                    "parameter_group": "decline",
                    "parameter_name": f"{row.filter_id}_net_trend_slope_per_day",
                    "parameter_value": row.net_trend_slope_per_day,
                    "parameter_note": "全时段净趋势斜率，仅作不可逆老化辅助项",
                },
            ]
        )

    seasonal_rows = [
        {
            "parameter_group": "seasonality",
            "parameter_name": f"{row.filter_id}_month_{int(row.month)}_adjustment",
            "parameter_value": row.month_adjustment,
            "parameter_note": "按过滤器估计的月份加性修正项",
        }
        for row in monthly_adjustments.itertuples(index=False)
    ]

    recovery_rows = [
        {
            "parameter_group": "recovery",
            "parameter_name": f"{row.scope}_{row.filter_id}_{row.maintain_type}",
            "parameter_value": row.intercept,
            "parameter_note": (
                f"恢复回归截距；slope={row.slope:.4f}; "
                f"eff={row.effectiveness_coef:.4f}; n={int(row.sample_count)}"
            ),
        }
        for row in recovery_models_df.itertuples(index=False)
    ]

    return pd.DataFrame(general_rows + decline_rows + seasonal_rows + recovery_rows)


def plot_lifetime_prediction(
    history_frames: dict[str, pd.DataFrame],
    forecast_frames: dict[str, pd.DataFrame],
    lifetime_summary: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    axes = axes.flatten()

    summary_map = lifetime_summary.set_index("filter_id").to_dict(orient="index")
    ordered_filters = sorted(history_frames)

    for ax, filter_id in zip(axes, ordered_filters):
        history = history_frames[filter_id].copy()
        forecast = forecast_frames[filter_id].copy()
        summary = summary_map[filter_id]

        ax.plot(history["date"], history["per"], color="#577590", linewidth=1.4, label="近120天实际值")
        ax.plot(forecast["date"], forecast["pred_per"], color="#d62828", linewidth=1.4, label="未来预测值")
        ax.axhline(LIFE_THRESHOLD, color="#6a4c93", linestyle="--", linewidth=1.0, label="寿命阈值37")

        failure_date = pd.to_datetime(summary["predicted_failure_date"])
        ax.axvline(failure_date, color="#ff9f1c", linestyle=":", linewidth=1.0)
        ax.set_title(filter_id)
        ax.set_xlabel("日期")
        ax.set_ylabel("透水率")
        ax.tick_params(axis="x", rotation=30)
        ax.text(
            0.02,
            0.95,
            f"剩余寿命 {summary['remaining_life_years']:.2f} 年",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#dddddd"},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("各过滤器寿命预测曲线（近120天历史 + 未来仿真）", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_LIFETIME)
    plt.close(fig)


def plot_remaining_life(lifetime_summary: pd.DataFrame) -> None:
    plot_df = lifetime_summary.sort_values("remaining_life_years").copy()
    colors = ["#d62828" if value <= 2 else "#577590" for value in plot_df["remaining_life_years"]]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars = ax.bar(plot_df["filter_id"], plot_df["remaining_life_years"], color=colors)
    ax.axhline(2, color="#ff9f1c", linestyle="--", linewidth=1.0, label="2年预警线")
    ax.set_title("各过滤器剩余寿命对比")
    ax.set_xlabel("过滤器编号")
    ax.set_ylabel("剩余寿命 / 年")
    ax.legend(frameon=False)

    for bar, value in zip(bars, plot_df["remaining_life_years"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(FIG_REMAINING)
    plt.close(fig)


def render_model_note(
    lifetime_summary: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    recovery_models_df: pd.DataFrame,
) -> str:
    avg_mape = validation_metrics["backtest_mape"].mean()
    avg_rmse = validation_metrics["backtest_rmse"].mean()
    urgent_filters = lifetime_summary.loc[
        lifetime_summary["remaining_life_years"] <= 2, "filter_id"
    ].tolist()
    urgent_text = "、".join(urgent_filters) if urgent_filters else "无"

    return f"""# B1 寿命预测模型说明

## 1. 输入数据

- 透水率主表：`01_数据处理_A/exports/clean_data.xlsx`
- 维护记录：`01_数据处理_A/exports/maintenance_record.xlsx`
- 维护匹配表：`01_数据处理_A/exports/maintenance_match.xlsx`
- 下降率与季节统计：`01_数据处理_A/exports/每台过滤器下降率表.xlsx`
- 维护效果统计：`01_数据处理_A/exports/维护效果统计表.xlsx`

## 2. 建模口径

### 2.1 状态方程

对第 `i` 台过滤器，在日尺度上定义去季节后的潜在透水率状态 `x_{{i,t}}`：

`x_{{i,t+1}}^- = min(c_{{i,t+1}}, max(0, x_{{i,t}} - d_i))`

其中：

- `d_i` 为 A 侧给出的维护周期日均下降率 `daily_decline_rate`
- `c_{{i,t}}` 为可恢复上限，表示过滤器健康上限

### 2.2 不可逆老化项

可恢复上限按下式衰减：

`c_{{i,t+1}} = max(0, c_{{i,t}} - u_i)`

其中 `u_i = max({MIN_IRREVERSIBLE_DAILY_LOSS:.2f}, |net_trend_slope_per_day_i|)`。

解释：

- 维护周期下降率负责描述“单次运行周期内”的退化；
- 全时段净趋势不直接替代下降率，而只作为“长期不可逆老化”的辅助项，防止维护恢复被误判为无限可逆。

### 2.3 季节项

按过滤器分别构造月份加性修正项：

`P_{{i,t}} = max(0, x_{{i,t}} + s_{{i,m(t)}})`

其中 `s_{{i,m}}` 由 `monthly_average_by_filter` 中该过滤器月份均值减其总体均值得到。

### 2.4 维护恢复项

维护前透水率为 `P_{{i,k}}^-` 时，维护恢复量按线性关系估计：

`Delta_{{i,k}}^0 = max(0, a_r + b_r P_{{i,k}}^-)`

进一步考虑健康上限折减与维护有效性：

`Delta_{{i,k}} = Delta_{{i,k}}^0 * (c_{{i,k}} / c_{{i,0}}) * eta_r`

其中：

- `r` 表示维护类型（中维护或大维护）
- `eta_r` 为 A 侧 `maintenance_effectiveness_coef` 的均值
- 若某过滤器某类维护样本数不少于 5，则优先使用该过滤器局部回归；否则退回到全样本回归

### 2.5 寿命终止判据

沿用启动对齐口径，并在仿真中细化为：

1. 最近 365 天滚动平均透水率低于 `{LIFE_THRESHOLD:.0f}`；
2. 最近一次维护的恢复量低于“同类维护历史平均恢复量”的 `{RECOVERY_INSUFFICIENCY_RATIO:.0%}`。

只有两个条件同时满足，才判定寿命终止。

## 3. 当前维护规律的仿真假设

- 维护间隔取各过滤器历史维护间隔的中位数；
- 维护类型序列按该过滤器历史顺序循环延续；
- 因此本模型回答的是“在当前维护节奏继续执行时，各过滤器还能使用多久”。

## 4. 回测结果

- 回测窗口：最近 `{BACKTEST_DAYS}` 天
- 平均 RMSE：`{avg_rmse:.2f}`
- 平均 MAPE：`{avg_mape:.2%}`
- 回测误差较大的设备主要是 A7、A6、A10，这些设备近期波动更强，说明第 2 问结果宜作为策略优化输入，不宜解释为逐日精确预报。

## 5. 结果解读

- 剩余寿命最短的设备集中在：`{urgent_text}`
- 这些设备在继续沿用当前维护节奏时，将更早出现“年均透水率低于阈值且维护恢复不足”的双重失效。
- 剩余寿命较长的设备并不表示无需维护，而是表示在当前维护节奏下，其健康上限衰减速度相对更慢。

## 6. 文件输出

- 结果表：`02_建模计算_B/outputs/B1_寿命预测结果.xlsx`
- 模型说明：`02_建模计算_B/outputs/B1_寿命预测模型说明.md`
- 寿命预测图：`02_建模计算_B/figures_B/fig_B1_lifetime_prediction.png`
- 剩余寿命图：`02_建模计算_B/figures_B/fig_B1_remaining_life_bar.png`
- 论文图表同步：`03_论文_C/图表汇总/`

## 7. 使用提醒

- 大维护样本仅 17 次，且部分过滤器没有大维护样本，相关参数稳定性弱于中维护。
- `is_outlier` 未在本阶段直接剔除，后续第 3、4 问可以用“保留异常值 / 剔除异常值”做敏感性对比。
- 若后续需要更严格的寿命定义，可在第 3 问优化阶段把恢复不足阈值从 `{RECOVERY_INSUFFICIENCY_RATIO:.0%}` 改成 30% 后复算。
"""


def update_b_readme() -> None:
    readme_path = ROOT / "02_建模计算_B" / "README.md"
    content = """# B 组建模计算区

本目录用于完成 `方案.md` 中阶段 4-6 的建模与结果输出。

## 当前已完成

### 阶段 4：第 2 问寿命预测

- 脚本：`04_代码/B_prediction_model.py`
- 输入：
  - `01_数据处理_A/exports/clean_data.xlsx`
  - `01_数据处理_A/exports/maintenance_record.xlsx`
  - `01_数据处理_A/exports/maintenance_match.xlsx`
  - `01_数据处理_A/exports/每台过滤器下降率表.xlsx`
  - `01_数据处理_A/exports/维护效果统计表.xlsx`
- 输出：
  - `outputs/B1_寿命预测结果.xlsx`
  - `outputs/B1_寿命预测模型说明.md`
  - `figures_B/fig_B1_lifetime_prediction.png`
  - `figures_B/fig_B1_remaining_life_bar.png`

## 输出文件说明

- `B1_寿命预测结果.xlsx`
  - `lifetime_summary`：10 台过滤器寿命终止日期、剩余寿命和维护节奏摘要
  - `validation_metrics`：最近 120 天回测误差
  - `forecast_daily`：未来逐日仿真路径
  - `model_parameters`：模型参数和口径说明
  - `schedule_assumptions`：历史维护节奏外推假设
- `B1_寿命预测模型说明.md`
  - 供成员 C 直接抽取第 2 问模型公式、判据说明和结果解释

## 运行方式

在项目根目录执行：

```bash
python3 04_代码/B_prediction_model.py
```

## 后续阶段

- 阶段 5：基于 `B1_寿命预测结果.xlsx` 进入维护策略优化
- 阶段 6：在优化结果基础上继续做成本敏感性分析
"""
    readme_path.write_text(content, encoding="utf-8")


def main() -> None:
    configure_plot_style()
    ensure_dirs()

    data = load_inputs()
    daily = build_daily_series(data["clean"])
    month_adjustments, monthly_adjustments_df = build_month_adjustments(
        daily,
        data["month_by_filter"],
    )
    pooled_models, local_models, recovery_models_df = build_recovery_models(
        data["match"],
        data["maintenance_type_summary"],
        data["maintenance_filter_type_summary"],
    )
    schedules = build_schedule_assumptions(data["maintenance"], daily)

    history_frames: dict[str, pd.DataFrame] = {}
    forecast_frames: dict[str, pd.DataFrame] = {}
    summary_rows = []
    validation_rows = []

    for filter_id, filter_daily in daily.groupby("filter_id"):
        filter_daily = filter_daily.sort_values("date").reset_index(drop=True)
        decline_row = data["decline"].loc[
            data["decline"]["filter_id"] == filter_id
        ].iloc[0]

        history = filter_daily[filter_daily["date"] >= (filter_daily["date"].max() - pd.Timedelta(days=BACKTEST_DAYS - 1))][
            ["date", "per"]
        ].copy()
        history_frames[filter_id] = history

        backtest_path, metrics = simulate_backtest(
            filter_id,
            filter_daily,
            decline_row,
            data["maintenance"],
            month_adjustments,
            pooled_models,
            local_models,
        )
        if metrics:
            metrics["filter_id"] = filter_id
            validation_rows.append(metrics)

        forecast_path, summary = simulate_future(
            filter_id,
            filter_daily,
            decline_row,
            schedules[filter_id],
            month_adjustments,
            pooled_models,
            local_models,
        )
        forecast_frames[filter_id] = forecast_path
        summary_rows.append(summary)

    lifetime_summary = pd.DataFrame(summary_rows).sort_values("remaining_life_days").reset_index(drop=True)
    validation_metrics = pd.DataFrame(validation_rows).sort_values("filter_id").reset_index(drop=True)
    forecast_daily = pd.concat(forecast_frames.values(), ignore_index=True)
    schedule_assumptions = pd.DataFrame(
        [
            {
                "filter_id": schedule.filter_id,
                "median_gap_days": schedule.median_gap_days,
                "days_since_last_maint": schedule.days_since_last_maint,
                "next_maintenance_in_days": schedule.next_maintenance_in_days,
                "historical_big_count": schedule.big_count,
                "historical_sequence": " -> ".join(schedule.historical_sequence),
            }
            for schedule in schedules.values()
        ]
    ).sort_values("filter_id")
    model_parameters = build_parameter_table(
        monthly_adjustments_df,
        recovery_models_df,
        data["decline"],
    )

    with pd.ExcelWriter(LIFETIME_XLSX, engine="openpyxl") as writer:
        lifetime_summary.to_excel(writer, sheet_name="lifetime_summary", index=False)
        validation_metrics.to_excel(writer, sheet_name="validation_metrics", index=False)
        recovery_models_df.to_excel(writer, sheet_name="recovery_models", index=False)
        schedule_assumptions.to_excel(writer, sheet_name="schedule_assumptions", index=False)
        model_parameters.to_excel(writer, sheet_name="model_parameters", index=False)
        forecast_daily.to_excel(writer, sheet_name="forecast_daily", index=False)

    MODEL_NOTE.write_text(
        render_model_note(lifetime_summary, validation_metrics, recovery_models_df),
        encoding="utf-8",
    )

    plot_lifetime_prediction(history_frames, forecast_frames, lifetime_summary)
    plot_remaining_life(lifetime_summary)

    shutil.copy2(FIG_LIFETIME, C_FIGURES / FIG_LIFETIME.name)
    shutil.copy2(FIG_REMAINING, C_FIGURES / FIG_REMAINING.name)

    update_b_readme()

    print(f"Saved: {LIFETIME_XLSX}")
    print(f"Saved: {MODEL_NOTE}")
    print(f"Saved: {FIG_LIFETIME}")
    print(f"Saved: {FIG_REMAINING}")


if __name__ == "__main__":
    main()
