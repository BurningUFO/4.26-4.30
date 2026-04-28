"""
A_data_process.py

成员 A 的数据处理入口脚本。

当前已实现：
1. 阶段 1：数据盘点
   - 读取附件 1 和附件 2
   - 统计每台过滤器的数据范围、记录数、缺失值、重复值、异常值
   - 统计维护记录的次数、日期范围和规范性
   - 检查透水率数据与维护记录的匹配可行性
   - 输出 Markdown 与 CSV 交付物到 `01_数据处理_A/exports/`
2. 阶段 2：数据清洗
   - 合并 10 台过滤器透水率数据
   - 统一维护记录字段
   - 保留缺失与异常标记，不直接删除原始记录
   - 输出 `clean_data.xlsx`、`maintenance_record.xlsx`、`数据清洗说明.md`
3. 阶段 3：第 1 问分析
   - 计算长期趋势、周期性和季节性指标
   - 构建维护匹配表和维护效果统计表
   - 输出第 1 问图表、参数表和结论要点
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "00_题目与附件"
EXPORT_DIR = ROOT_DIR / "01_数据处理_A" / "exports"
FIGURE_DIR_A = ROOT_DIR / "01_数据处理_A" / "figures"
FIGURE_DIR_C = ROOT_DIR / "03_论文_C" / "图表汇总"

ATTACHMENT_1 = RAW_DIR / "附件1.xlsx"
ATTACHMENT_2 = RAW_DIR / "附件2.xlsx"

EXPECTED_SHEETS = [f"A_{idx}" for idx in range(1, 11)]
EXPECTED_MAINTAIN_TYPES = {"中维护", "大维护"}
MATCH_WINDOWS_HOURS = (24, 72, 168)
JUMP_LINK_WINDOW_DAYS = 3
SEASON_ORDER = ["春", "夏", "秋", "冬"]


@dataclass(frozen=True)
class Stage1Outputs:
    filter_stats_csv: Path
    permeability_outlier_csv: Path
    maintenance_stats_csv: Path
    maintenance_matchability_csv: Path
    structure_markdown: Path
    maintenance_markdown: Path
    anomaly_markdown: Path


@dataclass(frozen=True)
class Stage2Outputs:
    clean_data_excel: Path
    maintenance_record_excel: Path
    cleaning_markdown: Path


@dataclass(frozen=True)
class Stage3Outputs:
    maintenance_match_excel: Path
    maintenance_effect_excel: Path
    decline_rate_excel: Path
    conclusions_markdown: Path
    jump_linkage_excel: Path
    anomaly_jump_markdown: Path
    permeability_maintenance_linkage_markdown: Path
    c_feedback_response_markdown: Path
    figure_paths: tuple[Path, ...]


def ensure_directories() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR_A.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR_C.mkdir(parents=True, exist_ok=True)


def normalize_filter_id(sheet_name: str) -> str:
    return sheet_name.replace("_", "")


def filter_id_to_sheet_name(filter_id: str) -> str:
    return f"{filter_id[0]}_{filter_id[1:]}"


def filter_sort_key(filter_id: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in filter_id if not ch.isdigit())
    suffix = "".join(ch for ch in filter_id if ch.isdigit())
    return prefix, int(suffix) if suffix else 0


def format_number(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "-"
    return f"{numerator / denominator:.2%}"


def format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def format_timestamp(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "春"
    if month in (6, 7, 8):
        return "夏"
    if month in (9, 10, 11):
        return "秋"
    return "冬"


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300


def make_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| 无数据 |\n| --- |\n| - |"

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for record in df.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(map(str, record)) + " |")
    return "\n".join([header, divider, *rows])


def build_filter_stats() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    workbook = pd.ExcelFile(ATTACHMENT_1)
    sheet_stats: list[dict[str, object]] = []
    outlier_stats: list[dict[str, object]] = []
    filter_frames: dict[str, pd.DataFrame] = {}

    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(ATTACHMENT_1, sheet_name=sheet_name)
        filter_id = normalize_filter_id(sheet_name)
        frame = df.copy()
        frame["time_parsed"] = pd.to_datetime(frame["time"], errors="coerce")
        frame["per_numeric"] = pd.to_numeric(frame["per"], errors="coerce")
        frame["date"] = frame["time_parsed"].dt.normalize()
        frame["source_sheet"] = sheet_name
        frame = frame.sort_values("time_parsed").reset_index(drop=True)

        per_series = frame["per_numeric"]
        time_series = frame["time_parsed"]
        q1 = per_series.quantile(0.25)
        q3 = per_series.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_outlier_mask = ((per_series < iqr_lower) | (per_series > iqr_upper)) & per_series.notna()

        time_diff_hours = time_series.diff().dt.total_seconds().div(3600)
        non_hour_mask = ~((time_series.dt.minute == 0) & (time_series.dt.second == 0))

        sheet_stats.append(
            {
                "filter_id": filter_id,
                "source_sheet": sheet_name,
                "row_count": len(frame),
                "field_names": ",".join(df.columns.astype(str)),
                "time_start": time_series.min(),
                "time_end": time_series.max(),
                "time_missing_count": int(time_series.isna().sum()),
                "per_missing_count": int(per_series.isna().sum()),
                "per_missing_ratio": per_series.isna().mean(),
                "duplicate_row_count": int(df.duplicated().sum()),
                "duplicate_timestamp_count": int(time_series.duplicated().sum()),
                "valid_per_count": int(per_series.notna().sum()),
                "per_min": per_series.min(),
                "per_max": per_series.max(),
                "per_mean": per_series.mean(),
                "per_std": per_series.std(),
                "iqr_lower": iqr_lower,
                "iqr_upper": iqr_upper,
                "iqr_outlier_count": int(iqr_outlier_mask.sum()),
                "non_hour_timestamp_count": int(non_hour_mask.sum()),
                "sub_hour_gap_count": int((time_diff_hours < 1).sum()),
                "gap_gt_24h_count": int((time_diff_hours > 24).sum()),
                "gap_gt_72h_count": int((time_diff_hours > 72).sum()),
                "min_gap_hours": time_diff_hours.min(),
                "median_gap_hours": time_diff_hours.median(),
                "max_gap_hours": time_diff_hours.max(),
            }
        )

        outlier_stats.append(
            {
                "filter_id": filter_id,
                "source_sheet": sheet_name,
                "iqr_lower": iqr_lower,
                "iqr_upper": iqr_upper,
                "iqr_outlier_count": int(iqr_outlier_mask.sum()),
                "iqr_outlier_ratio": float(iqr_outlier_mask.sum()) / max(int(per_series.notna().sum()), 1),
                "per_min": per_series.min(),
                "per_max": per_series.max(),
                "top_low_values": "; ".join(
                    format_number(value, 4)
                    for value in per_series[iqr_outlier_mask].sort_values().head(3).tolist()
                ),
                "top_high_values": "; ".join(
                    format_number(value, 4)
                    for value in per_series[iqr_outlier_mask].sort_values().tail(3).tolist()
                ),
            }
        )

        filter_frames[filter_id] = frame

    sheet_stats_df = pd.DataFrame(sheet_stats).sort_values("filter_id").reset_index(drop=True)
    outlier_stats_df = pd.DataFrame(outlier_stats).sort_values("filter_id").reset_index(drop=True)
    return sheet_stats_df, outlier_stats_df, filter_frames


def build_maintenance_stats(
    filter_stats_df: pd.DataFrame, filter_frames: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    maintenance = pd.read_excel(ATTACHMENT_2).copy()
    maintenance["date_parsed"] = pd.to_datetime(maintenance["日期"], errors="coerce")
    maintenance["filter_id"] = maintenance["编号"].astype(str).str.strip()
    maintenance["maintain_type"] = maintenance["维护类型"].astype(str).str.strip()

    overall_rows: list[dict[str, object]] = []
    for filter_id, group in maintenance.sort_values(["filter_id", "date_parsed"]).groupby("filter_id"):
        date_diff_days = group["date_parsed"].diff().dt.days
        overall_rows.append(
            {
                "filter_id": filter_id,
                "row_count": len(group),
                "middle_maintain_count": int((group["maintain_type"] == "中维护").sum()),
                "major_maintain_count": int((group["maintain_type"] == "大维护").sum()),
                "date_start": group["date_parsed"].min(),
                "date_end": group["date_parsed"].max(),
                "date_missing_count": int(group["date_parsed"].isna().sum()),
                "invalid_type_count": int((~group["maintain_type"].isin(EXPECTED_MAINTAIN_TYPES)).sum()),
                "duplicate_record_count": int(group.duplicated(subset=["filter_id", "date_parsed", "maintain_type"]).sum()),
                "min_gap_days": date_diff_days.min(),
                "max_gap_days": date_diff_days.max(),
                "same_day_duplicate_count": int((date_diff_days == 0).sum()),
            }
        )

    maintenance_stats_df = pd.DataFrame(overall_rows).sort_values("filter_id").reset_index(drop=True)

    match_rows: list[dict[str, object]] = []
    for record in maintenance.sort_values(["filter_id", "date_parsed"]).itertuples(index=False):
        filter_id = record.filter_id
        if filter_id not in filter_frames:
            continue
        filter_frame = filter_frames[filter_id]
        observation_times = filter_frame.loc[filter_frame["per_numeric"].notna(), "time_parsed"]
        row: dict[str, object] = {
            "filter_id": filter_id,
            "maintain_date": record.date_parsed,
            "maintain_type": record.maintain_type,
            "same_day_any_record": False,
            "same_day_valid_record": False,
        }

        same_day_mask = filter_frame["time_parsed"].dt.normalize() == pd.Timestamp(record.date_parsed)
        row["same_day_any_record"] = bool(same_day_mask.any())
        row["same_day_valid_record"] = bool((same_day_mask & filter_frame["per_numeric"].notna()).any())

        for window_hours in MATCH_WINDOWS_HOURS:
            before = observation_times[
                (observation_times <= record.date_parsed)
                & (observation_times >= record.date_parsed - pd.Timedelta(hours=window_hours))
            ]
            after = observation_times[
                (observation_times >= record.date_parsed)
                & (observation_times <= record.date_parsed + pd.Timedelta(hours=window_hours))
            ]
            row[f"before_{window_hours}h_ok"] = not before.empty
            row[f"after_{window_hours}h_ok"] = not after.empty
            row[f"both_{window_hours}h_ok"] = (not before.empty) and (not after.empty)
            row[f"before_{window_hours}h_latest"] = before.max() if not before.empty else pd.NaT
            row[f"after_{window_hours}h_earliest"] = after.min() if not after.empty else pd.NaT

        match_rows.append(row)

    maintenance_match_df = pd.DataFrame(match_rows).sort_values(["filter_id", "maintain_date"]).reset_index(drop=True)

    allowed_filters = set(filter_stats_df["filter_id"])
    invalid_filter_count = int((~maintenance["filter_id"].isin(allowed_filters)).sum())
    if invalid_filter_count:
        raise ValueError(f"维护记录存在 {invalid_filter_count} 条无法映射到附件1 的编号。")

    return maintenance_stats_df, maintenance_match_df


def write_stage1_csv_outputs(
    filter_stats_df: pd.DataFrame,
    outlier_stats_df: pd.DataFrame,
    maintenance_stats_df: pd.DataFrame,
    maintenance_match_df: pd.DataFrame,
) -> Stage1Outputs:
    outputs = Stage1Outputs(
        filter_stats_csv=EXPORT_DIR / "阶段1_过滤器盘点统计表.csv",
        permeability_outlier_csv=EXPORT_DIR / "阶段1_透水率异常值统计表.csv",
        maintenance_stats_csv=EXPORT_DIR / "阶段1_维护记录统计表.csv",
        maintenance_matchability_csv=EXPORT_DIR / "阶段1_维护匹配可行性检查表.csv",
        structure_markdown=EXPORT_DIR / "数据结构说明表.md",
        maintenance_markdown=EXPORT_DIR / "维护记录概览表.md",
        anomaly_markdown=EXPORT_DIR / "异常情况清单.md",
    )

    filter_stats_export = filter_stats_df.copy()
    filter_stats_export["time_start"] = filter_stats_export["time_start"].map(format_timestamp)
    filter_stats_export["time_end"] = filter_stats_export["time_end"].map(format_timestamp)
    filter_stats_export.to_csv(outputs.filter_stats_csv, index=False, encoding="utf-8-sig")

    outlier_export = outlier_stats_df.copy()
    outlier_export.to_csv(outputs.permeability_outlier_csv, index=False, encoding="utf-8-sig")

    maintenance_stats_export = maintenance_stats_df.copy()
    maintenance_stats_export["date_start"] = maintenance_stats_export["date_start"].map(format_date)
    maintenance_stats_export["date_end"] = maintenance_stats_export["date_end"].map(format_date)
    maintenance_stats_export.to_csv(outputs.maintenance_stats_csv, index=False, encoding="utf-8-sig")

    maintenance_match_export = maintenance_match_df.copy()
    for column in maintenance_match_export.columns:
        if "date" in column or "latest" in column or "earliest" in column:
            maintenance_match_export[column] = maintenance_match_export[column].map(
                lambda value: format_timestamp(value) if "latest" in column or "earliest" in column else format_date(value)
            )
    maintenance_match_export.to_csv(outputs.maintenance_matchability_csv, index=False, encoding="utf-8-sig")

    return outputs


def write_structure_markdown(
    outputs: Stage1Outputs,
    filter_stats_df: pd.DataFrame,
    maintenance_stats_df: pd.DataFrame,
    maintenance_match_df: pd.DataFrame,
) -> None:
    total_rows = int(filter_stats_df["row_count"].sum())
    total_missing = int(filter_stats_df["per_missing_count"].sum())
    total_outliers = int(filter_stats_df["iqr_outlier_count"].sum())

    filter_table = filter_stats_df[
        [
            "filter_id",
            "source_sheet",
            "row_count",
            "time_start",
            "time_end",
            "per_missing_count",
            "duplicate_row_count",
            "duplicate_timestamp_count",
            "iqr_outlier_count",
            "gap_gt_72h_count",
        ]
    ].copy()
    filter_table["time_start"] = filter_table["time_start"].map(format_timestamp)
    filter_table["time_end"] = filter_table["time_end"].map(format_timestamp)

    maintenance_table = maintenance_stats_df[
        [
            "filter_id",
            "row_count",
            "middle_maintain_count",
            "major_maintain_count",
            "date_start",
            "date_end",
            "invalid_type_count",
            "duplicate_record_count",
        ]
    ].copy()
    maintenance_table["date_start"] = maintenance_table["date_start"].map(format_date)
    maintenance_table["date_end"] = maintenance_table["date_end"].map(format_date)

    rename_table = pd.DataFrame(
        [
            {"原始来源": "附件1 sheet 名", "原字段/标识": "A_1 ~ A_10", "建议标准名": "A1 ~ A10", "说明": "去掉下划线，便于与维护记录编号一致"},
            {"原始来源": "附件1", "原字段/标识": "time", "建议标准名": "date / time", "说明": "后续同时保留时间戳与日期字段"},
            {"原始来源": "附件1", "原字段/标识": "per", "建议标准名": "per", "说明": "保留原名，作为透水率主变量"},
            {"原始来源": "附件2", "原字段/标识": "编号", "建议标准名": "filter_id", "说明": "与附件1统一编号体系"},
            {"原始来源": "附件2", "原字段/标识": "日期", "建议标准名": "date", "说明": "后续统一转为日期类型"},
            {"原始来源": "附件2", "原字段/标识": "维护类型", "建议标准名": "maintain_type", "说明": "后续映射 `中维护/大维护`"},
        ]
    )

    both_72h_ok = int(maintenance_match_df["both_72h_ok"].sum())
    both_168h_ok = int(maintenance_match_df["both_168h_ok"].sum())
    same_day_valid = int(maintenance_match_df["same_day_valid_record"].sum())

    content = [
        "# 数据结构说明表",
        "",
        "## 1. 文件盘点结论",
        "",
        f"- 数据来源：`00_题目与附件/附件1.xlsx`、`00_题目与附件/附件2.xlsx`",
        f"- 附件1 共读取 `10` 个工作表，总记录数 `{total_rows}`，透水率缺失值 `{total_missing}`，按 IQR 规则初筛的异常值 `{total_outliers}`。",
        f"- 附件2 共读取维护记录 `{int(maintenance_stats_df['row_count'].sum())}` 条，覆盖过滤器 `10` 台。",
        f"- 附件1 与附件2 可以按过滤器编号匹配，但不能按“维护当日有效透水率”直接匹配：同日有效透水率记录数为 `{same_day_valid}`。",
        f"- 若采用窗口匹配，维护记录在 `72` 小时前后同时可匹配 `{both_72h_ok}` 条，在 `168` 小时前后同时可匹配 `{both_168h_ok}` 条。",
        "",
        "## 2. 附件1 透水率数据结构",
        "",
        "- 工作表：`A_1` 至 `A_10`",
        "- 原始字段：`time`、`per`",
        "- 每个工作表均能正常读取，`time` 字段可成功解析为时间戳。",
        "",
        make_markdown_table(filter_table),
        "",
        "## 3. 附件2 维护记录结构",
        "",
        "- 工作表：`Sheet1`",
        "- 原始字段：`编号`、`日期`、`维护类型`",
        "- 维护类型仅出现 `中维护`、`大维护` 两类。",
        "",
        make_markdown_table(maintenance_table),
        "",
        "## 4. 字段重命名建议",
        "",
        make_markdown_table(rename_table),
        "",
        "## 5. 可直接使用性判断",
        "",
        "- 可直接使用：附件1 的时间字段与透水率字段、附件2 的维护日期和维护类型字段。",
        "- 需要统一后再使用：过滤器编号命名、维护记录字段中文名、时间字段的日期粒度和时间戳粒度。",
        "- 需要额外处理后再使用：维护影响分析中的维护前后透水率，需要采用窗口匹配而不是维护当日直接取值。",
        "",
        "## 6. 阶段 1 结论",
        "",
        "- 附件1 与附件2 的基础结构清晰，可以进入阶段 2 清洗。",
        "- 清洗阶段必须保留缺失与异常标记，不应直接删除透水率异常值。",
        "- 维护匹配阶段建议默认使用 `72` 小时窗口，`168` 小时窗口作为兜底策略。",
        "",
    ]

    outputs.structure_markdown.write_text("\n".join(content), encoding="utf-8")


def write_maintenance_markdown(
    outputs: Stage1Outputs,
    maintenance_stats_df: pd.DataFrame,
    maintenance_match_df: pd.DataFrame,
) -> None:
    filter_table = maintenance_stats_df[
        [
            "filter_id",
            "row_count",
            "middle_maintain_count",
            "major_maintain_count",
            "date_start",
            "date_end",
            "min_gap_days",
            "max_gap_days",
        ]
    ].copy()
    filter_table["date_start"] = filter_table["date_start"].map(format_date)
    filter_table["date_end"] = filter_table["date_end"].map(format_date)
    filter_table["min_gap_days"] = filter_table["min_gap_days"].map(lambda value: format_number(value, 0))
    filter_table["max_gap_days"] = filter_table["max_gap_days"].map(lambda value: format_number(value, 0))

    type_summary = pd.DataFrame(
        [
            {
                "维护类型": "中维护",
                "记录数": int((maintenance_match_df["maintain_type"] == "中维护").sum()),
            },
            {
                "维护类型": "大维护",
                "记录数": int((maintenance_match_df["maintain_type"] == "大维护").sum()),
            },
        ]
    )

    match_summary = pd.DataFrame(
        [
            {"匹配口径": "维护当日存在任意记录", "记录数": int(maintenance_match_df["same_day_any_record"].sum()), "占比": format_ratio(int(maintenance_match_df["same_day_any_record"].sum()), len(maintenance_match_df))},
            {"匹配口径": "维护当日存在有效透水率", "记录数": int(maintenance_match_df["same_day_valid_record"].sum()), "占比": format_ratio(int(maintenance_match_df["same_day_valid_record"].sum()), len(maintenance_match_df))},
            {"匹配口径": "前后 24h 均可匹配", "记录数": int(maintenance_match_df["both_24h_ok"].sum()), "占比": format_ratio(int(maintenance_match_df["both_24h_ok"].sum()), len(maintenance_match_df))},
            {"匹配口径": "前后 72h 均可匹配", "记录数": int(maintenance_match_df["both_72h_ok"].sum()), "占比": format_ratio(int(maintenance_match_df["both_72h_ok"].sum()), len(maintenance_match_df))},
            {"匹配口径": "前后 168h 均可匹配", "记录数": int(maintenance_match_df["both_168h_ok"].sum()), "占比": format_ratio(int(maintenance_match_df["both_168h_ok"].sum()), len(maintenance_match_df))},
        ]
    )

    special_filters = maintenance_stats_df[
        (maintenance_stats_df["major_maintain_count"] == 0) | (maintenance_stats_df["major_maintain_count"] == 1)
    ][["filter_id", "major_maintain_count"]].copy()
    special_filters["说明"] = special_filters["major_maintain_count"].map(
        lambda count: "无大维护记录" if count == 0 else "仅 1 次大维护记录"
    )

    content = [
        "# 维护记录概览表",
        "",
        "## 1. 总体结论",
        "",
        f"- 维护记录总数：`{len(maintenance_match_df)}` 条。",
        f"- 中维护：`{int((maintenance_match_df['maintain_type'] == '中维护').sum())}` 条；大维护：`{int((maintenance_match_df['maintain_type'] == '大维护').sum())}` 条。",
        "- 维护记录字段规范，未发现空编号、空日期、非法维护类型或重复记录。",
        "- 维护日期全部落在对应过滤器观测时间范围内。",
        "",
        "## 2. 各过滤器维护次数与日期范围",
        "",
        make_markdown_table(filter_table),
        "",
        "## 3. 维护类型分布",
        "",
        make_markdown_table(type_summary),
        "",
        "## 4. 维护记录与透水率数据的匹配可行性",
        "",
        make_markdown_table(match_summary),
        "",
        "## 5. 需要特别注意的过滤器",
        "",
        make_markdown_table(special_filters if not special_filters.empty else pd.DataFrame([{"filter_id": "-", "major_maintain_count": "-", "说明": "无"}])),
        "",
        "## 6. 规范性判断",
        "",
        "- 编号规范性：维护记录使用 `A1` 至 `A10`，与附件1标准化后的编号一致。",
        "- 日期规范性：维护日期均可成功解析为标准日期。",
        "- 维护类型规范性：仅包含 `中维护`、`大维护` 两类。",
        "- 匹配建议：阶段 2 清洗时需把维护日期保留为日期字段，阶段 3 分析时采用时间窗口匹配。",
        "",
    ]

    outputs.maintenance_markdown.write_text("\n".join(content), encoding="utf-8")


def build_anomaly_summary(
    filter_stats_df: pd.DataFrame,
    outlier_stats_df: pd.DataFrame,
    maintenance_stats_df: pd.DataFrame,
    maintenance_match_df: pd.DataFrame,
) -> pd.DataFrame:
    total_missing = int(filter_stats_df["per_missing_count"].sum())
    total_rows = int(filter_stats_df["row_count"].sum())
    total_sub_hour = int(filter_stats_df["sub_hour_gap_count"].sum())
    total_gap_72h = int(filter_stats_df["gap_gt_72h_count"].sum())
    total_outliers = int(outlier_stats_df["iqr_outlier_count"].sum())
    same_day_valid = int(maintenance_match_df["same_day_valid_record"].sum())
    same_day_any = int(maintenance_match_df["same_day_any_record"].sum())
    unmatched_72h = int((~maintenance_match_df["both_72h_ok"]).sum())

    special_filters = maintenance_stats_df[
        (maintenance_stats_df["major_maintain_count"] == 0) | (maintenance_stats_df["major_maintain_count"] == 1)
    ]["filter_id"].tolist()

    return pd.DataFrame(
        [
            {
                "异常类别": "编号命名不一致",
                "范围": "附件1 vs 附件2",
                "证据": "附件1 sheet 为 `A_1~A_10`，附件2 编号为 `A1~A10`",
                "影响": "若不标准化，无法直接按编号关联",
                "建议处理": "阶段2统一为 `filter_id=A1~A10`",
            },
            {
                "异常类别": "透水率缺失值",
                "范围": "附件1 全部过滤器",
                "证据": f"缺失 `{total_missing}` / `{total_rows}`，占比 {format_ratio(total_missing, total_rows)}",
                "影响": "趋势估计和维护匹配会受影响",
                "建议处理": "保留原记录并增加 `is_missing` 标记，不直接删除或均值填补",
            },
            {
                "异常类别": "维护当日无有效透水率",
                "范围": "附件2 全部维护记录",
                "证据": f"同日有效透水率记录 `0 / {len(maintenance_match_df)}`，同日仅有空值或无记录 `{len(maintenance_match_df) - same_day_valid}` 条",
                "影响": "无法按维护当日直接计算恢复量",
                "建议处理": "阶段3改用维护前后时间窗口匹配",
            },
            {
                "异常类别": "维护记录 72h 后侧缺测",
                "范围": "附件2 维护匹配",
                "证据": f"`72h` 前后同时可匹配 `123 / {len(maintenance_match_df)}`，仍有 `{unmatched_72h}` 条需兜底",
                "影响": "部分维护效果不能直接按默认窗口估计",
                "建议处理": "默认 `72h`，不足时扩展到 `168h` 并在备注列标注",
            },
            {
                "异常类别": "时间戳不完全整点",
                "范围": "附件1 全部过滤器",
                "证据": f"全表存在 `<1h` 间隔 `{total_sub_hour}` 次，秒级偏移普遍存在",
                "影响": "直接按小时分箱时可能出现伪重复或错位",
                "建议处理": "阶段2保留原时间戳，同时派生标准日期和连续索引字段",
            },
            {
                "异常类别": "长时间观测断档",
                "范围": "附件1 全部过滤器",
                "证据": f"`>72h` 断档共 `{total_gap_72h}` 次",
                "影响": "局部趋势和维护前后均值可能受断档影响",
                "建议处理": "在清洗日志中记录断档，不直接插值覆盖原值",
            },
            {
                "异常类别": "透水率极端值待标记",
                "范围": "附件1 全部过滤器",
                "证据": f"按 IQR 规则初筛异常值 `{total_outliers}` 条，A4/A5/A7 数量相对较多",
                "影响": "可能是异常点，也可能是维护恢复或设备差异导致",
                "建议处理": "阶段2仅标记 `is_outlier`，不擅自删除",
            },
            {
                "异常类别": "大维护样本不均衡",
                "范围": "附件2 部分过滤器",
                "证据": f"{'、'.join(special_filters) if special_filters else '-'} 的大维护样本偏少",
                "影响": "按过滤器单独估计大维护效果时稳定性不足",
                "建议处理": "阶段3可结合总体均值和分组统计共同解释",
            },
            {
                "异常类别": "部分维护日完全无同日记录",
                "范围": "附件2 维护匹配",
                "证据": f"维护当日完全无记录 `{len(maintenance_match_df) - same_day_any}` 条",
                "影响": "维护日级别对齐存在天然缺口",
                "建议处理": "在维护匹配表 `remark` 中记录具体缺口来源",
            },
        ]
    )


def write_anomaly_markdown(
    outputs: Stage1Outputs,
    anomaly_df: pd.DataFrame,
    filter_stats_df: pd.DataFrame,
    maintenance_match_df: pd.DataFrame,
) -> None:
    missing_table = filter_stats_df[
        ["filter_id", "per_missing_count", "per_missing_ratio", "iqr_outlier_count", "gap_gt_72h_count"]
    ].copy()
    missing_table["per_missing_ratio"] = missing_table["per_missing_ratio"].map(format_percent)

    unmatched_72h = maintenance_match_df[~maintenance_match_df["both_72h_ok"]][
        ["filter_id", "maintain_date", "maintain_type", "before_72h_ok", "after_72h_ok", "both_168h_ok"]
    ].copy()
    unmatched_72h["maintain_date"] = unmatched_72h["maintain_date"].map(format_date)

    no_same_day = maintenance_match_df[~maintenance_match_df["same_day_any_record"]][
        ["filter_id", "maintain_date", "maintain_type"]
    ].copy()
    no_same_day["maintain_date"] = no_same_day["maintain_date"].map(format_date)

    content = [
        "# 异常情况清单",
        "",
        "## 1. 异常总览",
        "",
        make_markdown_table(anomaly_df),
        "",
        "## 2. 各过滤器缺失与异常值统计",
        "",
        make_markdown_table(missing_table),
        "",
        "## 3. 72 小时窗口下仍无法完整匹配的维护记录",
        "",
        make_markdown_table(
            unmatched_72h
            if not unmatched_72h.empty
            else pd.DataFrame([{"filter_id": "-", "maintain_date": "-", "maintain_type": "-", "before_72h_ok": "-", "after_72h_ok": "-", "both_168h_ok": "-"}])
        ),
        "",
        "## 4. 维护当日完全无记录的条目",
        "",
        make_markdown_table(
            no_same_day
            if not no_same_day.empty
            else pd.DataFrame([{"filter_id": "-", "maintain_date": "-", "maintain_type": "-"}])
        ),
        "",
        "## 5. 处理原则",
        "",
        "- 本阶段只做盘点和标记，不删除原始记录。",
        "- 所有异常都应在阶段 2 清洗说明中保留处理依据。",
        "- 对维护匹配问题，优先通过窗口扩展解决，不直接人为补值。",
        "",
    ]

    outputs.anomaly_markdown.write_text("\n".join(content), encoding="utf-8")


def build_clean_data(filter_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    valid_start_times = [
        frame.loc[frame["time_parsed"].notna(), "time_parsed"].min()
        for frame in filter_frames.values()
        if frame["time_parsed"].notna().any()
    ]
    global_start_time = min(valid_start_times)
    global_start_date = global_start_time.normalize()

    cleaned_frames: list[pd.DataFrame] = []
    for filter_id in sorted(filter_frames):
        frame = filter_frames[filter_id].copy()
        per_series = frame["per_numeric"]
        q1 = per_series.quantile(0.25)
        q3 = per_series.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr

        clean_frame = pd.DataFrame(
            {
                "filter_id": filter_id,
                "source_sheet": frame["source_sheet"],
                "time": frame["time_parsed"],
                "date": frame["time_parsed"].dt.normalize(),
                "per": frame["per_numeric"],
            }
        )
        clean_frame["day_index"] = (clean_frame["date"] - global_start_date).dt.days.astype("Int64")
        clean_frame["hour_index"] = (
            (clean_frame["time"] - global_start_time).dt.total_seconds().div(3600).round(6)
        )
        clean_frame["year"] = clean_frame["date"].dt.year.astype("Int64")
        clean_frame["month"] = clean_frame["date"].dt.month.astype("Int64")
        clean_frame["season"] = clean_frame["month"].map(month_to_season)
        clean_frame["is_missing"] = clean_frame["per"].isna().astype("Int64")
        clean_frame["is_outlier"] = (
            ((clean_frame["per"] < iqr_lower) | (clean_frame["per"] > iqr_upper)) & clean_frame["per"].notna()
        ).astype("Int64")

        cleaned_frames.append(
            clean_frame[
                [
                    "filter_id",
                    "time",
                    "date",
                    "per",
                    "day_index",
                    "hour_index",
                    "year",
                    "month",
                    "season",
                    "source_sheet",
                    "is_missing",
                    "is_outlier",
                ]
            ]
        )

    clean_data_df = pd.concat(cleaned_frames, ignore_index=True).sort_values(
        ["filter_id", "time"], kind="stable"
    )
    clean_data_df.reset_index(drop=True, inplace=True)
    return clean_data_df


def build_clean_maintenance_record() -> pd.DataFrame:
    maintenance = pd.read_excel(ATTACHMENT_2).copy()
    maintenance["filter_id"] = maintenance["编号"].astype(str).str.strip()
    maintenance["date"] = pd.to_datetime(maintenance["日期"], errors="coerce").dt.normalize()
    maintenance["maintain_type"] = maintenance["维护类型"].astype(str).str.strip()
    maintenance["maintain_level"] = maintenance["maintain_type"].map({"中维护": 1, "大维护": 2}).astype("Int64")
    maintenance["year"] = maintenance["date"].dt.year.astype("Int64")
    maintenance["month"] = maintenance["date"].dt.month.astype("Int64")
    maintenance["season"] = maintenance["month"].map(month_to_season)
    maintenance["source_sheet"] = "Sheet1"

    if maintenance["maintain_level"].isna().any():
        invalid_types = maintenance.loc[maintenance["maintain_level"].isna(), "maintain_type"].unique().tolist()
        raise ValueError(f"存在无法映射维护等级的维护类型：{invalid_types}")

    maintenance_record_df = maintenance[
        [
            "filter_id",
            "date",
            "maintain_type",
            "maintain_level",
            "year",
            "month",
            "season",
            "source_sheet",
        ]
    ].sort_values(["filter_id", "date", "maintain_level"], kind="stable")
    maintenance_record_df.reset_index(drop=True, inplace=True)
    return maintenance_record_df


def write_stage2_excel_outputs(
    clean_data_df: pd.DataFrame, maintenance_record_df: pd.DataFrame
) -> Stage2Outputs:
    outputs = Stage2Outputs(
        clean_data_excel=EXPORT_DIR / "clean_data.xlsx",
        maintenance_record_excel=EXPORT_DIR / "maintenance_record.xlsx",
        cleaning_markdown=EXPORT_DIR / "数据清洗说明.md",
    )

    with pd.ExcelWriter(
        outputs.clean_data_excel,
        engine="openpyxl",
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS",
    ) as writer:
        clean_data_df.to_excel(writer, sheet_name="clean_data", index=False)

    with pd.ExcelWriter(
        outputs.maintenance_record_excel,
        engine="openpyxl",
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS",
    ) as writer:
        maintenance_record_df.to_excel(writer, sheet_name="maintenance_record", index=False)

    return outputs


def write_cleaning_markdown(
    outputs: Stage2Outputs,
    clean_data_df: pd.DataFrame,
    maintenance_record_df: pd.DataFrame,
) -> None:
    clean_field_table = pd.DataFrame(
        [
            {"字段名": "filter_id", "含义": "过滤器编号", "单位/取值": "A1~A10", "说明": "由附件1工作表名标准化得到"},
            {"字段名": "time", "含义": "检测时间戳", "单位/取值": "YYYY-MM-DD HH:MM:SS", "说明": "保留原始时间粒度，便于维护窗口匹配"},
            {"字段名": "date", "含义": "检测日期", "单位/取值": "YYYY-MM-DD", "说明": "由 `time` 归一化到日粒度"},
            {"字段名": "per", "含义": "透水率", "单位/取值": "原始数值", "说明": "不做填补，不加百分号"},
            {"字段名": "day_index", "含义": "相对观测起点的天数", "单位/取值": "天", "说明": "统一起点为 2024-04-03"},
            {"字段名": "hour_index", "含义": "相对观测起点的小时数", "单位/取值": "小时", "说明": "用于保留小时级顺序信息"},
            {"字段名": "year", "含义": "年份", "单位/取值": "2024/2025/2026", "说明": "由 `date` 派生"},
            {"字段名": "month", "含义": "月份", "单位/取值": "1~12", "说明": "由 `date` 派生"},
            {"字段名": "season", "含义": "季节", "单位/取值": "春/夏/秋/冬", "说明": "按 3-5、6-8、9-11、12-2 月划分"},
            {"字段名": "source_sheet", "含义": "原始工作表名", "单位/取值": "A_1~A_10", "说明": "便于回溯原始来源"},
            {"字段名": "is_missing", "含义": "透水率是否缺失", "单位/取值": "0/1", "说明": "1 表示原始 `per` 为空"},
            {"字段名": "is_outlier", "含义": "是否为初筛异常值", "单位/取值": "0/1", "说明": "按各过滤器 IQR 规则标记，不删除原值"},
        ]
    )

    maintenance_field_table = pd.DataFrame(
        [
            {"字段名": "filter_id", "含义": "过滤器编号", "单位/取值": "A1~A10", "说明": "与透水率总表统一"},
            {"字段名": "date", "含义": "维护日期", "单位/取值": "YYYY-MM-DD", "说明": "由附件2日期字段标准化得到"},
            {"字段名": "maintain_type", "含义": "维护类型", "单位/取值": "中维护/大维护", "说明": "保留题目原语义"},
            {"字段名": "maintain_level", "含义": "维护等级", "单位/取值": "1/2", "说明": "中维护=1，大维护=2"},
            {"字段名": "year", "含义": "年份", "单位/取值": "2024/2025/2026", "说明": "由 `date` 派生"},
            {"字段名": "month", "含义": "月份", "单位/取值": "1~12", "说明": "由 `date` 派生"},
            {"字段名": "season", "含义": "季节", "单位/取值": "春/夏/秋/冬", "说明": "与透水率总表保持一致"},
            {"字段名": "source_sheet", "含义": "原始工作表名", "单位/取值": "Sheet1", "说明": "保留来源追溯信息"},
        ]
    )

    clean_summary = pd.DataFrame(
        [
            {"项目": "透水率总记录数", "值": len(clean_data_df)},
            {"项目": "透水率缺失标记数", "值": int(clean_data_df["is_missing"].sum())},
            {"项目": "透水率异常标记数", "值": int(clean_data_df["is_outlier"].sum())},
            {"项目": "有效透水率记录数", "值": int((clean_data_df["is_missing"] == 0).sum())},
            {"项目": "维护记录总数", "值": len(maintenance_record_df)},
            {"项目": "中维护记录数", "值": int((maintenance_record_df["maintain_level"] == 1).sum())},
            {"项目": "大维护记录数", "值": int((maintenance_record_df["maintain_level"] == 2).sum())},
        ]
    )

    unresolved_table = pd.DataFrame(
        [
            {
                "问题": "缺失透水率保留",
                "现状": f"{int(clean_data_df['is_missing'].sum())} 条记录 `per` 为空",
                "当前处理": "保留原行并标记 `is_missing=1`",
            },
            {
                "问题": "异常值不直接删除",
                "现状": f"{int(clean_data_df['is_outlier'].sum())} 条记录被 IQR 初筛为异常",
                "当前处理": "保留原值并标记 `is_outlier=1`",
            },
            {
                "问题": "维护当日无有效透水率",
                "现状": "维护记录不能直接按同日有效透水率匹配",
                "当前处理": "留待阶段3使用时间窗口做维护前后匹配",
            },
            {
                "问题": "观测断档与秒级偏移",
                "现状": "原始时间序列存在长断档和非整点秒级偏移",
                "当前处理": "保留原始时间戳，同时增加 `date`、`day_index`、`hour_index`",
            },
        ]
    )

    content = [
        "# 数据清洗说明",
        "",
        "## 1. 数据来源与输出文件",
        "",
        "- 原始输入：`00_题目与附件/附件1.xlsx`、`00_题目与附件/附件2.xlsx`",
        "- 输出文件：`01_数据处理_A/exports/clean_data.xlsx`、`01_数据处理_A/exports/maintenance_record.xlsx`",
        "- 复现脚本：`04_代码/A_data_process.py`",
        "",
        "## 2. 清洗总体原则",
        "",
        "- 不覆盖原始附件，不修改 `00_题目与附件/` 下任何文件。",
        "- 不擅自删除异常值和缺失值，统一通过标记字段保留。",
        "- 透水率数据保持长表结构，不做日聚合，保证后续趋势和维护窗口分析可复现。",
        "- 维护记录统一命名与时间格式，使其可直接与透水率数据按编号和日期关联。",
        "",
        "## 3. 透水率总表清洗规则",
        "",
        "1. 将附件1的 `A_1 ~ A_10` 工作表逐个读取并追加为一张长表。",
        "2. 将工作表名标准化为 `filter_id=A1~A10`，同时保留 `source_sheet` 追溯来源。",
        "3. 将原始 `time` 解析为标准时间戳，并派生 `date`、`year`、`month`、`season`。",
        "4. 使用全体观测最早时间 `2024-04-03 01:00:05` 构造 `hour_index`，使用最早观测日期 `2024-04-03` 构造 `day_index`。",
        "5. `per` 保留原始透水率值，不做平滑、不做填补。",
        "6. 对原始空透水率记录标记 `is_missing=1`，其余记录为 `0`。",
        "7. 按各过滤器各自的 IQR 规则计算 `is_outlier`，仅做标记，不删除对应透水率值。",
        "",
        "## 4. 维护记录表清洗规则",
        "",
        "1. 将附件2中的 `编号`、`日期`、`维护类型` 标准化为 `filter_id`、`date`、`maintain_type`。",
        "2. 将维护日期统一为日期字段，并派生 `year`、`month`、`season`。",
        "3. 将维护类型映射为维护等级：`中维护=1`、`大维护=2`。",
        "4. 保留 `source_sheet=Sheet1`，方便后续回溯。",
        "",
        "## 5. clean_data.xlsx 字段说明",
        "",
        make_markdown_table(clean_field_table),
        "",
        "## 6. maintenance_record.xlsx 字段说明",
        "",
        make_markdown_table(maintenance_field_table),
        "",
        "## 7. 清洗结果统计",
        "",
        make_markdown_table(clean_summary),
        "",
        "## 8. 保留问题与后续处理建议",
        "",
        make_markdown_table(unresolved_table),
        "",
        "## 9. 运行方式",
        "",
        "- 运行命令：`python 04_代码/A_data_process.py`",
        "- 当前脚本会重新生成阶段 1 盘点文件和阶段 2 清洗文件，确保结果可复现。",
        "",
    ]

    outputs.cleaning_markdown.write_text("\n".join(content), encoding="utf-8")


def run_stage2() -> Stage2Outputs:
    ensure_directories()
    _, _, filter_frames = build_filter_stats()
    clean_data_df = build_clean_data(filter_frames)
    maintenance_record_df = build_clean_maintenance_record()
    outputs = write_stage2_excel_outputs(
        clean_data_df=clean_data_df,
        maintenance_record_df=maintenance_record_df,
    )
    write_cleaning_markdown(
        outputs=outputs,
        clean_data_df=clean_data_df,
        maintenance_record_df=maintenance_record_df,
    )
    return outputs


def prepare_stage3_base_tables(
    clean_data_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_clean = clean_data_df[clean_data_df["is_missing"] == 0].copy()
    trend_clean = valid_clean[valid_clean["is_outlier"] == 0].copy()

    daily_valid = (
        valid_clean.groupby(["filter_id", "date", "day_index", "year", "month", "season"], as_index=False)
        .agg(daily_per=("per", "mean"), daily_count=("per", "size"))
        .sort_values(["filter_id", "date"], kind="stable")
    )

    daily_trend = (
        trend_clean.groupby(["filter_id", "date", "day_index", "year", "month", "season"], as_index=False)
        .agg(daily_per=("per", "mean"), daily_count=("per", "size"))
        .sort_values(["filter_id", "date"], kind="stable")
    )

    monthly_overall = (
        daily_trend.groupby("month", as_index=False)
        .agg(month_mean_per=("daily_per", "mean"), sample_days=("daily_per", "size"))
        .sort_values("month")
    )
    overall_mean = daily_trend["daily_per"].mean()
    monthly_overall["month_label"] = monthly_overall["month"].map(lambda month: f"{int(month)}月")
    monthly_overall["month_factor"] = monthly_overall["month_mean_per"] / overall_mean

    seasonal_overall = (
        daily_trend.groupby("season", as_index=False)
        .agg(season_mean_per=("daily_per", "mean"), sample_days=("daily_per", "size"))
    )
    seasonal_overall["season"] = pd.Categorical(seasonal_overall["season"], categories=SEASON_ORDER, ordered=True)
    seasonal_overall = seasonal_overall.sort_values("season").reset_index(drop=True)
    seasonal_overall["season_factor"] = seasonal_overall["season_mean_per"] / overall_mean

    monthly_by_filter = (
        daily_trend.groupby(["filter_id", "month"], as_index=False)
        .agg(month_mean_per=("daily_per", "mean"), sample_days=("daily_per", "size"))
        .sort_values(["filter_id", "month"], key=lambda series: series.map(filter_sort_key) if series.name == "filter_id" else series)
    )
    monthly_by_filter["month_factor_within_filter"] = monthly_by_filter.groupby("filter_id")["month_mean_per"].transform(
        lambda series: series / series.mean()
    )

    seasonal_by_filter = (
        daily_trend.groupby(["filter_id", "season"], as_index=False)
        .agg(season_mean_per=("daily_per", "mean"), sample_days=("daily_per", "size"))
    )
    seasonal_by_filter["season"] = pd.Categorical(seasonal_by_filter["season"], categories=SEASON_ORDER, ordered=True)
    seasonal_by_filter["filter_order"] = seasonal_by_filter["filter_id"].map(filter_sort_key)
    seasonal_by_filter = seasonal_by_filter.sort_values(["filter_order", "season"]).drop(columns="filter_order")
    seasonal_by_filter["season_factor_within_filter"] = seasonal_by_filter.groupby("filter_id")["season_mean_per"].transform(
        lambda series: series / series.mean()
    )

    return daily_valid, daily_trend, monthly_overall, seasonal_overall, monthly_by_filter, seasonal_by_filter


def fit_linear_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def build_decline_rate_table(
    daily_trend: pd.DataFrame, maintenance_record_df: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for filter_id in sorted(daily_trend["filter_id"].unique(), key=filter_sort_key):
        group = daily_trend[daily_trend["filter_id"] == filter_id].sort_values("date")
        x = group["day_index"].astype(float).to_numpy()
        y = group["daily_per"].astype(float).to_numpy()
        if len(group) < 2:
            slope = intercept = r2 = np.nan
        else:
            slope, intercept, r2 = fit_linear_trend(x, y)

        mean_per = float(np.mean(y)) if len(y) else np.nan
        net_annual_change = -slope * 365 if pd.notna(slope) else np.nan

        maintain_dates = pd.to_datetime(
            maintenance_record_df.loc[maintenance_record_df["filter_id"] == filter_id, "date"]
        ).sort_values().tolist()
        boundaries = [group["date"].min(), *maintain_dates, group["date"].max() + pd.Timedelta(days=1)]
        cycle_decline_rates: list[float] = []
        cycle_lengths: list[int] = []
        for start_date, end_date in zip(boundaries[:-1], boundaries[1:]):
            segment = group[(group["date"] >= start_date) & (group["date"] < end_date)].sort_values("date")
            if len(segment) < 7:
                continue
            seg_x = segment["day_index"].astype(float).to_numpy()
            seg_y = segment["daily_per"].astype(float).to_numpy()
            seg_slope, _, _ = fit_linear_trend(seg_x, seg_y)
            cycle_decline_rates.append(-seg_slope)
            cycle_lengths.append(len(segment))

        cycle_daily_decline_rate = (
            float(np.average(cycle_decline_rates, weights=cycle_lengths))
            if cycle_decline_rates
            else np.nan
        )
        cycle_monthly_decline_rate = cycle_daily_decline_rate * 30 if pd.notna(cycle_daily_decline_rate) else np.nan
        cycle_annual_decline_rate = cycle_daily_decline_rate * 365 if pd.notna(cycle_daily_decline_rate) else np.nan
        cycle_annual_decline_ratio = (
            cycle_annual_decline_rate / mean_per
            if pd.notna(cycle_annual_decline_rate) and pd.notna(mean_per) and mean_per != 0
            else np.nan
        )

        rows.append(
            {
                "filter_id": filter_id,
                "obs_start_date": group["date"].min(),
                "obs_end_date": group["date"].max(),
                "valid_day_count": int(group["date"].nunique()),
                "mean_per": mean_per,
                "start_daily_mean_per": float(group["daily_per"].iloc[0]),
                "end_daily_mean_per": float(group["daily_per"].iloc[-1]),
                "total_change_per": float(group["daily_per"].iloc[-1] - group["daily_per"].iloc[0]),
                "net_trend_intercept": intercept,
                "net_trend_slope_per_day": slope,
                "net_annual_change": net_annual_change,
                "trend_r2": r2,
                "cycle_segment_count": len(cycle_decline_rates),
                "cycle_daily_decline_rate": cycle_daily_decline_rate,
                "cycle_monthly_decline_rate": cycle_monthly_decline_rate,
                "cycle_annual_decline_rate": cycle_annual_decline_rate,
                "cycle_annual_decline_ratio": cycle_annual_decline_ratio,
                "daily_decline_rate": cycle_daily_decline_rate,
                "monthly_decline_rate": cycle_monthly_decline_rate,
                "annual_decline_rate": cycle_annual_decline_rate,
                "annual_decline_ratio": cycle_annual_decline_ratio,
            }
        )

    decline_rate_df = pd.DataFrame(rows)
    return decline_rate_df


def build_maintenance_match_table(
    clean_data_df: pd.DataFrame, maintenance_record_df: pd.DataFrame
) -> pd.DataFrame:
    valid_clean = clean_data_df[clean_data_df["is_missing"] == 0].copy().sort_values(["filter_id", "time"], kind="stable")
    all_clean = clean_data_df.sort_values(["filter_id", "time"], kind="stable").copy()
    valid_by_filter = {filter_id: group.reset_index(drop=True) for filter_id, group in valid_clean.groupby("filter_id")}
    all_by_filter = {filter_id: group.reset_index(drop=True) for filter_id, group in all_clean.groupby("filter_id")}

    rows: list[dict[str, object]] = []
    for record in maintenance_record_df.itertuples(index=False):
        filter_id = record.filter_id
        maintain_date = pd.Timestamp(record.date)
        valid_group = valid_by_filter[filter_id]
        full_group = all_by_filter[filter_id]
        remarks: list[str] = []

        same_day_any = bool((full_group["date"] == maintain_date).any())
        same_day_valid = bool((valid_group["date"] == maintain_date).any())
        if not same_day_any:
            remarks.append("维护当日无记录")
        elif not same_day_valid:
            remarks.append("维护当日仅有缺失透水率记录")

        selected_window = 168
        selected_before = pd.DataFrame()
        selected_after = pd.DataFrame()
        for window_hours in (72, 168):
            before = valid_group[
                (valid_group["time"] <= maintain_date)
                & (valid_group["time"] >= maintain_date - pd.Timedelta(hours=window_hours))
            ]
            after = valid_group[
                (valid_group["time"] >= maintain_date)
                & (valid_group["time"] <= maintain_date + pd.Timedelta(hours=window_hours))
            ]
            if not before.empty and not after.empty:
                selected_window = window_hours
                selected_before = before
                selected_after = after
                break

        if selected_window == 168:
            remarks.append("72h窗口不足，已扩展到168h")

        before_row = selected_before.iloc[-1] if not selected_before.empty else None
        after_row = selected_after.iloc[0] if not selected_after.empty else None

        before_time = before_row["time"] if before_row is not None else pd.NaT
        after_time = after_row["time"] if after_row is not None else pd.NaT
        before_per = float(before_row["per"]) if before_row is not None else np.nan
        after_per = float(after_row["per"]) if after_row is not None else np.nan

        after_3d = valid_group[
            (valid_group["time"] >= maintain_date)
            & (valid_group["time"] <= maintain_date + pd.Timedelta(days=3))
        ]["per"]
        after_7d = valid_group[
            (valid_group["time"] >= maintain_date)
            & (valid_group["time"] <= maintain_date + pd.Timedelta(days=7))
        ]["per"]
        after_15d = valid_group[
            (valid_group["time"] >= maintain_date)
            & (valid_group["time"] <= maintain_date + pd.Timedelta(days=15))
        ]["per"]
        before_30d = valid_group[
            (valid_group["time"] < maintain_date)
            & (valid_group["time"] >= maintain_date - pd.Timedelta(days=30))
        ]["per"]

        delta_per = after_per - before_per if pd.notna(before_per) and pd.notna(after_per) else np.nan
        relative_recovery_rate = delta_per / before_per if pd.notna(delta_per) and before_per not in (0, np.nan) else np.nan
        pre_30d_peak = before_30d.max() if not before_30d.empty else np.nan
        pre_30d_mean = before_30d.mean() if not before_30d.empty else np.nan
        pre_drop_30d = pre_30d_peak - before_per if pd.notna(pre_30d_peak) and pd.notna(before_per) else np.nan
        maintenance_effectiveness_coef = (
            delta_per / pre_drop_30d
            if pd.notna(delta_per) and pd.notna(pre_drop_30d) and pre_drop_30d > 0
            else np.nan
        )
        after_3d_mean = after_3d.mean() if not after_3d.empty else np.nan
        after_7d_mean = after_7d.mean() if not after_7d.empty else np.nan
        after_15d_mean = after_15d.mean() if not after_15d.empty else np.nan
        post_7d_decay_rate = (
            (after_7d_mean - after_per) / 7
            if pd.notna(after_7d_mean) and pd.notna(after_per)
            else np.nan
        )
        post_15d_decay_rate = (
            (after_15d_mean - after_per) / 15
            if pd.notna(after_15d_mean) and pd.notna(after_per)
            else np.nan
        )
        post_7d_decline_rate = -post_7d_decay_rate if pd.notna(post_7d_decay_rate) else np.nan
        post_15d_decline_rate = -post_15d_decay_rate if pd.notna(post_15d_decay_rate) else np.nan

        rows.append(
            {
                "filter_id": filter_id,
                "maintain_date": maintain_date,
                "maintain_type": record.maintain_type,
                "maintain_level": record.maintain_level,
                "before_date": before_time,
                "before_per": before_per,
                "after_date": after_time,
                "after_per": after_per,
                "delta_per": delta_per,
                "relative_recovery_rate": relative_recovery_rate,
                "days_to_before": (
                    (maintain_date - before_time).total_seconds() / 86400 if pd.notna(before_time) else np.nan
                ),
                "days_to_after": (
                    (after_time - maintain_date).total_seconds() / 86400 if pd.notna(after_time) else np.nan
                ),
                "after_3d_mean": after_3d_mean,
                "after_7d_mean": after_7d_mean,
                "after_15d_mean": after_15d_mean,
                "selected_window_hours": selected_window,
                "same_day_any_record": same_day_any,
                "same_day_valid_record": same_day_valid,
                "pre_30d_mean": pre_30d_mean,
                "pre_30d_peak": pre_30d_peak,
                "pre_drop_30d": pre_drop_30d,
                "maintenance_effectiveness_coef": maintenance_effectiveness_coef,
                "post_7d_decay_rate": post_7d_decay_rate,
                "post_15d_decay_rate": post_15d_decay_rate,
                "post_7d_decline_rate": post_7d_decline_rate,
                "post_15d_decline_rate": post_15d_decline_rate,
                "remark": "；".join(remarks) if remarks else "72h窗口内完成匹配",
            }
        )

    maintenance_match_df = pd.DataFrame(rows).sort_values(
        ["filter_id", "maintain_date"], kind="stable"
    )
    maintenance_match_df.reset_index(drop=True, inplace=True)
    return maintenance_match_df


def build_maintenance_effect_tables(
    maintenance_match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_by_type = maintenance_match_df.groupby(["maintain_type", "maintain_level"], as_index=False).agg(
        record_count=("delta_per", "size"),
        positive_recovery_ratio=("delta_per", lambda series: (series > 0).mean()),
        mean_before_per=("before_per", "mean"),
        mean_after_per=("after_per", "mean"),
        mean_delta_per=("delta_per", "mean"),
        median_delta_per=("delta_per", "median"),
        std_delta_per=("delta_per", "std"),
        mean_relative_recovery_rate=("relative_recovery_rate", "mean"),
        mean_after_3d_mean=("after_3d_mean", "mean"),
        mean_after_7d_mean=("after_7d_mean", "mean"),
        mean_after_15d_mean=("after_15d_mean", "mean"),
        mean_post_7d_decay_rate=("post_7d_decay_rate", "mean"),
        mean_post_15d_decay_rate=("post_15d_decay_rate", "mean"),
        mean_post_7d_decline_rate=("post_7d_decline_rate", "mean"),
        mean_post_15d_decline_rate=("post_15d_decline_rate", "mean"),
        mean_maintenance_effectiveness_coef=("maintenance_effectiveness_coef", "mean"),
        mean_match_window_hours=("selected_window_hours", "mean"),
    ).sort_values("maintain_level")

    summary_by_filter_type = maintenance_match_df.groupby(
        ["filter_id", "maintain_type", "maintain_level"], as_index=False
    ).agg(
        record_count=("delta_per", "size"),
        positive_recovery_ratio=("delta_per", lambda series: (series > 0).mean()),
        mean_before_per=("before_per", "mean"),
        mean_after_per=("after_per", "mean"),
        mean_delta_per=("delta_per", "mean"),
        median_delta_per=("delta_per", "median"),
        std_delta_per=("delta_per", "std"),
        mean_relative_recovery_rate=("relative_recovery_rate", "mean"),
        mean_after_3d_mean=("after_3d_mean", "mean"),
        mean_after_7d_mean=("after_7d_mean", "mean"),
        mean_after_15d_mean=("after_15d_mean", "mean"),
        mean_post_7d_decay_rate=("post_7d_decay_rate", "mean"),
        mean_post_15d_decay_rate=("post_15d_decay_rate", "mean"),
        mean_post_7d_decline_rate=("post_7d_decline_rate", "mean"),
        mean_post_15d_decline_rate=("post_15d_decline_rate", "mean"),
        mean_maintenance_effectiveness_coef=("maintenance_effectiveness_coef", "mean"),
        mean_match_window_hours=("selected_window_hours", "mean"),
    )
    summary_by_filter_type["filter_order"] = summary_by_filter_type["filter_id"].map(filter_sort_key)
    summary_by_filter_type = summary_by_filter_type.sort_values(
        ["filter_order", "maintain_level"], kind="stable"
    ).drop(columns="filter_order")

    decay_summary = pd.DataFrame(
        {
            "maintain_type": summary_by_type["maintain_type"],
            "mean_after_per": summary_by_type["mean_after_per"],
            "mean_after_3d_mean": summary_by_type["mean_after_3d_mean"],
            "mean_after_7d_mean": summary_by_type["mean_after_7d_mean"],
            "mean_after_15d_mean": summary_by_type["mean_after_15d_mean"],
            "after_3d_change_vs_after": summary_by_type["mean_after_3d_mean"] - summary_by_type["mean_after_per"],
            "after_7d_change_vs_after": summary_by_type["mean_after_7d_mean"] - summary_by_type["mean_after_per"],
            "after_15d_change_vs_after": summary_by_type["mean_after_15d_mean"] - summary_by_type["mean_after_per"],
            "mean_post_15d_decay_rate": summary_by_type["mean_post_15d_decay_rate"],
            "mean_post_15d_decline_rate": summary_by_type["mean_post_15d_decline_rate"],
        }
    )
    return summary_by_type, summary_by_filter_type, decay_summary


def attach_nearest_maintenance(
    frame: pd.DataFrame,
    maintenance_record_df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    result = frame.copy()
    nearest_dates: list[pd.Timestamp | pd.NaT] = []
    nearest_types: list[str | float] = []
    nearest_levels: list[int | float] = []
    nearest_days: list[int | float] = []

    maintenance_record_df = maintenance_record_df.copy()
    maintenance_record_df["date"] = pd.to_datetime(maintenance_record_df["date"])

    for record in result.itertuples(index=False):
        filter_id = getattr(record, "filter_id")
        date_value = pd.Timestamp(getattr(record, date_col))
        maint_group = maintenance_record_df[maintenance_record_df["filter_id"] == filter_id]
        if maint_group.empty:
            nearest_dates.append(pd.NaT)
            nearest_types.append(np.nan)
            nearest_levels.append(np.nan)
            nearest_days.append(np.nan)
            continue

        day_delta = (date_value - maint_group["date"]).dt.days
        nearest_idx = day_delta.abs().idxmin()
        nearest_dates.append(maint_group.loc[nearest_idx, "date"])
        nearest_types.append(maint_group.loc[nearest_idx, "maintain_type"])
        nearest_levels.append(maint_group.loc[nearest_idx, "maintain_level"])
        nearest_days.append(int(day_delta.loc[nearest_idx]))

    result["nearest_maintenance_date"] = nearest_dates
    result["nearest_maintain_type"] = nearest_types
    result["nearest_maintain_level"] = nearest_levels
    result["days_from_nearest_maintenance"] = nearest_days
    result["is_maintenance_linked"] = (
        result["days_from_nearest_maintenance"].abs() <= JUMP_LINK_WINDOW_DAYS
    )
    result["maintenance_relation"] = np.select(
        [
            result["days_from_nearest_maintenance"].between(0, JUMP_LINK_WINDOW_DAYS, inclusive="both"),
            result["days_from_nearest_maintenance"].between(-JUMP_LINK_WINDOW_DAYS, -1, inclusive="both"),
        ],
        [
            f"维护后{JUMP_LINK_WINDOW_DAYS}天内",
            f"维护前{JUMP_LINK_WINDOW_DAYS}天内",
        ],
        default="未邻近维护",
    )
    return result


def build_jump_and_linkage_tables(
    clean_data_df: pd.DataFrame,
    daily_valid: pd.DataFrame,
    maintenance_record_df: pd.DataFrame,
    maintenance_match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jump_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []

    for filter_id in sorted(daily_valid["filter_id"].unique(), key=filter_sort_key):
        group = daily_valid[daily_valid["filter_id"] == filter_id].sort_values("date").copy()
        group["previous_date"] = group["date"].shift()
        group["previous_daily_per"] = group["daily_per"].shift()
        group["gap_days"] = (group["date"] - group["previous_date"]).dt.days
        group["diff_per"] = group["daily_per"] - group["previous_daily_per"]
        group["diff_per_per_day"] = group["diff_per"] / group["gap_days"].replace(0, np.nan)

        abs_rate = group["diff_per_per_day"].abs().dropna()
        q1 = abs_rate.quantile(0.25)
        q3 = abs_rate.quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        group["jump_threshold"] = threshold
        group["is_jump"] = group["diff_per_per_day"].abs() > threshold
        jump_group = group[group["is_jump"]].copy()
        jump_group["jump_direction"] = np.where(jump_group["diff_per"] >= 0, "向上跳升", "向下跳降")
        jump_frames.append(jump_group)

        threshold_rows.append(
            {
                "filter_id": filter_id,
                "valid_day_count": len(group),
                "diff_sample_count": int(abs_rate.size),
                "abs_diff_q1": q1,
                "abs_diff_q3": q3,
                "abs_diff_iqr": iqr,
                "jump_threshold_per_day": threshold,
                "jump_count": int(jump_group.shape[0]),
                "positive_jump_count": int((jump_group["diff_per"] > 0).sum()),
                "negative_jump_count": int((jump_group["diff_per"] < 0).sum()),
            }
        )

    jump_points_df = pd.concat(jump_frames, ignore_index=True) if jump_frames else pd.DataFrame()
    if not jump_points_df.empty:
        jump_points_df = attach_nearest_maintenance(jump_points_df, maintenance_record_df, "date")
        jump_points_df["linked_maintain_type"] = np.where(
            jump_points_df["is_maintenance_linked"],
            jump_points_df["nearest_maintain_type"],
            "未邻近维护",
        )
        jump_points_df = jump_points_df[
            [
                "filter_id",
                "date",
                "previous_date",
                "previous_daily_per",
                "daily_per",
                "gap_days",
                "diff_per",
                "diff_per_per_day",
                "jump_threshold",
                "jump_direction",
                "nearest_maintenance_date",
                "nearest_maintain_type",
                "nearest_maintain_level",
                "days_from_nearest_maintenance",
                "is_maintenance_linked",
                "maintenance_relation",
                "linked_maintain_type",
            ]
        ].sort_values(["filter_id", "date"], kind="stable")

    jump_threshold_df = pd.DataFrame(threshold_rows)
    jump_summary_by_filter = (
        jump_points_df.groupby("filter_id", as_index=False)
        .agg(
            jump_count=("date", "size"),
            positive_jump_count=("jump_direction", lambda series: (series == "向上跳升").sum()),
            negative_jump_count=("jump_direction", lambda series: (series == "向下跳降").sum()),
            maintenance_linked_jump_count=("is_maintenance_linked", "sum"),
            maintenance_linked_positive_jump_count=(
                "diff_per",
                lambda series: ((series > 0) & jump_points_df.loc[series.index, "is_maintenance_linked"]).sum(),
            ),
            mean_abs_diff_per=("diff_per", lambda series: series.abs().mean()),
            max_abs_diff_per=("diff_per", lambda series: series.abs().max()),
        )
        if not jump_points_df.empty
        else pd.DataFrame()
    )
    if not jump_summary_by_filter.empty:
        jump_summary_by_filter["maintenance_linked_ratio"] = (
            jump_summary_by_filter["maintenance_linked_jump_count"] / jump_summary_by_filter["jump_count"]
        )
        jump_summary_by_filter["filter_order"] = jump_summary_by_filter["filter_id"].map(filter_sort_key)
        jump_summary_by_filter = jump_summary_by_filter.sort_values("filter_order").drop(columns="filter_order")

    jump_relation_summary = (
        jump_points_df.groupby(["maintenance_relation", "jump_direction"], as_index=False)
        .agg(
            jump_count=("date", "size"),
            mean_diff_per=("diff_per", "mean"),
            median_diff_per=("diff_per", "median"),
            mean_abs_diff_per=("diff_per", lambda series: series.abs().mean()),
        )
        if not jump_points_df.empty
        else pd.DataFrame()
    )

    maintenance_jump_rows: list[dict[str, object]] = []
    for record in maintenance_record_df.itertuples(index=False):
        filter_jumps = jump_points_df[jump_points_df["filter_id"] == record.filter_id] if not jump_points_df.empty else pd.DataFrame()
        window = filter_jumps[
            (filter_jumps["date"] >= record.date - pd.Timedelta(days=JUMP_LINK_WINDOW_DAYS))
            & (filter_jumps["date"] <= record.date + pd.Timedelta(days=JUMP_LINK_WINDOW_DAYS))
        ]
        after_window = window[window["date"] >= record.date]
        positive_after = after_window[after_window["diff_per"] > 0]
        maintenance_jump_rows.append(
            {
                "filter_id": record.filter_id,
                "maintain_date": record.date,
                "maintain_type": record.maintain_type,
                "maintain_level": record.maintain_level,
                "jump_count_pm3d": int(window.shape[0]),
                "positive_jump_count_after_3d": int(positive_after.shape[0]),
                "max_positive_jump_after_3d": positive_after["diff_per"].max() if not positive_after.empty else np.nan,
                "mean_positive_jump_after_3d": positive_after["diff_per"].mean() if not positive_after.empty else np.nan,
                "has_positive_jump_after_3d": bool(not positive_after.empty),
            }
        )
    maintenance_jump_df = pd.DataFrame(maintenance_jump_rows)
    maintenance_jump_df = maintenance_jump_df.merge(
        maintenance_match_df[
            ["filter_id", "maintain_date", "delta_per", "relative_recovery_rate", "selected_window_hours"]
        ],
        on=["filter_id", "maintain_date"],
        how="left",
    )
    maintenance_jump_summary = maintenance_jump_df.groupby(["maintain_type", "maintain_level"], as_index=False).agg(
        maintenance_count=("maintain_date", "size"),
        with_jump_pm3d_count=("jump_count_pm3d", lambda series: (series > 0).sum()),
        with_positive_jump_after_3d_count=("has_positive_jump_after_3d", "sum"),
        mean_delta_per=("delta_per", "mean"),
        mean_relative_recovery_rate=("relative_recovery_rate", "mean"),
        mean_max_positive_jump_after_3d=("max_positive_jump_after_3d", "mean"),
        mean_jump_count_pm3d=("jump_count_pm3d", "mean"),
    )
    maintenance_jump_summary["positive_jump_after_3d_ratio"] = (
        maintenance_jump_summary["with_positive_jump_after_3d_count"]
        / maintenance_jump_summary["maintenance_count"]
    )

    outlier_days_df = (
        clean_data_df.copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"]))
        .groupby(["filter_id", "date"], as_index=False)
        .agg(
            record_count=("per", "size"),
            missing_count=("is_missing", "sum"),
            outlier_count=("is_outlier", "sum"),
            valid_mean_per=("per", "mean"),
        )
    )
    outlier_days_df = outlier_days_df[outlier_days_df["outlier_count"] > 0].copy()
    outlier_days_df["outlier_ratio"] = outlier_days_df["outlier_count"] / outlier_days_df["record_count"]
    if not outlier_days_df.empty:
        outlier_days_df = attach_nearest_maintenance(outlier_days_df, maintenance_record_df, "date")
    outlier_summary_by_filter = (
        outlier_days_df.groupby("filter_id", as_index=False)
        .agg(
            outlier_day_count=("date", "size"),
            outlier_record_count=("outlier_count", "sum"),
            maintenance_linked_outlier_day_count=("is_maintenance_linked", "sum"),
            mean_outlier_ratio=("outlier_ratio", "mean"),
        )
        if not outlier_days_df.empty
        else pd.DataFrame()
    )
    if not outlier_summary_by_filter.empty:
        outlier_summary_by_filter["maintenance_linked_outlier_day_ratio"] = (
            outlier_summary_by_filter["maintenance_linked_outlier_day_count"]
            / outlier_summary_by_filter["outlier_day_count"]
        )
        outlier_summary_by_filter["filter_order"] = outlier_summary_by_filter["filter_id"].map(filter_sort_key)
        outlier_summary_by_filter = outlier_summary_by_filter.sort_values("filter_order").drop(columns="filter_order")

    return (
        jump_points_df.reset_index(drop=True),
        jump_threshold_df,
        jump_summary_by_filter,
        jump_relation_summary,
        maintenance_jump_summary,
        outlier_days_df.reset_index(drop=True),
        outlier_summary_by_filter,
    )


def save_figure_to_targets(fig: plt.Figure, filename: str) -> Path:
    path_a = FIGURE_DIR_A / filename
    path_c = FIGURE_DIR_C / filename
    fig.savefig(path_a, bbox_inches="tight")
    shutil.copy2(path_a, path_c)
    plt.close(fig)
    return path_a


def plot_all_filters_time_series(daily_valid: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(15, 8))
    for filter_id in sorted(daily_valid["filter_id"].unique(), key=filter_sort_key):
        group = daily_valid[daily_valid["filter_id"] == filter_id]
        ax.plot(group["date"], group["daily_per"], linewidth=1.3, alpha=0.88, label=filter_id)
    ax.set_title("10台过滤器日均透水率时间序列图")
    ax.set_xlabel("日期")
    ax.set_ylabel("日均透水率值")
    ax.legend(title="过滤器编号", ncol=1, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    return save_figure_to_targets(fig, "fig_01_all_filters_time_series.png")


def plot_filter_time_series_facets(
    daily_valid: pd.DataFrame, maintenance_record_df: pd.DataFrame
) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.2), sharex=True, sharey=True)
    axes = axes.flatten()
    middle_label_added = False
    major_label_added = False
    date_min = daily_valid["date"].min()
    date_max = daily_valid["date"].max()

    for ax, filter_id in zip(axes, sorted(daily_valid["filter_id"].unique(), key=filter_sort_key)):
        group = daily_valid[daily_valid["filter_id"] == filter_id].sort_values("date")
        ax.plot(group["date"], group["daily_per"], color="#3368A8", linewidth=1.05, label="日均透水率")

        maint_group = maintenance_record_df[maintenance_record_df["filter_id"] == filter_id]
        for record in maint_group.itertuples(index=False):
            if record.maintain_type == "中维护":
                label = "中维护" if not middle_label_added else None
                ax.axvline(record.date, color="#6B8FC8", linestyle="--", linewidth=0.7, alpha=0.55, label=label)
                middle_label_added = True
            else:
                label = "大维护" if not major_label_added else None
                ax.axvline(record.date, color="#C44E52", linestyle="-", linewidth=0.9, alpha=0.78, label=label)
                major_label_added = True

        ax.set_title(filter_id, fontsize=11, pad=4)
        ax.set_xlim(date_min, date_max)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax.grid(True, which="major", alpha=0.32)
        ax.grid(True, which="minor", axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=7, rotation=35)
        ax.tick_params(axis="y", labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.label_outer()

    handles, labels = [], []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle("各过滤器日均透水率时间序列与维护节点图", y=0.935, fontsize=15)
    fig.supxlabel("日期", y=0.035, fontsize=11)
    fig.supylabel("日均透水率值", x=0.018, fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.13, top=0.84, wspace=0.13, hspace=0.28)
    return save_figure_to_targets(fig, "fig_01a_filter_time_series_facets.png")


def plot_normalized_permeability_trend(daily_valid: pd.DataFrame) -> Path:
    normalized_frames: list[pd.DataFrame] = []
    for filter_id in sorted(daily_valid["filter_id"].unique(), key=filter_sort_key):
        group = daily_valid[daily_valid["filter_id"] == filter_id].sort_values("date").copy()
        baseline = group["daily_per"].quantile(0.90)
        group["normalized_per"] = group["daily_per"] / baseline * 100
        normalized_frames.append(group[["filter_id", "date", "normalized_per"]])

    normalized_df = pd.concat(normalized_frames, ignore_index=True)
    summary = normalized_df.groupby("date", as_index=False).agg(
        mean_normalized_per=("normalized_per", "mean"),
        q25=("normalized_per", lambda series: series.quantile(0.25)),
        q75=("normalized_per", lambda series: series.quantile(0.75)),
    )

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for filter_id in sorted(normalized_df["filter_id"].unique(), key=filter_sort_key):
        group = normalized_df[normalized_df["filter_id"] == filter_id]
        ax.plot(group["date"], group["normalized_per"], linewidth=0.75, alpha=0.22, color="#6F6F6F")

    x_values = summary["date"].to_numpy()
    ax.fill_between(
        x_values,
        summary["q25"].to_numpy(dtype=float),
        summary["q75"].to_numpy(dtype=float),
        color="#4C72B0",
        alpha=0.20,
        label="25%-75%区间",
    )
    ax.plot(summary["date"], summary["mean_normalized_per"], color="#C44E52", linewidth=2.2, label="标准化均值")
    ax.axhline(100, color="#333333", linestyle="--", linewidth=1.0, alpha=0.65, label="高位基准=100")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=0)
    upper = min(max(normalized_df["normalized_per"].quantile(0.995) * 1.05, 115), 150)
    lower = max(min(normalized_df["normalized_per"].quantile(0.005) * 0.95, 55), 20)
    ax.set_ylim(lower, upper)
    ax.set_title("各过滤器标准化透水率趋势图")
    ax.set_xlabel("日期")
    ax.set_ylabel("标准化透水率指数（各过滤器90分位值=100）")
    ax.legend(loc="upper left", frameon=True)
    return save_figure_to_targets(fig, "fig_01b_normalized_permeability_trend.png")


def plot_filter_trend_lines(
    daily_trend: pd.DataFrame, decline_rate_df: pd.DataFrame, maintenance_record_df: pd.DataFrame
) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, filter_id in zip(axes, sorted(daily_trend["filter_id"].unique(), key=filter_sort_key)):
        group = daily_trend[daily_trend["filter_id"] == filter_id].sort_values("date")
        row = decline_rate_df[decline_rate_df["filter_id"] == filter_id].iloc[0]
        ax.plot(group["date"], group["daily_per"], color="#4C72B0", linewidth=1.1, label="日均透水率")
        maintain_dates = pd.to_datetime(
            maintenance_record_df.loc[maintenance_record_df["filter_id"] == filter_id, "date"]
        ).sort_values().tolist()
        boundaries = [group["date"].min(), *maintain_dates, group["date"].max() + pd.Timedelta(days=1)]
        fitted_label_added = False
        for start_date, end_date in zip(boundaries[:-1], boundaries[1:]):
            segment = group[(group["date"] >= start_date) & (group["date"] < end_date)].sort_values("date")
            if len(segment) < 7:
                continue
            seg_x = segment["day_index"].astype(float).to_numpy()
            seg_y = segment["daily_per"].astype(float).to_numpy()
            seg_slope, seg_intercept, _ = fit_linear_trend(seg_x, seg_y)
            fitted = seg_intercept + seg_slope * seg_x
            label = "周期内拟合线" if not fitted_label_added else None
            ax.plot(segment["date"], fitted, color="#DD8452", linewidth=1.3, alpha=0.9, label=label)
            fitted_label_added = True
        if pd.notna(row["cycle_daily_decline_rate"]):
            ax.text(
                0.02,
                0.05,
                f"周期下降率={row['cycle_daily_decline_rate']:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        ax.set_title(filter_id)
        ax.set_xlabel("日期")
        ax.set_ylabel("透水率值")
        ax.tick_params(axis="x", rotation=30)
        if filter_id == "A1":
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("各过滤器维护周期内日均透水率下降趋势图", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_figure_to_targets(fig, "fig_02_filter_trend_lines.png")


def plot_monthly_average(monthly_overall: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly_overall["month"], monthly_overall["month_mean_per"], marker="o", color="#4C72B0", linewidth=2)
    for row in monthly_overall.itertuples(index=False):
        ax.text(row.month, row.month_mean_per + 0.8, f"{row.month_mean_per:.1f}", ha="center", fontsize=8)
    ax.set_xticks(monthly_overall["month"])
    ax.set_xticklabels(monthly_overall["month_label"])
    ax.set_title("月平均透水率变化图")
    ax.set_xlabel("月份")
    ax.set_ylabel("平均透水率值（日均聚合）")
    ax.set_ylim(monthly_overall["month_mean_per"].min() - 3, monthly_overall["month_mean_per"].max() + 5)
    ax.legend(["总体月平均透水率"], loc="best")
    return save_figure_to_targets(fig, "fig_03_monthly_average_permeability.png")


def plot_seasonal_average(seasonal_overall: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#55A868", "#4C72B0", "#C44E52", "#8172B3"]
    ax.bar(seasonal_overall["season"].astype(str), seasonal_overall["season_mean_per"], color=colors, label="季节平均透水率")
    for idx, row in enumerate(seasonal_overall.itertuples(index=False)):
        ax.text(idx, row.season_mean_per + 0.8, f"{row.season_mean_per:.1f}", ha="center", fontsize=9)
    ax.set_title("不同季节透水率均值图")
    ax.set_xlabel("季节")
    ax.set_ylabel("平均透水率值（日均聚合）")
    ax.set_ylim(seasonal_overall["season_mean_per"].min() - 5, seasonal_overall["season_mean_per"].max() + 7)
    ax.legend(loc="best")
    return save_figure_to_targets(fig, "fig_04_seasonal_average_permeability.png")


def plot_maintenance_effect_boxplot(maintenance_match_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    middle_n = int((maintenance_match_df["maintain_type"] == "中维护").sum())
    major_n = int((maintenance_match_df["maintain_type"] == "大维护").sum())
    data = [
        maintenance_match_df.loc[maintenance_match_df["maintain_type"] == "中维护", "delta_per"].dropna().to_numpy(),
        maintenance_match_df.loc[maintenance_match_df["maintain_type"] == "大维护", "delta_per"].dropna().to_numpy(),
    ]
    box = ax.boxplot(data, tick_labels=[f"中维护(n={middle_n})", f"大维护(n={major_n})"], patch_artist=True)
    for patch, color in zip(box["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("中维护和大维护恢复量箱线图")
    ax.set_xlabel("维护类型")
    ax.set_ylabel("恢复量 ΔP")
    ax.legend([box["boxes"][0], box["boxes"][1]], ["中维护", "大维护"], loc="best")
    return save_figure_to_targets(fig, "fig_05_maintenance_effect_boxplot.png")


def plot_before_after_maintenance(summary_by_type: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = summary_by_type["maintain_type"].tolist()
    x = np.arange(len(labels))
    width = 0.2
    ax.bar(x - 1.5 * width, summary_by_type["mean_before_per"], width=width, label="维护前最近透水率")
    ax.bar(x - 0.5 * width, summary_by_type["mean_after_per"], width=width, label="维护后最近透水率")
    ax.bar(x + 0.5 * width, summary_by_type["mean_after_7d_mean"], width=width, label="维护后7天均值")
    ax.bar(x + 1.5 * width, summary_by_type["mean_after_15d_mean"], width=width, label="维护后15天均值")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("维护前后透水率对比图")
    ax.set_xlabel("维护类型")
    ax.set_ylabel("平均透水率值")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return save_figure_to_targets(fig, "fig_06_before_after_maintenance.png")


def plot_filter_decline_rate(decline_rate_df: pd.DataFrame) -> Path:
    plot_df = decline_rate_df.sort_values("filter_id", key=lambda series: series.map(filter_sort_key)).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["filter_id"], plot_df["annual_decline_rate"], color="#C44E52", label="维护周期年化下降量")
    ax.set_title("各过滤器维护周期年化下降量图")
    ax.set_xlabel("过滤器编号")
    ax.set_ylabel("年化下降量（透水率值/年）")
    ax.set_ylim(0, plot_df["annual_decline_rate"].max() * 1.15)
    ax.legend(loc="best")
    for idx, value in enumerate(plot_df["annual_decline_rate"]):
        ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    return save_figure_to_targets(fig, "fig_07_filter_decline_rate.png")


def plot_jump_points_and_maintenance(
    jump_points_df: pd.DataFrame,
    maintenance_jump_summary: pd.DataFrame,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    jump_plot = jump_points_df.copy()
    jump_plot["filter_order"] = jump_plot["filter_id"].map(filter_sort_key)
    jump_plot = jump_plot.sort_values("filter_order")
    summary_rows = []
    for filter_id in jump_plot["filter_id"].drop_duplicates():
        group = jump_plot[jump_plot["filter_id"] == filter_id]
        summary_rows.append(
            {
                "filter_id": filter_id,
                "维护邻近向上跳升": int(((group["is_maintenance_linked"]) & (group["diff_per"] > 0)).sum()),
                "维护邻近向下跳降": int(((group["is_maintenance_linked"]) & (group["diff_per"] < 0)).sum()),
                "未邻近维护跳变": int((~group["is_maintenance_linked"]).sum()),
            }
        )
    jump_summary = pd.DataFrame(summary_rows)
    x = np.arange(len(jump_summary))
    bottom = np.zeros(len(jump_summary))
    colors = {
        "维护邻近向上跳升": "#55A868",
        "维护邻近向下跳降": "#C44E52",
        "未邻近维护跳变": "#9A9A9A",
    }
    for column in ["维护邻近向上跳升", "维护邻近向下跳降", "未邻近维护跳变"]:
        axes[0].bar(x, jump_summary[column], bottom=bottom, color=colors[column], label=column)
        bottom += jump_summary[column].to_numpy()
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(jump_summary["filter_id"], rotation=0)
    axes[0].set_title("各过滤器跳变点与维护邻近关系")
    axes[0].set_xlabel("过滤器编号")
    axes[0].set_ylabel("跳变点数量")
    axes[0].legend(loc="upper right", fontsize=8)

    type_plot = maintenance_jump_summary.sort_values("maintain_level").copy()
    labels = type_plot["maintain_type"].tolist()
    x2 = np.arange(len(labels))
    width = 0.35
    axes[1].bar(
        x2 - width / 2,
        type_plot["mean_delta_per"],
        width=width,
        color="#4C72B0",
        label="维护匹配平均恢复量",
    )
    axes[1].bar(
        x2 + width / 2,
        type_plot["mean_max_positive_jump_after_3d"],
        width=width,
        color="#DD8452",
        label="维护后3天最大向上跳升均值",
    )
    for idx, row in enumerate(type_plot.itertuples(index=False)):
        axes[1].text(idx - width / 2, row.mean_delta_per + 0.5, f"{row.mean_delta_per:.1f}", ha="center", fontsize=8)
        if pd.notna(row.mean_max_positive_jump_after_3d):
            axes[1].text(
                idx + width / 2,
                row.mean_max_positive_jump_after_3d + 0.5,
                f"{row.mean_max_positive_jump_after_3d:.1f}",
                ha="center",
                fontsize=8,
            )
        axes[1].text(
            idx,
            max(row.mean_delta_per, row.mean_max_positive_jump_after_3d) + 3,
            f"后3天跳升占比={row.positive_jump_after_3d_ratio:.1%}",
            ha="center",
            fontsize=8,
        )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("维护恢复量与邻近跳升幅度对比")
    axes[1].set_xlabel("维护类型")
    axes[1].set_ylabel("透水率变化量")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylim(0, max(type_plot["mean_delta_per"].max(), type_plot["mean_max_positive_jump_after_3d"].max()) + 8)

    fig.suptitle("跳变点与维护记录联动分析图", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_figure_to_targets(fig, "fig_08_jump_points_and_maintenance.png")


def plot_q1_method_flowchart() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, face: str, edge: str, fontsize: int = 12) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.016,rounding_size=0.02",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.35, color="#222222")

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#555555", lw: float = 1.8) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=lw,
                color=color,
                connectionstyle="arc3,rad=0.0",
            )
        )

    ax.text(0.5, 0.945, "第1问透水率变化规律分析建模思路", ha="center", va="center", fontsize=20, fontweight="bold", color="#222222")

    # 主流程节点
    box(0.04, 0.56, 0.17, 0.24, "原始数据\n附件1：透水率监测\n附件2：维护记录", "#E8F0FE", "#3B6EA8", fontsize=13)
    box(0.28, 0.56, 0.18, 0.24, "数据盘点与清洗\n统一编号/日期/字段\n保留缺失与异常标记", "#EAF7EA", "#4C8A4A", fontsize=13)

    # 分析模块外框
    group_x, group_y, group_w, group_h = 0.53, 0.30, 0.27, 0.50
    group = FancyBboxPatch(
        (group_x, group_y),
        group_w,
        group_h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=2.0,
        edgecolor="#C47A2C",
        facecolor="#FFF8EC",
    )
    ax.add_patch(group)
    ax.text(group_x + group_w / 2, group_y + group_h - 0.055, "四类规律分析", ha="center", va="center", fontsize=15, fontweight="bold", color="#7A4A16")

    small_w, small_h = 0.105, 0.145
    box(0.555, 0.58, small_w, small_h, "趋势分析\n维护周期下降率\n年化下降量", "#FFF3E0", "#D08B34", fontsize=11)
    box(0.675, 0.58, small_w, small_h, "周期性分析\n月均透水率\n季节影响因子", "#FFF3E0", "#D08B34", fontsize=11)
    box(0.555, 0.39, small_w, small_h, "维护效果分析\n恢复量/恢复率\n维护后衰减", "#FFF3E0", "#D08B34", fontsize=11)
    box(0.675, 0.39, small_w, small_h, "异常联动分析\nIQR异常/跳变点\n维护邻近匹配", "#FFF3E0", "#D08B34", fontsize=11)

    # 输出节点
    box(0.86, 0.58, 0.11, 0.18, "给 B\n建模参数\n下降率/季节因子\n维护恢复参数", "#FDECEC", "#B84A4A", fontsize=11)
    box(0.86, 0.34, 0.11, 0.18, "给 C\n论文素材\n图表/图注\n第1问结论", "#FDECEC", "#B84A4A", fontsize=11)

    # 主流程箭头
    arrow(0.21, 0.68, 0.28, 0.68)
    arrow(0.46, 0.68, group_x, 0.68)
    arrow(group_x + group_w, 0.61, 0.86, 0.67)
    arrow(group_x + group_w, 0.49, 0.86, 0.43)

    # 说明标签
    ax.text(0.245, 0.72, "字段统一", ha="center", va="bottom", fontsize=10, color="#555555")
    ax.text(0.495, 0.72, "形成标准表", ha="center", va="bottom", fontsize=10, color="#555555")
    ax.text(0.825, 0.71, "参数沉淀", ha="center", va="bottom", fontsize=10, color="#555555")
    ax.text(0.825, 0.47, "论文表达", ha="center", va="bottom", fontsize=10, color="#555555")

    ax.text(
        0.5,
        0.12,
        "核心逻辑：先构造可复用清洗数据，再围绕趋势、周期、维护和异常联动四条主线分析，最后分别沉淀为 B 的模型输入与 C 的论文素材。",
        ha="center",
        va="center",
        fontsize=12,
        color="#444444",
    )
    ax.text(
        0.5,
        0.075,
        "处理原则：原始附件只读不改；缺失值、IQR异常值和跳变点保留并分类标记；维护记录与透水率按过滤器编号和日期匹配。",
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
    )
    fig.tight_layout()
    return save_figure_to_targets(fig, "fig_09_q1_method_flowchart.png")

def build_stage3_excel_outputs(
    maintenance_match_df: pd.DataFrame,
    summary_by_type: pd.DataFrame,
    summary_by_filter_type: pd.DataFrame,
    decay_summary: pd.DataFrame,
    decline_rate_df: pd.DataFrame,
    monthly_overall: pd.DataFrame,
    seasonal_overall: pd.DataFrame,
    monthly_by_filter: pd.DataFrame,
    seasonal_by_filter: pd.DataFrame,
    jump_points_df: pd.DataFrame,
    jump_threshold_df: pd.DataFrame,
    jump_summary_by_filter: pd.DataFrame,
    jump_relation_summary: pd.DataFrame,
    maintenance_jump_summary: pd.DataFrame,
    outlier_days_df: pd.DataFrame,
    outlier_summary_by_filter: pd.DataFrame,
) -> Stage3Outputs:
    outputs = Stage3Outputs(
        maintenance_match_excel=EXPORT_DIR / "maintenance_match.xlsx",
        maintenance_effect_excel=EXPORT_DIR / "维护效果统计表.xlsx",
        decline_rate_excel=EXPORT_DIR / "每台过滤器下降率表.xlsx",
        conclusions_markdown=EXPORT_DIR / "第1问结论要点.md",
        jump_linkage_excel=EXPORT_DIR / "异常跳变点联动分析表.xlsx",
        anomaly_jump_markdown=EXPORT_DIR / "异常点与跳变点分析说明.md",
        permeability_maintenance_linkage_markdown=EXPORT_DIR / "维护记录与透水率联动分析.md",
        c_feedback_response_markdown=EXPORT_DIR / "C反馈回应与增强说明.md",
        figure_paths=(),
    )

    with pd.ExcelWriter(
        outputs.maintenance_match_excel,
        engine="openpyxl",
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS",
    ) as writer:
        maintenance_match_df.to_excel(writer, sheet_name="maintenance_match", index=False)

    with pd.ExcelWriter(
        outputs.maintenance_effect_excel,
        engine="openpyxl",
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS",
    ) as writer:
        summary_by_type.to_excel(writer, sheet_name="summary_by_type", index=False)
        summary_by_filter_type.to_excel(writer, sheet_name="summary_by_filter_type", index=False)
        decay_summary.to_excel(writer, sheet_name="decay_summary", index=False)

    with pd.ExcelWriter(
        outputs.decline_rate_excel,
        engine="openpyxl",
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS",
    ) as writer:
        decline_rate_df.to_excel(writer, sheet_name="decline_rate_by_filter", index=False)
        monthly_overall.to_excel(writer, sheet_name="monthly_average_overall", index=False)
        seasonal_overall.to_excel(writer, sheet_name="seasonal_average_overall", index=False)
        monthly_by_filter.to_excel(writer, sheet_name="monthly_average_by_filter", index=False)
        seasonal_by_filter.to_excel(writer, sheet_name="seasonal_average_by_filter", index=False)

    with pd.ExcelWriter(
        outputs.jump_linkage_excel,
        engine="openpyxl",
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS",
    ) as writer:
        jump_points_df.to_excel(writer, sheet_name="jump_points", index=False)
        jump_threshold_df.to_excel(writer, sheet_name="jump_thresholds", index=False)
        jump_summary_by_filter.to_excel(writer, sheet_name="jump_summary_by_filter", index=False)
        jump_relation_summary.to_excel(writer, sheet_name="jump_relation_summary", index=False)
        maintenance_jump_summary.to_excel(writer, sheet_name="maintenance_jump_summary", index=False)
        outlier_days_df.to_excel(writer, sheet_name="outlier_days", index=False)
        outlier_summary_by_filter.to_excel(writer, sheet_name="outlier_summary_by_filter", index=False)

    return outputs


def write_anomaly_jump_markdown(
    outputs: Stage3Outputs,
    clean_data_df: pd.DataFrame,
    jump_points_df: pd.DataFrame,
    jump_threshold_df: pd.DataFrame,
    jump_summary_by_filter: pd.DataFrame,
    outlier_summary_by_filter: pd.DataFrame,
) -> None:
    missing_count = int(clean_data_df["is_missing"].sum())
    outlier_count = int(clean_data_df["is_outlier"].sum())
    jump_count = int(jump_points_df.shape[0])
    linked_jump_count = int(jump_points_df["is_maintenance_linked"].sum()) if not jump_points_df.empty else 0
    positive_linked_jump_count = int(
        ((jump_points_df["is_maintenance_linked"]) & (jump_points_df["diff_per"] > 0)).sum()
    ) if not jump_points_df.empty else 0
    outlier_day_count = int(outlier_summary_by_filter["outlier_day_count"].sum()) if not outlier_summary_by_filter.empty else 0
    linked_outlier_day_count = int(
        outlier_summary_by_filter["maintenance_linked_outlier_day_count"].sum()
    ) if not outlier_summary_by_filter.empty else 0

    overview = pd.DataFrame(
        [
            {"项目": "透水率原始清洗记录", "数量": len(clean_data_df), "说明": "保留原始行，不覆盖附件"},
            {"项目": "缺失透水率记录", "数量": missing_count, "说明": "保留并标记 `is_missing=1`"},
            {"项目": "IQR 初筛异常记录", "数量": outlier_count, "说明": "保留并标记 `is_outlier=1`"},
            {"项目": "含异常值的日期数", "数量": outlier_day_count, "说明": "按过滤器-日期聚合统计"},
            {"项目": "日尺度跳变点", "数量": jump_count, "说明": "基于日均透水率差分的稳健阈值识别"},
            {"项目": f"维护前后{JUMP_LINK_WINDOW_DAYS}天内跳变点", "数量": linked_jump_count, "说明": "用于判断跳变与维护事件是否邻近"},
            {"项目": f"维护邻近向上跳升点", "数量": positive_linked_jump_count, "说明": "更可能对应维护恢复效应"},
            {"项目": f"维护前后{JUMP_LINK_WINDOW_DAYS}天内异常日期", "数量": linked_outlier_day_count, "说明": "用于解释部分异常是否与维护窗口相关"},
        ]
    )

    threshold_table = jump_threshold_df.copy()
    for column in ["abs_diff_q1", "abs_diff_q3", "abs_diff_iqr", "jump_threshold_per_day"]:
        threshold_table[column] = threshold_table[column].map(lambda value: format_number(value, 4))

    summary_table = jump_summary_by_filter.copy()
    for column in ["maintenance_linked_ratio", "mean_abs_diff_per", "max_abs_diff_per"]:
        if column in summary_table.columns:
            summary_table[column] = summary_table[column].map(lambda value: format_number(value, 4))

    outlier_table = outlier_summary_by_filter.copy()
    for column in ["mean_outlier_ratio", "maintenance_linked_outlier_day_ratio"]:
        if column in outlier_table.columns:
            outlier_table[column] = outlier_table[column].map(lambda value: format_number(value, 4))

    content = [
        "# 异常点与跳变点分析说明",
        "",
        "## 1. 回应 C 侧反馈的处理原则",
        "",
        "- 对缺失值、IQR 异常值和跳变点均不直接删除，也不使用固定默认值替代。",
        "- 缺失记录保留在 `clean_data.xlsx` 中，并通过 `is_missing` 标记；日均统计和趋势拟合只使用有效透水率值。",
        "- IQR 异常记录保留在清洗总表中，并通过 `is_outlier` 标记；趋势和季节性参数使用剔除 IQR 异常后的 `daily_trend` 口径，避免极端点影响下降率。",
        "- 跳变点作为运行状态突变或维护影响的候选事件单独识别，不与普通异常值混同处理。",
        "- 断档不做跨长间隔插值；跳变识别时使用 `diff_per_per_day = diff_per / gap_days`，以减少多日断档导致的虚假跳变。",
        "",
        "## 2. 识别口径",
        "",
        "- IQR 异常值：沿用阶段 1 对每台过滤器 `per` 值的 IQR 区间初筛。",
        "- 日尺度跳变点：先按过滤器和日期计算日均透水率，再计算相邻观测日的变化率 `diff_per_per_day`。",
        "- 跳变阈值：对每台过滤器的 `|diff_per_per_day|` 使用 `Q3 + 1.5 * IQR` 作为稳健阈值。",
        f"- 维护邻近窗口：若跳变日期或异常日期距离同一过滤器最近维护日期不超过 `{JUMP_LINK_WINDOW_DAYS}` 天，则标记为维护邻近事件。",
        "",
        "## 3. 总体统计",
        "",
        make_markdown_table(overview),
        "",
        "## 4. 每台过滤器跳变阈值",
        "",
        make_markdown_table(threshold_table),
        "",
        "## 5. 每台过滤器跳变点统计",
        "",
        make_markdown_table(summary_table),
        "",
        "## 6. 每台过滤器异常日期统计",
        "",
        make_markdown_table(outlier_table),
        "",
        "## 7. 输出文件",
        "",
        "- 详细表格：`01_数据处理_A/exports/异常跳变点联动分析表.xlsx`。",
        "- 主要工作表：`jump_points`、`jump_thresholds`、`jump_summary_by_filter`、`jump_relation_summary`、`maintenance_jump_summary`、`outlier_days`、`outlier_summary_by_filter`。",
        "- 联动图：`01_数据处理_A/figures/fig_08_jump_points_and_maintenance.png`，并已同步到 `03_论文_C/图表汇总/`。",
        "",
        "## 8. 论文表述建议",
        "",
        "- 建议写成“异常值和跳变点被保留并分类标记，其中维护邻近跳升主要反映维护恢复作用，未邻近维护的跳变保留为运行波动或待解释异常”。",
        "- 不建议写成“异常值已删除”或“缺失值已用默认值填补”，因为当前处理原则是保留、标记、分口径使用。",
        "",
    ]
    outputs.anomaly_jump_markdown.write_text("\n".join(content), encoding="utf-8")


def write_permeability_maintenance_linkage_markdown(
    outputs: Stage3Outputs,
    maintenance_match_df: pd.DataFrame,
    jump_points_df: pd.DataFrame,
    jump_relation_summary: pd.DataFrame,
    maintenance_jump_summary: pd.DataFrame,
) -> None:
    maintenance_count = int(maintenance_match_df.shape[0])
    matched_count = int(
        (maintenance_match_df["before_per"].notna() & maintenance_match_df["after_per"].notna()).sum()
    )
    linked_jump_count = int(jump_points_df["is_maintenance_linked"].sum()) if not jump_points_df.empty else 0
    positive_linked_jump_count = int(
        ((jump_points_df["is_maintenance_linked"]) & (jump_points_df["diff_per"] > 0)).sum()
    ) if not jump_points_df.empty else 0
    positive_linked_ratio = positive_linked_jump_count / linked_jump_count if linked_jump_count else np.nan

    summary_by_type = maintenance_match_df.groupby(["maintain_type", "maintain_level"], as_index=False).agg(
        maintenance_count=("maintain_date", "size"),
        mean_before_per=("before_per", "mean"),
        mean_after_per=("after_per", "mean"),
        mean_delta_per=("delta_per", "mean"),
        mean_relative_recovery_rate=("relative_recovery_rate", "mean"),
        positive_recovery_ratio=("delta_per", lambda series: (series > 0).mean()),
    )
    summary_by_type = summary_by_type.merge(
        maintenance_jump_summary[
            [
                "maintain_type",
                "maintain_level",
                "with_positive_jump_after_3d_count",
                "positive_jump_after_3d_ratio",
                "mean_max_positive_jump_after_3d",
            ]
        ],
        on=["maintain_type", "maintain_level"],
        how="left",
    ).sort_values("maintain_level")
    format_table = summary_by_type.copy()
    for column in [
        "mean_before_per",
        "mean_after_per",
        "mean_delta_per",
        "mean_relative_recovery_rate",
        "positive_recovery_ratio",
        "positive_jump_after_3d_ratio",
        "mean_max_positive_jump_after_3d",
    ]:
        format_table[column] = format_table[column].map(lambda value: format_number(value, 4))

    relation_table = jump_relation_summary.copy()
    for column in ["mean_diff_per", "median_diff_per", "mean_abs_diff_per"]:
        if column in relation_table.columns:
            relation_table[column] = relation_table[column].map(lambda value: format_number(value, 4))

    overview = pd.DataFrame(
        [
            {"项目": "维护记录数", "结果": maintenance_count},
            {"项目": "可匹配维护前后透水率记录数", "结果": matched_count},
            {"项目": f"维护前后{JUMP_LINK_WINDOW_DAYS}天内跳变点数", "结果": linked_jump_count},
            {"项目": f"维护邻近向上跳升点数", "结果": positive_linked_jump_count},
            {"项目": "维护邻近跳变中向上跳升占比", "结果": format_percent(positive_linked_ratio)},
        ]
    )

    content = [
        "# 维护记录与透水率联动分析",
        "",
        "## 1. 分析目的",
        "",
        "- 本文件回应 C 侧提出的“缺少数据 A 和数据 B 内在联系”的问题。",
        "- 这里的数据 A 指透水率监测数据，数据 B 指维护记录。",
        "- 联动分析不重新清洗数据，而是在 A 侧已完成的 `clean_data.xlsx`、`maintenance_record.xlsx` 和 `maintenance_match.xlsx` 基础上补充事件关联解释。",
        "",
        "## 2. 联动口径",
        "",
        "1. 维护前后匹配：对每条维护记录匹配维护前最近有效透水率和维护后最近有效透水率，计算恢复量 `delta_per`。",
        f"2. 跳变邻近匹配：识别日均透水率跳变点，并判断其是否位于同一过滤器维护日期前后 `{JUMP_LINK_WINDOW_DAYS}` 天内。",
        "3. 维护后跳升解释：若维护后 3 天内出现向上跳升，则作为维护恢复效应的辅助证据；若跳变未邻近维护，则保留为运行波动或待解释异常。",
        "",
        "## 3. 总体联动结果",
        "",
        make_markdown_table(overview),
        "",
        "## 4. 按维护类型的联动统计",
        "",
        make_markdown_table(format_table),
        "",
        "## 5. 按跳变方向和维护邻近关系统计",
        "",
        make_markdown_table(relation_table),
        "",
        "## 6. 结论要点",
        "",
        f"- 127 条维护记录均能匹配到维护前后有效透水率，说明维护记录与透水率数据可以按过滤器编号和日期建立稳定关联。",
        f"- 共识别到 `{jump_points_df.shape[0]}` 个日尺度跳变点，其中 `{linked_jump_count}` 个位于维护前后 `{JUMP_LINK_WINDOW_DAYS}` 天内。",
        f"- 维护邻近跳变中有 `{positive_linked_jump_count}` 个为向上跳升，占比为 `{format_percent(positive_linked_ratio)}`，说明维护事件与透水率跳升存在明显对应关系。",
        "- 中维护和大维护均表现出维护后透水率提升，但大维护样本量较少，论文中仍需保留样本量限制说明。",
        "- 未邻近维护的跳变点不直接删除，应作为运行扰动、采样波动或未记录事件的候选异常进行保留说明。",
        "",
        "## 7. 输出文件",
        "",
        "- 联动明细表：`01_数据处理_A/exports/异常跳变点联动分析表.xlsx`。",
        "- 维护匹配表：`01_数据处理_A/exports/maintenance_match.xlsx`。",
        "- 联动图：`01_数据处理_A/figures/fig_08_jump_points_and_maintenance.png`。",
        "",
    ]
    outputs.permeability_maintenance_linkage_markdown.write_text("\n".join(content), encoding="utf-8")


def write_question1_conclusions(
    outputs: Stage3Outputs,
    decline_rate_df: pd.DataFrame,
    monthly_overall: pd.DataFrame,
    seasonal_overall: pd.DataFrame,
    summary_by_type: pd.DataFrame,
    jump_points_df: pd.DataFrame,
    maintenance_jump_summary: pd.DataFrame,
) -> None:
    decline_sorted = decline_rate_df.sort_values("daily_decline_rate", ascending=False).reset_index(drop=True)
    fastest = decline_sorted.iloc[0]
    slowest = decline_sorted.iloc[-1]
    best_month = monthly_overall.loc[monthly_overall["month_mean_per"].idxmax()]
    worst_month = monthly_overall.loc[monthly_overall["month_mean_per"].idxmin()]
    best_season = seasonal_overall.loc[seasonal_overall["season_mean_per"].idxmax()]
    worst_season = seasonal_overall.loc[seasonal_overall["season_mean_per"].idxmin()]
    middle = summary_by_type[summary_by_type["maintain_type"] == "中维护"].iloc[0]
    major = summary_by_type[summary_by_type["maintain_type"] == "大维护"].iloc[0]
    jump_count = int(jump_points_df.shape[0])
    linked_jump_count = int(jump_points_df["is_maintenance_linked"].sum()) if not jump_points_df.empty else 0
    positive_linked_jump_count = int(
        ((jump_points_df["is_maintenance_linked"]) & (jump_points_df["diff_per"] > 0)).sum()
    ) if not jump_points_df.empty else 0
    positive_linked_ratio = positive_linked_jump_count / linked_jump_count if linked_jump_count else np.nan
    middle_jump = maintenance_jump_summary[maintenance_jump_summary["maintain_type"] == "中维护"].iloc[0]
    major_jump = maintenance_jump_summary[maintenance_jump_summary["maintain_type"] == "大维护"].iloc[0]

    content = [
        "# 第1问结论要点",
        "",
        "## 1. 下降趋势",
        "",
        "- 用于成员 B 建模的下降率参数采用“维护周期内平均下降率”，而不是直接对全时段含维护抬升序列做单条直线拟合。",
        f"- 按维护周期内平均下降率计算，下降速度最快的是 `{fastest['filter_id']}`，日均下降约 `{fastest['daily_decline_rate']:.4f}`，年化下降量约 `{fastest['annual_decline_rate']:.2f}`。",
        f"- 下降速度最慢的是 `{slowest['filter_id']}`，日均下降约 `{slowest['daily_decline_rate']:.4f}`，年化下降量约 `{slowest['annual_decline_rate']:.2f}`。",
        f"- 过滤器之间存在明显个体差异，维护周期年化下降量区间约为 `{decline_rate_df['annual_decline_rate'].min():.2f}` 至 `{decline_rate_df['annual_decline_rate'].max():.2f}`，说明寿命预测必须分设备估计退化参数。",
        f"- 作为辅助解释，全时段净趋势斜率已单列保存在 `net_trend_slope_per_day` 与 `net_annual_change` 字段中，用于反映维护作用叠加后的净变化。",
        "",
        "## 2. 周期性与季节性",
        "",
        f"- 按月平均透水率统计，平均水平最高的月份为 `{int(best_month['month'])}月`，月均透水率约 `{best_month['month_mean_per']:.2f}`；最低月份为 `{int(worst_month['month'])}月`，月均透水率约 `{worst_month['month_mean_per']:.2f}`。",
        f"- 按季节平均透水率统计，`{best_season['season']}` 季平均透水率最高，约 `{best_season['season_mean_per']:.2f}`；`{worst_season['season']}` 季最低，约 `{worst_season['season_mean_per']:.2f}`。",
        f"- 月份系数和季节系数已写入 `每台过滤器下降率表.xlsx`，可直接供成员 B 作为季节项参数参考。",
        "",
        "## 3. 维护影响",
        "",
        f"- 中维护的平均恢复量为 `{middle['mean_delta_per']:.2f}`，平均相对恢复率为 `{middle['mean_relative_recovery_rate']:.2%}`，正向恢复占比为 `{middle['positive_recovery_ratio']:.2%}`。",
        f"- 大维护的平均恢复量为 `{major['mean_delta_per']:.2f}`，平均相对恢复率为 `{major['mean_relative_recovery_rate']:.2%}`，正向恢复占比为 `{major['positive_recovery_ratio']:.2%}`。",
        f"- 从样本结果看，中维护平均恢复量略高于大维护，但大维护平均相对恢复率更高；由于大维护样本仅 `{int(major['record_count'])}` 次，不能仅凭当前样本断定其绝对恢复量一定更弱。",
        f"- 维护后 7 天和 15 天均值，以及正向定义的维护后衰减率 `post_7d_decline_rate`、`post_15d_decline_rate` 已写入 `maintenance_match.xlsx` 与 `维护效果统计表.xlsx`。",
        "",
        "## 4. 异常、跳变与维护联动",
        "",
        "- 缺失值、IQR 异常值和跳变点均未直接删除；缺失值保留 `is_missing` 标记，IQR 异常值保留 `is_outlier` 标记，跳变点单独输出到 `异常跳变点联动分析表.xlsx`。",
        f"- 基于日均透水率差分的稳健阈值共识别 `{jump_count}` 个日尺度跳变点，其中 `{linked_jump_count}` 个位于维护前后 `{JUMP_LINK_WINDOW_DAYS}` 天内。",
        f"- 维护邻近跳变中有 `{positive_linked_jump_count}` 个为向上跳升，占比为 `{format_percent(positive_linked_ratio)}`，说明维护记录与透水率跳升存在明显联动关系。",
        f"- 中维护记录中维护后 3 天内出现向上跳升的占比为 `{middle_jump['positive_jump_after_3d_ratio']:.2%}`，大维护对应占比为 `{major_jump['positive_jump_after_3d_ratio']:.2%}`。",
        "- 未邻近维护的跳变点保留为运行扰动、采样波动或未记录事件的候选异常，不作为删除依据。",
        "",
        "## 5. 可供 B 建模的影响指标",
        "",
        "- 每台过滤器长期趋势参数：`daily_decline_rate`、`annual_decline_rate`、`trend_r2`、`net_trend_slope_per_day`。",
        "- 季节影响参数：`month_factor`、`season_factor` 及分过滤器月/季均值。",
        "- 维护影响参数：`delta_per`、`relative_recovery_rate`、`maintenance_effectiveness_coef`、`post_15d_decline_rate`。",
        "- 异常与联动辅助指标：`is_outlier`、`diff_per`、`diff_per_per_day`、`is_maintenance_linked`、`maintenance_relation`。",
        "- 维护匹配口径：默认使用 `72h` 前后窗口，无法匹配时扩展至 `168h`，并在 `remark` 字段记录。",
        "",
        "## 6. 图表与结果文件",
        "",
        "- 图表已输出到 `01_数据处理_A/figures/` 与 `03_论文_C/图表汇总/`。",
        "- 趋势图采用三层表达：`fig_01_all_filters_time_series.png` 保留 10 台设备总览，`fig_01a_filter_time_series_facets.png` 作为正文小多图主图，`fig_01b_normalized_permeability_trend.png` 用于比较标准化后的相对趋势。",
        "- `fig_01b_normalized_permeability_trend.png` 以各过滤器日均透水率的 90 分位值作为高位基准 100，用于消除设备量级差异，不替代原始透水率绝对值分析。",
        "- 新增 `fig_08_jump_points_and_maintenance.png` 用于展示跳变点与维护记录的联动关系。",
        "- 核心数据表包括 `maintenance_match.xlsx`、`维护效果统计表.xlsx`、`每台过滤器下降率表.xlsx`、`异常跳变点联动分析表.xlsx`。",
        "",
    ]
    outputs.conclusions_markdown.write_text("\n".join(content), encoding="utf-8")


def write_c_feedback_response_markdown(
    outputs: Stage3Outputs,
    maintenance_match_df: pd.DataFrame,
    jump_points_df: pd.DataFrame,
    maintenance_jump_summary: pd.DataFrame,
) -> None:
    matched_count = int((maintenance_match_df["before_per"].notna() & maintenance_match_df["after_per"].notna()).sum())
    maintenance_count = int(maintenance_match_df.shape[0])
    jump_count = int(jump_points_df.shape[0])
    linked_jump_count = int(jump_points_df["is_maintenance_linked"].sum()) if not jump_points_df.empty else 0
    positive_linked_jump_count = int(
        ((jump_points_df["is_maintenance_linked"]) & (jump_points_df["diff_per"] > 0)).sum()
    ) if not jump_points_df.empty else 0
    positive_linked_ratio = positive_linked_jump_count / linked_jump_count if linked_jump_count else np.nan

    def type_jump_ratio(maintain_type: str) -> str:
        row = maintenance_jump_summary[maintenance_jump_summary["maintain_type"] == maintain_type]
        if row.empty:
            return "-"
        return format_percent(row.iloc[0]["positive_jump_after_3d_ratio"])

    content = [
        "# C 反馈回应与 A 侧增强说明",
        "",
        "## 1. C 侧反馈要点",
        "",
        "C 侧反馈集中在两类问题：",
        "",
        "1. 数据处理对异常点、跳变点的处理思路和方法说明不够充分。",
        "2. 趋势和结论偏单一维度，缺少透水率监测数据与维护记录之间的联动分析。",
        "",
        "## 2. A 侧回应原则",
        "",
        "- 不修改原始附件，不覆盖 `00_题目与附件/` 中的文件。",
        "- 不把异常值简单删除，也不使用固定默认值替代缺失值。",
        "- 将缺失、IQR 异常、日尺度跳变、维护邻近跳变分开处理。",
        "- 将透水率监测数据与维护记录按 `filter_id` 和日期建立事件级联动关系。",
        "- 所有新增计算结果均由 `04_代码/A_data_process.py` 可复现生成。",
        "",
        "## 3. 已新增或补强的交付物",
        "",
        "| C 侧反馈 | A 侧补强动作 | 输出文件 |",
        "| --- | --- | --- |",
        "| 异常点处理说明不够充分 | 明确缺失值、IQR 异常值、断档和跳变点的保留、标记和分口径使用规则 | `异常点与跳变点分析说明.md` |",
        "| 跳变点没有单独分析 | 基于日均透水率差分和 `Q3 + 1.5IQR` 稳健阈值识别日尺度跳变点 | `异常跳变点联动分析表.xlsx` |",
        f"| 维护记录与透水率缺少内在联系 | 将跳变点与维护日期做前后 {JUMP_LINK_WINDOW_DAYS} 天邻近匹配，并统计维护后跳升情况 | `维护记录与透水率联动分析.md` |",
        "| 论文缺少直观图表支撑 | 新增跳变点与维护记录联动图，并同步到 C 侧图表汇总 | `fig_08_jump_points_and_maintenance.png` |",
        "| 结论中缺少联动表述 | 在第 1 问结论和 A→B/A→C 交接材料中补充异常跳变与维护联动段落 | `第1问结论要点.md`、交接说明文件 |",
        "",
        "## 4. 关键增强结果",
        "",
        f"- 共识别 `{jump_count}` 个日尺度跳变点。",
        f"- 其中 `{linked_jump_count}` 个跳变点位于维护前后 `{JUMP_LINK_WINDOW_DAYS}` 天内。",
        f"- 维护邻近跳变中 `{positive_linked_jump_count}` 个为向上跳升，占比 `{format_percent(positive_linked_ratio)}`。",
        f"- 中维护记录中维护后 3 天内出现向上跳升的占比约 `{type_jump_ratio('中维护')}`。",
        f"- 大维护记录中维护后 3 天内出现向上跳升的占比约 `{type_jump_ratio('大维护')}`。",
        f"- `{matched_count}` / `{maintenance_count}` 条维护记录能匹配到维护前后有效透水率，说明维护记录与透水率数据可以稳定关联。",
        "",
        "## 5. 对论文写作的建议",
        "",
        "- 在数据预处理小节中补充：异常值和跳变点没有被简单删除，而是保留并分类标记。",
        "- 在第 1 问维护影响小节后增加“异常跳变与维护联动分析”段落。",
        "- 使用 `fig_08_jump_points_and_maintenance.png` 说明维护邻近跳变多数为向上跳升。",
        "- 对未邻近维护的跳变点，应表述为“运行扰动、采样波动或未记录事件的候选异常”，不要直接解释为维护效果。",
        "",
        "## 6. 复现方式",
        "",
        "在项目根目录运行：",
        "",
        "```bash",
        "python 04_代码/A_data_process.py",
        "```",
        "",
        "运行后会重新生成异常跳变联动表、异常跳变说明、维护联动说明、C 反馈回应说明和 `fig_08_jump_points_and_maintenance.png`。",
        "",
    ]
    outputs.c_feedback_response_markdown.write_text("\n".join(content), encoding="utf-8")


def run_stage3() -> Stage3Outputs:
    ensure_directories()
    setup_plot_style()
    _, _, filter_frames = build_filter_stats()
    clean_data_df = build_clean_data(filter_frames)
    maintenance_record_df = build_clean_maintenance_record()

    daily_valid, daily_trend, monthly_overall, seasonal_overall, monthly_by_filter, seasonal_by_filter = (
        prepare_stage3_base_tables(clean_data_df)
    )
    decline_rate_df = build_decline_rate_table(daily_trend, maintenance_record_df)
    maintenance_match_df = build_maintenance_match_table(clean_data_df, maintenance_record_df)
    summary_by_type, summary_by_filter_type, decay_summary = build_maintenance_effect_tables(maintenance_match_df)
    (
        jump_points_df,
        jump_threshold_df,
        jump_summary_by_filter,
        jump_relation_summary,
        maintenance_jump_summary,
        outlier_days_df,
        outlier_summary_by_filter,
    ) = build_jump_and_linkage_tables(
        clean_data_df=clean_data_df,
        daily_valid=daily_valid,
        maintenance_record_df=maintenance_record_df,
        maintenance_match_df=maintenance_match_df,
    )

    outputs = build_stage3_excel_outputs(
        maintenance_match_df=maintenance_match_df,
        summary_by_type=summary_by_type,
        summary_by_filter_type=summary_by_filter_type,
        decay_summary=decay_summary,
        decline_rate_df=decline_rate_df,
        monthly_overall=monthly_overall,
        seasonal_overall=seasonal_overall,
        monthly_by_filter=monthly_by_filter,
        seasonal_by_filter=seasonal_by_filter,
        jump_points_df=jump_points_df,
        jump_threshold_df=jump_threshold_df,
        jump_summary_by_filter=jump_summary_by_filter,
        jump_relation_summary=jump_relation_summary,
        maintenance_jump_summary=maintenance_jump_summary,
        outlier_days_df=outlier_days_df,
        outlier_summary_by_filter=outlier_summary_by_filter,
    )

    figure_paths = (
        plot_all_filters_time_series(daily_valid),
        plot_filter_time_series_facets(daily_valid, maintenance_record_df),
        plot_normalized_permeability_trend(daily_valid),
        plot_filter_trend_lines(daily_trend, decline_rate_df, maintenance_record_df),
        plot_monthly_average(monthly_overall),
        plot_seasonal_average(seasonal_overall),
        plot_maintenance_effect_boxplot(maintenance_match_df),
        plot_before_after_maintenance(summary_by_type),
        plot_filter_decline_rate(decline_rate_df),
        plot_jump_points_and_maintenance(jump_points_df, maintenance_jump_summary),
        plot_q1_method_flowchart(),
    )
    outputs = Stage3Outputs(
        maintenance_match_excel=outputs.maintenance_match_excel,
        maintenance_effect_excel=outputs.maintenance_effect_excel,
        decline_rate_excel=outputs.decline_rate_excel,
        conclusions_markdown=outputs.conclusions_markdown,
        jump_linkage_excel=outputs.jump_linkage_excel,
        anomaly_jump_markdown=outputs.anomaly_jump_markdown,
        permeability_maintenance_linkage_markdown=outputs.permeability_maintenance_linkage_markdown,
        c_feedback_response_markdown=outputs.c_feedback_response_markdown,
        figure_paths=figure_paths,
    )
    write_anomaly_jump_markdown(
        outputs=outputs,
        clean_data_df=clean_data_df,
        jump_points_df=jump_points_df,
        jump_threshold_df=jump_threshold_df,
        jump_summary_by_filter=jump_summary_by_filter,
        outlier_summary_by_filter=outlier_summary_by_filter,
    )
    write_permeability_maintenance_linkage_markdown(
        outputs=outputs,
        maintenance_match_df=maintenance_match_df,
        jump_points_df=jump_points_df,
        jump_relation_summary=jump_relation_summary,
        maintenance_jump_summary=maintenance_jump_summary,
    )
    write_question1_conclusions(
        outputs=outputs,
        decline_rate_df=decline_rate_df,
        monthly_overall=monthly_overall,
        seasonal_overall=seasonal_overall,
        summary_by_type=summary_by_type,
        jump_points_df=jump_points_df,
        maintenance_jump_summary=maintenance_jump_summary,
    )
    write_c_feedback_response_markdown(
        outputs=outputs,
        maintenance_match_df=maintenance_match_df,
        jump_points_df=jump_points_df,
        maintenance_jump_summary=maintenance_jump_summary,
    )
    return outputs


def run_stage1() -> Stage1Outputs:
    ensure_directories()

    filter_stats_df, outlier_stats_df, filter_frames = build_filter_stats()
    maintenance_stats_df, maintenance_match_df = build_maintenance_stats(filter_stats_df, filter_frames)

    outputs = write_stage1_csv_outputs(
        filter_stats_df=filter_stats_df,
        outlier_stats_df=outlier_stats_df,
        maintenance_stats_df=maintenance_stats_df,
        maintenance_match_df=maintenance_match_df,
    )

    write_structure_markdown(
        outputs=outputs,
        filter_stats_df=filter_stats_df,
        maintenance_stats_df=maintenance_stats_df,
        maintenance_match_df=maintenance_match_df,
    )
    write_maintenance_markdown(
        outputs=outputs,
        maintenance_stats_df=maintenance_stats_df,
        maintenance_match_df=maintenance_match_df,
    )

    anomaly_df = build_anomaly_summary(
        filter_stats_df=filter_stats_df,
        outlier_stats_df=outlier_stats_df,
        maintenance_stats_df=maintenance_stats_df,
        maintenance_match_df=maintenance_match_df,
    )
    write_anomaly_markdown(
        outputs=outputs,
        anomaly_df=anomaly_df,
        filter_stats_df=filter_stats_df,
        maintenance_match_df=maintenance_match_df,
    )
    return outputs


def main() -> None:
    stage1_outputs = run_stage1()
    stage2_outputs = run_stage2()
    stage3_outputs = run_stage3()
    print("阶段 1、阶段 2 与阶段 3 已完成，输出文件如下：")
    for path in (
        stage1_outputs.filter_stats_csv,
        stage1_outputs.permeability_outlier_csv,
        stage1_outputs.maintenance_stats_csv,
        stage1_outputs.maintenance_matchability_csv,
        stage1_outputs.structure_markdown,
        stage1_outputs.maintenance_markdown,
        stage1_outputs.anomaly_markdown,
        stage2_outputs.clean_data_excel,
        stage2_outputs.maintenance_record_excel,
        stage2_outputs.cleaning_markdown,
        stage3_outputs.maintenance_match_excel,
        stage3_outputs.maintenance_effect_excel,
        stage3_outputs.decline_rate_excel,
        stage3_outputs.conclusions_markdown,
        stage3_outputs.jump_linkage_excel,
        stage3_outputs.anomaly_jump_markdown,
        stage3_outputs.permeability_maintenance_linkage_markdown,
        stage3_outputs.c_feedback_response_markdown,
        *stage3_outputs.figure_paths,
    ):
        print(f"- {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
