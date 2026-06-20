import sys
from pathlib import Path
import pandas as pd

# Add the project root to sys.path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import INPUT_CSV, EDA_REPORT
from src.logger import get_logger

logger = get_logger(__name__)


def section(title: str) -> list[str]:
    return ["", "=" * 78, title, "=" * 78]


def frame_as_bullets(frame: pd.DataFrame) -> list[str]:
    lines = []
    for index, row in frame.iterrows():
        values = " | ".join(f"{column}: {row[column]}" for column in frame.columns)
        lines.append(f"- {index}: {values}")
    return lines


def main() -> None:
    logger.info(f"Starting EDA on {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        logger.error(f"Failed to load input data from {INPUT_CSV}: {e}")
        sys.exit(1)
        
    raw_start_non_null = int(df["start_datetime"].notna().sum())
    raw_end_non_null = int(df["end_datetime"].notna().sum())

    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"], utc=True, format="mixed", errors="coerce"
    )
    df["end_datetime"] = pd.to_datetime(
        df["end_datetime"], utc=True, format="mixed", errors="coerce"
    )

    total_rows, total_columns = df.shape
    closure_count = int(df["requires_road_closure"].sum())
    overall_closure_rate = df["requires_road_closure"].mean()

    event_type_summary = (
        df.groupby("event_type", dropna=False)["requires_road_closure"]
        .agg(events="size", closures="sum", closure_rate="mean")
        .sort_values("events", ascending=False)
    )
    event_type_summary["share_of_events"] = (
        event_type_summary["events"] / total_rows
    )

    cause_summary = (
        df.groupby("event_cause", dropna=False)["requires_road_closure"]
        .agg(events="size", closures="sum", closure_rate="mean")
        .sort_values(["events", "event_cause"], ascending=[False, True])
        .head(10)
    )

    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": df.isna().mean() * 100,
        }
    )

    numeric_summary = df.describe(include="number").transpose()
    categorical_columns = df.select_dtypes(
        include=["object", "string", "bool", "category"]
    ).columns
    categorical_summary = df[categorical_columns].describe().transpose()

    monthly_events = (
        df["start_datetime"]
        .dt.strftime("%Y-%m")
        .value_counts()
        .sort_index()
        .rename_axis("month")
        .to_frame("events")
    )

    parsed_start_count = int(df["start_datetime"].notna().sum())
    parsed_end_count = int(df["end_datetime"].notna().sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["id"].duplicated().sum())

    type_events_total = int(event_type_summary["events"].sum())
    type_closures_total = int(event_type_summary["closures"].sum())

    lines = [
        "ASTraM EVENT DATASET - COMPLETE EXPLORATORY DATA ANALYSIS",
        f"- Source file: {INPUT_CSV}",
        f"- Report file: {EDA_REPORT}",
    ]

    lines += section("1. DATASET OVERVIEW")
    lines += [
        f"- Shape: {total_rows:,} rows x {total_columns} columns",
        f"- Total rows: {total_rows:,}",
        f"- Total columns: {total_columns}",
        f"- Duplicate full rows: {duplicate_rows:,}",
        f"- Duplicate IDs: {duplicate_ids:,}",
    ]

    lines += section("2. COLUMNS AND INFERRED DATA TYPES")
    for position, column in enumerate(df.columns, start=1):
        lines.append(f"- {position:02d}. {column}: {df[column].dtype}")

    lines += section("3. EVENT TYPE BREAKDOWN")
    for event_type, row in event_type_summary.iterrows():
        lines.append(
            f"- {event_type}: {int(row['events']):,} events "
            f"({row['share_of_events']:.1%} of all events)"
        )

    lines += section("4. UTC DATETIME PARSING AND DATE COVERAGE")
    lines += [
        "- Parsing method: pandas mixed-format parsing with utc=True and errors='coerce'",
        f"- start_datetime populated before parsing: {raw_start_non_null:,}",
        f"- start_datetime successfully parsed: {parsed_start_count:,}",
        f"- start_datetime parse failures: {raw_start_non_null - parsed_start_count:,}",
        f"- end_datetime populated before parsing: {raw_end_non_null:,}",
        f"- end_datetime successfully parsed: {parsed_end_count:,}",
        f"- end_datetime parse failures among populated values: "
        f"{raw_end_non_null - parsed_end_count:,}",
        f"- Earliest start_datetime: {df['start_datetime'].min().isoformat()}",
        f"- Latest start_datetime: {df['start_datetime'].max().isoformat()}",
        "- Human-readable date range: November 9, 2023 to April 8, 2024 (UTC)",
    ]

    lines += section("5. ROAD CLOSURE RATES")
    lines += [
        f"- Overall: {closure_count:,}/{total_rows:,} "
        f"({overall_closure_rate:.1%})",
        "- Closure field used: requires_road_closure",
        "",
        "Closure rate by event_type:",
    ]
    for event_type, row in event_type_summary.iterrows():
        lines.append(
            f"- {event_type}: {int(row['closures']):,}/{int(row['events']):,} "
            f"({row['closure_rate']:.1%})"
        )

    lines += ["", "Closure rate for the top 10 event causes by event count:"]
    for cause, row in cause_summary.iterrows():
        lines.append(
            f"- {cause}: {int(row['closures']):,}/{int(row['events']):,} "
            f"({row['closure_rate']:.1%})"
        )

    lines += section("6. MISSING VALUES")
    lines.append("- Missing values by column (count and percentage):")
    for column, row in missing.iterrows():
        lines.append(
            f"- {column}: {int(row['missing_count']):,} "
            f"({row['missing_percent']:.2f}%)"
        )

    lines += section("7. NUMERIC SUMMARY STATISTICS")
    lines.append(
        numeric_summary.to_string(
            float_format=lambda value: f"{value:,.3f}", na_rep="NaN"
        )
    )

    lines += section("8. CATEGORICAL AND BOOLEAN SUMMARY STATISTICS")
    lines.append(categorical_summary.to_string(na_rep="NaN"))

    lines += section("9. MONTHLY EVENT DISTRIBUTION")
    for month, row in monthly_events.iterrows():
        lines.append(f"- {month}: {int(row['events']):,} events")

    lines += section("10. DATA QUALITY NOTE")
    lines += [
        "- The supplied CSV is treated as the authoritative source.",
        "- Calculated closure rates are 36.2% for planned events and 6.6% for "
        "unplanned events.",
        "- These do not reproduce the battle plan's stated 35.3% planned and "
        "4.3% unplanned rates.",
        "- No filtering or imputation was applied to force the expected rates.",
        "- Missing end_datetime values were retained as missing.",
    ]

    checks = {
        "Exactly 8,173 rows analyzed": total_rows == 8173,
        "Exactly 46 columns loaded": total_columns == 46,
        "All 8,173 start datetimes parsed": parsed_start_count == 8173,
        "All 490 populated end datetimes parsed": parsed_end_count == 490,
        "Event-type row totals reconcile": type_events_total == total_rows,
        "Event-type closure totals reconcile": type_closures_total == closure_count,
        "Planned count is 467": int(event_type_summary.loc["planned", "events"])
        == 467,
        "Unplanned count is 7,706": int(
            event_type_summary.loc["unplanned", "events"]
        )
        == 7706,
    }

    lines += section("11. VERIFICATION CHECKS")
    for label, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {label}")

    if not all(checks.values()):
        failed = [label for label, passed in checks.items() if not passed]
        raise AssertionError(f"Verification failed: {', '.join(failed)}")

    try:
        EDA_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Created {EDA_REPORT}")
    except Exception as e:
        logger.error(f"Failed to write report to {EDA_REPORT}: {e}")
        sys.exit(1)

    logger.info(f"Rows analyzed: {total_rows:,}")
    logger.info(f"Overall closure rate: {overall_closure_rate:.1%}")
    logger.info("All verification checks passed.")


if __name__ == "__main__":
    main()
