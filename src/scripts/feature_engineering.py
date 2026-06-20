"""Engineer unified, context-aware features for all ASTraM events."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to sys.path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    INPUT_CSV, FEATURES_CSV, EXPECTED_ROWS, EXPECTED_ORIGINAL_COLUMNS, 
    EXPECTED_OUTPUT_COLUMNS, PEAK_HOURS
)
from src.logger import get_logger

logger = get_logger(__name__)

ENGINEERED_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "advance_notice",
    "duration_hours",
    "days_until_event",
    "minutes_since_reported",
    "y_closure",
    "y_severity",
    "y_duration",
]


def print_header(title: str) -> None:
    """Log a readable console section header."""
    logger.info(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def validate_features(df: pd.DataFrame) -> None:
    """Fail fast if the engineered dataset violates required invariants."""
    planned_mask = df["event_type"].eq("planned")
    unplanned_mask = df["event_type"].eq("unplanned")

    assert df.shape == (
        EXPECTED_ROWS,
        EXPECTED_OUTPUT_COLUMNS,
    ), f"Unexpected output shape: {df.shape}"
    assert int(planned_mask.sum()) == 467
    assert int(unplanned_mask.sum()) == 7706
    assert not df[ENGINEERED_COLUMNS].isna().any().any()

    for column in [
        "is_weekend",
        "is_peak_hour",
        "advance_notice",
        "y_closure",
    ]:
        assert set(df[column].unique()).issubset({0, 1}), (
            f"{column} contains non-binary values"
        )

    assert df.loc[unplanned_mask, "days_until_event"].eq(0).all()
    assert df.loc[planned_mask, "minutes_since_reported"].eq(0).all()
    assert np.allclose(df["y_duration"], df["duration_hours"])
    assert int(df["y_closure"].sum()) == 676


def main() -> None:
    """Load, engineer, validate, summarize, and save the unified feature data."""
    logger.info(f"Loading data from {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        logger.error(f"Failed to load input data from {INPUT_CSV}: {e}")
        sys.exit(1)

    assert df.shape == (
        EXPECTED_ROWS,
        EXPECTED_ORIGINAL_COLUMNS,
    ), f"Unexpected source shape: {df.shape}"

    # Parse both timestamps in UTC. The source contains mixed fractional-second
    # precision, so format="mixed" is required to retain every valid timestamp.
    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"], utc=True, format="mixed", errors="coerce"
    )
    df["end_datetime"] = pd.to_datetime(
        df["end_datetime"], utc=True, format="mixed", errors="coerce"
    )
    assert df["start_datetime"].notna().all()

    # Universal temporal features used by the single unified model.
    df["hour"] = df["start_datetime"].dt.hour.astype("int8")
    df["day_of_week"] = df["start_datetime"].dt.dayofweek.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["is_peak_hour"] = df["hour"].isin(PEAK_HOURS).astype("int8")

    # This key context indicator lets one model distinguish advance-planned
    # events from incidents reported after they occur.
    df["advance_notice"] = df["event_type"].eq("planned").astype("int8")

    # Calculate observed durations. Per the agreed policy, negative and extreme
    # values remain unchanged and are reported as data-quality anomalies.
    raw_duration = (
        df["end_datetime"] - df["start_datetime"]
    ).dt.total_seconds() / 3600
    observed_duration_count = int(raw_duration.notna().sum())
    negative_duration_count = int(raw_duration.lt(0).sum())
    observed_min_duration = float(raw_duration.min())
    observed_max_duration = float(raw_duration.max())

    # Fill missing durations using the median for the same event cause. Causes
    # with no observed duration fall back to the global observed median.
    cause_medians = raw_duration.groupby(df["event_cause"]).transform("median")
    after_cause_imputation = raw_duration.fillna(cause_medians)
    cause_imputed_count = int(
        (raw_duration.isna() & after_cause_imputation.notna()).sum()
    )

    global_median = float(raw_duration.median())
    df["duration_hours"] = after_cause_imputation.fillna(global_median)
    global_imputed_count = int(
        (after_cause_imputation.isna() & df["duration_hours"].notna()).sum()
    )

    # Capture the reference time once so every row uses exactly the same UTC
    # instant. These values intentionally change whenever the script is rerun.
    reference_time = pd.Timestamp.now(tz="UTC")
    planned_mask = df["event_type"].eq("planned")
    unplanned_mask = df["event_type"].eq("unplanned")

    df["days_until_event"] = 0.0
    df.loc[planned_mask, "days_until_event"] = (
        df.loc[planned_mask, "start_datetime"] - reference_time
    ).dt.days.astype(float)

    df["minutes_since_reported"] = 0.0
    df.loc[unplanned_mask, "minutes_since_reported"] = (
        reference_time - df.loc[unplanned_mask, "start_datetime"]
    ).dt.total_seconds() / 60

    # Targets shared by both planned and unplanned training examples.
    df["y_closure"] = df["requires_road_closure"].astype("int8")
    max_duration = float(df["duration_hours"].max())
    assert max_duration != 0, "Cannot normalize severity with zero max duration"
    df["y_severity"] = (
        0.6 * df["y_closure"]
        + 0.3 * (df["duration_hours"] / max_duration)
        + 0.05
    )
    df["y_duration"] = df["duration_hours"]

    validate_features(df)
    try:
        df.to_csv(FEATURES_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
        logger.info(f"Successfully saved features to {FEATURES_CSV}")
    except Exception as e:
        logger.error(f"Failed to save features to {FEATURES_CSV}: {e}")
        sys.exit(1)

    summary = (
        df.groupby("event_type")
        .agg(
            count=("id", "size"),
            avg_duration_hours=("duration_hours", "mean"),
            closures=("y_closure", "sum"),
            closure_rate=("y_closure", "mean"),
        )
        .reindex(["planned", "unplanned"])
    )

    print_header("UNIFIED ASTraM FEATURE ENGINEERING")
    logger.info(f"- Reference time (UTC): {reference_time.isoformat()}")
    logger.info(f"- Source shape: {EXPECTED_ROWS:,} rows x {EXPECTED_ORIGINAL_COLUMNS} columns")
    logger.info(f"- Output shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"- Output file: {FEATURES_CSV}")

    print_header("FEATURE SUMMARY")
    for event_type, row in summary.iterrows():
        logger.info(f"- {event_type.title()} events:")
        logger.info(f"  - Count: {int(row['count']):,}")
        logger.info(f"  - Average duration: {row['avg_duration_hours']:,.2f} hours")
        logger.info(
            f"  - Closures: {int(row['closures']):,}/{int(row['count']):,} "
            f"({row['closure_rate']:.1%})"
        )

    logger.info(
        f"- Average planned days_until_event: "
        f"{df.loc[planned_mask, 'days_until_event'].mean():,.2f} days"
    )
    logger.info(
        f"- Average unplanned minutes_since_reported: "
        f"{df.loc[unplanned_mask, 'minutes_since_reported'].mean():,.2f} minutes"
    )

    print_header("DURATION IMPUTATION AND DATA-QUALITY WARNING")
    logger.info(f"- Observed durations: {observed_duration_count:,}")
    logger.info(f"- Missing durations before imputation: {raw_duration.isna().sum():,}")
    logger.info(f"- Filled with event-cause median: {cause_imputed_count:,}")
    logger.info(f"- Filled with global median ({global_median:,.2f} hours): {global_imputed_count:,}")
    logger.info(f"- Negative observed durations retained: {negative_duration_count:,}")
    logger.info(
        f"- Observed duration range retained: "
        f"{observed_min_duration:,.2f} to {observed_max_duration:,.2f} hours"
    )
    logger.info("- Warning: retained negative/extreme values affect duration averages and severity.")

    print_header("VERIFICATION")
    logger.info("- PASS: 8,173 rows and 57 columns")
    logger.info("- PASS: Planned/unplanned counts remain 467/7,706")
    logger.info("- PASS: Engineered features and targets contain no missing values")
    logger.info("- PASS: Binary features contain only 0 and 1")
    logger.info("- PASS: Context-inapplicable feature values are zero")
    logger.info("- PASS: y_duration matches duration_hours")
    logger.info("- PASS: Closure target total remains 676")


if __name__ == "__main__":
    main()
