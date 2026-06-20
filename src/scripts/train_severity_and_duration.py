"""Train the ASTraM severity regressor and build duration lookup statistics."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add the project root to sys.path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    FEATURES_CSV,
    MODELS_DIR,
    FEATURE_NAMES_PATH,
    SEVERITY_MODEL_PATH,
    DURATION_MODEL_PATH,
    EXPECTED_ROWS,
)
from src.logger import get_logger

logger = get_logger(__name__)

DURATION_LOOKUP_PATH = DURATION_MODEL_PATH

EXPECTED_CAUSES = 17
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "event_cause",
    "corridor",
    "is_peak_hour",
    "is_weekend",
    "advance_notice",
    "days_until_event",
    "minutes_since_reported",
    "duration_hours",
]
CATEGORICAL_COLUMNS = ["event_cause", "corridor"]
TARGET_COLUMN = "y_severity"


def print_header(title: str) -> None:
    """Print a clear console section header."""
    logger.info(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def validate_source(df: pd.DataFrame) -> None:
    """Validate all columns required for both requested artifacts."""
    required = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN, "event_type", "duration_hours"]
    )
    missing = sorted(required.difference(df.columns))

    assert len(df) == EXPECTED_ROWS, f"Expected 8,173 rows, found {len(df):,}"
    assert not missing, f"Missing required columns: {missing}"
    assert set(df["event_type"].unique()) == {"planned", "unplanned"}
    assert df[TARGET_COLUMN].between(0, 1).all()
    assert df["duration_hours"].notna().all()


def build_duration_lookup(
    df: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Create median, mean, and count statistics for each event cause."""
    grouped = (
        df.groupby("event_cause", sort=True)["duration_hours"]
        .agg(["median", "mean", "count"])
    )

    return {
        str(cause): {
            "median_hours": float(row["median"]),
            "mean_hours": float(row["mean"]),
            "count": int(row["count"]),
        }
        for cause, row in grouped.iterrows()
    }


def main() -> None:
    """Train, evaluate, save, reload, and verify both requested artifacts."""
    try:
        df = pd.read_csv(FEATURES_CSV)
    except Exception as e:
        logger.error(f"Failed to load input data from {FEATURES_CSV}: {e}")
        sys.exit(1)
    validate_source(df)

    # Reproduce the exact preprocessing schema used by the closure classifier.
    X = df[FEATURE_COLUMNS].copy()
    X["corridor"] = X["corridor"].fillna("Unknown")
    X["event_cause"] = X["event_cause"].fillna("Unknown")
    X_encoded = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
        dtype="int8",
    )
    encoded_feature_names = X_encoded.columns.tolist()

    with FEATURE_NAMES_PATH.open("rb") as feature_file:
        closure_feature_names = pickle.load(feature_file)

    assert encoded_feature_names == closure_feature_names, (
        "Severity preprocessing does not match closure classifier schema"
    )
    assert not X_encoded.isna().any().any()
    assert all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in X_encoded.dtypes
    )

    y = df[TARGET_COLUMN].astype(float).copy()
    event_types = df["event_type"].copy()

    # Split by event context exactly as the closure classifier does.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        event_type_train,
        event_type_test,
    ) = train_test_split(
        X_encoded,
        y,
        event_types,
        test_size=TEST_SIZE,
        stratify=event_types,
        random_state=RANDOM_STATE,
    )

    severity_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    severity_model.fit(X_train, y_train)
    severity_predictions = severity_model.predict(X_test)

    mae = mean_absolute_error(y_test, severity_predictions)
    rmse = mean_squared_error(y_test, severity_predictions) ** 0.5
    r_squared = r2_score(y_test, severity_predictions)

    # Build the requested lookup from the processed duration values verbatim.
    duration_lookup = build_duration_lookup(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with SEVERITY_MODEL_PATH.open("wb") as model_file:
        pickle.dump(severity_model, model_file)
    with DURATION_LOOKUP_PATH.open("wb") as lookup_file:
        pickle.dump(duration_lookup, lookup_file)

    print_header("UNIFIED ASTraM REGRESSORS")
    logger.info(f"- Source file: {FEATURES_CSV}")
    logger.info(f"- Source shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"- Encoded features: {X_encoded.shape[1]}")
    logger.info(f"- Training rows: {len(X_train):,}")
    logger.info(
        f"  - Planned: {event_type_train.eq('planned').sum():,}"
        f" | Unplanned: {event_type_train.eq('unplanned').sum():,}"
    )
    logger.info(f"- Test rows: {len(X_test):,}")
    logger.info(
        f"  - Planned: {event_type_test.eq('planned').sum():,}"
        f" | Unplanned: {event_type_test.eq('unplanned').sum():,}"
    )

    print_header("SEVERITY TEST METRICS")
    logger.info(f"- MAE:  {mae:.6f}")
    logger.info(f"- RMSE: {rmse:.6f}")
    logger.info(f"- R²:   {r_squared:.6f}")

    print_header("PART B: DURATION LOOKUP BY EVENT CAUSE")
    for cause in sorted(duration_lookup):
        stats = duration_lookup[cause]
        logger.info(
            f"- {cause}: median={stats['median_hours']:,.3f}h, "
            f"mean={stats['mean_hours']:,.3f}h, "
            f"count={stats['count']:,}"
        )

    print_header("DURATION DATA-QUALITY WARNING")
    logger.info(
        f"- Negative duration rows retained: "
        f"{df['duration_hours'].lt(0).sum():,}"
    )
    logger.info(
        f"- Duration range retained: {df['duration_hours'].min():,.3f}h "
        f"to {df['duration_hours'].max():,.3f}h"
    )

    # Reload both outputs and verify serialization independently.
    with SEVERITY_MODEL_PATH.open("rb") as model_file:
        reloaded_model = pickle.load(model_file)
    with DURATION_LOOKUP_PATH.open("rb") as lookup_file:
        reloaded_lookup = pickle.load(lookup_file)

    reloaded_predictions = reloaded_model.predict(X_test)
    fresh_lookup = build_duration_lookup(df)

    lookup_values_match = all(
        cause in reloaded_lookup
        and reloaded_lookup[cause]["count"] == stats["count"]
        and np.isclose(
            reloaded_lookup[cause]["median_hours"],
            stats["median_hours"],
        )
        and np.isclose(
            reloaded_lookup[cause]["mean_hours"],
            stats["mean_hours"],
        )
        for cause, stats in fresh_lookup.items()
    )

    metrics = np.array([mae, rmse, r_squared], dtype=float)
    checks = {
        "All 8,173 rows entered preprocessing": len(X_encoded) == EXPECTED_ROWS,
        "Train/test totals reconcile": (
            len(X_train) + len(X_test) == EXPECTED_ROWS
        ),
        "Both event types are present in training": (
            event_type_train.nunique() == 2
        ),
        "Both event types are present in test": event_type_test.nunique() == 2,
        "Encoded data is numeric and null-free": (
            not X_encoded.isna().any().any()
            and all(
                pd.api.types.is_numeric_dtype(dtype)
                for dtype in X_encoded.dtypes
            )
        ),
        "Feature schema matches closure classifier": (
            encoded_feature_names == closure_feature_names
            and reloaded_model.n_features_in_ == len(closure_feature_names)
        ),
        "Predictions are finite": np.isfinite(severity_predictions).all(),
        "MAE, RMSE, and R² are finite": np.isfinite(metrics).all(),
        "Reloaded model reproduces predictions": np.allclose(
            reloaded_predictions, severity_predictions
        ),
        "Lookup contains all 17 event causes": (
            len(reloaded_lookup) == EXPECTED_CAUSES
        ),
        "Lookup counts total 8,173": (
            sum(item["count"] for item in reloaded_lookup.values())
            == EXPECTED_ROWS
        ),
        "Lookup values match fresh group-by": lookup_values_match,
        "Severity artifact is non-empty": (
            SEVERITY_MODEL_PATH.stat().st_size > 0
        ),
        "Duration artifact is non-empty": (
            DURATION_LOOKUP_PATH.stat().st_size > 0
        ),
    }

    print_header("SAVED ARTIFACTS")
    logger.info(f"- Severity model: {SEVERITY_MODEL_PATH}")
    logger.info(f"- Duration lookup: {DURATION_LOOKUP_PATH}")

    print_header("VERIFICATION")
    for label, passed in checks.items():
        logger.info(f"- {'PASS' if passed else 'FAIL'}: {label}")

    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        logger.error(f"Verification failed: {', '.join(failed)}")
        sys.exit(1)

    logger.info("- All severity and duration artifact checks passed.")


if __name__ == "__main__":
    main()
