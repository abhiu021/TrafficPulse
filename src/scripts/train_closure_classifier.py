"""Train one closure classifier for planned and unplanned ASTraM events."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


import sys
from pathlib import Path

# Add the project root to sys.path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    FEATURES_CSV, MODELS_DIR, CLOSURE_CLASSIFIER_MODEL_PATH, 
    FEATURE_NAMES_PATH, EXPECTED_ROWS
)
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = CLOSURE_CLASSIFIER_MODEL_PATH

EXPECTED_ROWS = 8173
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
TARGET_COLUMN = "y_closure"


def print_header(title: str) -> None:
    """Print a clear console section header."""
    logger.info(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Calculate binary classification metrics, guarding subgroup AUC."""
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": float("nan"),
    }

    if np.unique(np.asarray(y_true)).size == 2:
        metrics["auc_roc"] = roc_auc_score(y_true, y_probability)

    return metrics


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """Print one metric block."""
    print_header(title)
    logger.info(f"- Precision: {metrics['precision']:.3f}")
    logger.info(f"- Recall:    {metrics['recall']:.3f}")
    logger.info(f"- F1-score:  {metrics['f1']:.3f}")
    if np.isnan(metrics["auc_roc"]):
        logger.info("- AUC-ROC:   unavailable (only one target class in subgroup)")
    else:
        logger.info(f"- AUC-ROC:   {metrics['auc_roc']:.3f}")


def validate_source(df: pd.DataFrame) -> None:
    """Validate source rows and required modeling columns."""
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, "event_type"])
    missing_columns = sorted(required_columns.difference(df.columns))

    assert len(df) == EXPECTED_ROWS, f"Expected 8,173 rows, found {len(df):,}"
    assert not missing_columns, f"Missing required columns: {missing_columns}"
    assert set(df["event_type"].unique()) == {"planned", "unplanned"}
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def main() -> None:
    """Prepare data, train, evaluate, save, reload, and verify the model."""
    try:
        df = pd.read_csv(FEATURES_CSV)
    except Exception as e:
        logger.error(f"Failed to load input data from {FEATURES_CSV}: {e}")
        sys.exit(1)
    validate_source(df)

    # Select only the requested unified-model predictors.
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype("int8").copy()
    event_types = df["event_type"].copy()

    missing_corridors = int(X["corridor"].isna().sum())
    X["corridor"] = X["corridor"].fillna("Unknown")

    # event_cause is complete in this dataset, but this explicit fallback keeps
    # preprocessing robust if a future compatible file contains a missing cause.
    X["event_cause"] = X["event_cause"].fillna("Unknown")

    # Keep every category so the saved ordered feature names define the exact
    # inference schema expected by the fitted estimator.
    X_encoded = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
        dtype="int8",
    )
    feature_names = X_encoded.columns.tolist()

    assert not X_encoded.isna().any().any(), "Encoded feature matrix has nulls"
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_encoded.dtypes)

    # Split all aligned arrays together. Stratification is intentionally by
    # event_type so both planned and unplanned contexts occur in each split.
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

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Default classifier threshold is 0.5, as requested.
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    overall_metrics = calculate_metrics(y_test, y_pred, y_probability)
    planned_mask = event_type_test.eq("planned").to_numpy()
    unplanned_mask = event_type_test.eq("unplanned").to_numpy()
    planned_metrics = calculate_metrics(
        y_test.to_numpy()[planned_mask],
        y_pred[planned_mask],
        y_probability[planned_mask],
    )
    unplanned_metrics = calculate_metrics(
        y_test.to_numpy()[unplanned_mask],
        y_pred[unplanned_mask],
        y_probability[unplanned_mask],
    )
    overall_confusion_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print_header("UNIFIED ASTraM CLOSURE CLASSIFIER")
    logger.info(f"- Source file: {FEATURES_CSV}")
    logger.info(f"- Source shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"- Missing corridors encoded as Unknown: {missing_corridors:,}")
    logger.info(f"- Encoded feature count: {X_encoded.shape[1]}")
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
    logger.info(
        f"- Training target classes: "
        f"no closure={y_train.eq(0).sum():,}, closure={y_train.eq(1).sum():,}"
    )
    logger.info(
        f"- Test target classes: "
        f"no closure={y_test.eq(0).sum():,}, closure={y_test.eq(1).sum():,}"
    )

    print_metrics("OVERALL TEST METRICS", overall_metrics)
    print_metrics("PLANNED EVENTS ONLY", planned_metrics)
    print_metrics("UNPLANNED EVENTS ONLY", unplanned_metrics)

    print_header("OVERALL CONFUSION MATRIX")
    logger.info("- Rows = actual [no closure, closure]")
    logger.info("- Columns = predicted [no closure, closure]")
    logger.info("\n" + str(overall_confusion_matrix))

    print_header("TOP 15 FEATURE IMPORTANCES")
    for rank, row in feature_importance.head(15).iterrows():
        logger.info(f"- {rank + 1:02d}. {row['feature']}: {row['importance']:.6f}")

    auc_expectations = [
        ("Overall", overall_metrics["auc_roc"], 0.78),
        ("Planned", planned_metrics["auc_roc"], 0.80),
        ("Unplanned", unplanned_metrics["auc_roc"], 0.70),
    ]
    below_benchmark = [
        f"{label} ({auc:.3f})"
        for label, auc, minimum in auc_expectations
        if not np.isnan(auc) and auc < minimum
    ]
    if below_benchmark:
        print_header("BENCHMARK NOTE")
        logger.warning(
            "- Measured AUC is below the battle-plan benchmark for: "
            + ", ".join(below_benchmark)
            + "."
        )
        logger.warning("- The requested model and threshold were retained without tuning.")

    # Persist the estimator and exact ordered encoded-column schema.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    with FEATURE_NAMES_PATH.open("wb") as feature_file:
        pickle.dump(feature_names, feature_file)

    # Reload both artifacts and verify that serialization preserves predictions.
    with MODEL_PATH.open("rb") as model_file:
        reloaded_model = pickle.load(model_file)
    with FEATURE_NAMES_PATH.open("rb") as feature_file:
        reloaded_feature_names = pickle.load(feature_file)

    reloaded_predictions = reloaded_model.predict(X_test)
    reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

    checks = {
        "All 8,173 rows entered preprocessing": len(X_encoded) == EXPECTED_ROWS,
        "Train/test totals reconcile": len(X_train) + len(X_test) == EXPECTED_ROWS,
        "Both event types exist in training": event_type_train.nunique() == 2,
        "Both event types exist in test": event_type_test.nunique() == 2,
        "Both target classes exist in training": y_train.nunique() == 2,
        "Both target classes exist in test": y_test.nunique() == 2,
        "Encoded data is numeric and null-free": (
            not X_encoded.isna().any().any()
            and all(
                pd.api.types.is_numeric_dtype(dtype)
                for dtype in X_encoded.dtypes
            )
        ),
        "Feature-name order matches estimator": (
            reloaded_feature_names == feature_names
            and reloaded_model.n_features_in_ == len(feature_names)
        ),
        "Reloaded predictions match": np.array_equal(
            reloaded_predictions, y_pred
        ),
        "Reloaded probabilities match": np.allclose(
            reloaded_probabilities, y_probability
        ),
        "Probabilities are within [0, 1]": (
            np.all(reloaded_probabilities >= 0)
            and np.all(reloaded_probabilities <= 1)
        ),
        "Feature importances sum to 1": np.isclose(
            reloaded_model.feature_importances_.sum(), 1.0
        ),
        "Reported metrics are finite": all(
            np.isfinite(value)
            for metric_group in [
                overall_metrics,
                planned_metrics,
                unplanned_metrics,
            ]
            for value in metric_group.values()
        ),
        "Model artifact is non-empty": MODEL_PATH.stat().st_size > 0,
        "Feature-name artifact is non-empty": (
            FEATURE_NAMES_PATH.stat().st_size > 0
        ),
    }

    print_header("SAVED ARTIFACTS")
    logger.info(f"- Model: {MODEL_PATH}")
    logger.info(f"- Feature names: {FEATURE_NAMES_PATH}")

    print_header("VERIFICATION")
    for label, passed in checks.items():
        logger.info(f"- {'PASS' if passed else 'FAIL'}: {label}")

    failed_checks = [label for label, passed in checks.items() if not passed]
    if failed_checks:
        logger.error(f"Verification failed: {', '.join(failed_checks)}")
        sys.exit(1)

    logger.info("- All training and artifact verification checks passed.")


if __name__ == "__main__":
    main()
