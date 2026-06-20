# TrafficPulse Codebase Analysis Report

## Executive Summary
This report details critical issues, hardcoded values, code quality concerns, and security/maintainability improvements needed in the TrafficPulse repository.

---

## 🔴 CRITICAL ISSUES

### 1. **Hardcoded File Paths (Windows-specific)**
**Severity: HIGH | Files: `astra_eda.py`, `feature_engineering.py`**

**Issue:**
```python
# astra_eda.py (lines 6-8)
INPUT_CSV = Path(
    r"C:\Users\abhin\Downloads"
    r"\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
)

# feature_engineering.py (lines 9-11)
INPUT_CSV = Path(
    r"C:\Users\abhin\Downloads"
    r"\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
)
```

**Problems:**
- Absolute Windows paths that only work on a specific machine
- Personal username (`abhin`) exposed in codebase
- Will break on any other system (Linux, macOS, different user)
- Not portable across development environments
- Makes repository unusable without manual modification

**Recommended Fix:**
```python
# Use environment variables or config files
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
INPUT_CSV = Path(os.getenv("ASTRAM_DATA_PATH", "./data/astram_events.csv"))
```

Create `.env` file:
```
ASTRAM_DATA_PATH=./data/astram_events.csv
```

---

### 2. **Hardcoded Simulation Parameters**
**Severity: MEDIUM | File: `models/simulation_engine.py` (lines 24-31)**

**Issue:**
```python
TIME_AMPLITUDE = 0.5          # Hardcoded
PEAK_OFFSET_HOURS = 6         # Hardcoded
DIRECT_IMPACT_MULTIPLIER = 2.5   # Hardcoded
ONE_HOP_SPILLOVER = 0.3       # Hardcoded
TWO_HOP_SPILLOVER = 0.1       # Hardcoded
DIVERSION_RELIEF_FACTOR = 0.6 # Hardcoded
DIVERSION_LOAD_FACTOR = 1.15  # Hardcoded
```

**Problems:**
- No way to adjust simulation parameters without code changes
- Making these tunable is essential for experimentation
- Comments acknowledge these are "tunable" but they're not

**Recommended Fix:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationConfig:
    time_amplitude: float = 0.5
    peak_offset_hours: float = 6
    direct_impact_multiplier: float = 2.5
    one_hop_spillover: float = 0.3
    two_hop_spillover: float = 0.1
    diversion_relief_factor: float = 0.6
    diversion_load_factor: float = 1.15
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SimulationConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

# Usage
CONFIG = SimulationConfig()

def get_live_state(hour_of_day: float, G: nx.Graph, config: Optional[SimulationConfig] = None) -> dict[str, float]:
    if config is None:
        config = CONFIG
    # ... rest of function
```

---

### 3. **No Error Handling in Data Loading**
**Severity: MEDIUM | Files: `train_closure_classifier.py`, `train_severity_and_duration.py`, `feature_engineering.py`**

**Issue:**
```python
df = pd.read_csv(INPUT_CSV)
validate_source(df)
```

**Problems:**
- If CSV file doesn't exist, cryptic pandas error
- No validation that required columns exist before read
- Silent failures if data format changes
- No logging of what's being loaded

**Recommended Fix:**
```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_validate_csv(csv_path: Path, expected_rows: int, expected_columns: int) -> pd.DataFrame:
    """Load and validate CSV with comprehensive error handling."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    if not csv_path.suffix == ".csv":
        raise ValueError(f"Expected CSV file, got: {csv_path.suffix}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {csv_path}")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise
    
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} rows, got {len(df):,}")
    
    if len(df.columns) != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns, got {len(df.columns)}")
    
    return df
```

---

## 🟡 HARDCODED VALUES & MAGIC NUMBERS

### 4. **Magic Numbers in Feature Engineering**
**File: `feature_engineering.py`**

```python
# Line 93: Peak hours hardcoded
df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19]).astype("int8")

# Lines 144-147: Severity calculation weights hardcoded
df["y_severity"] = (
    0.6 * df["y_closure"]
    + 0.3 * (df["duration_hours"] / max_duration)
    + 0.05
)
```

**Recommended Fix:**
```python
# config.py
PEAK_HOURS = [8, 9, 10, 17, 18, 19]
SEVERITY_WEIGHTS = {
    "closure": 0.6,
    "duration": 0.3,
    "baseline": 0.05,
}

# feature_engineering.py
from config import PEAK_HOURS, SEVERITY_WEIGHTS

df["is_peak_hour"] = df["hour"].isin(PEAK_HOURS).astype("int8")
df["y_severity"] = (
    SEVERITY_WEIGHTS["closure"] * df["y_closure"]
    + SEVERITY_WEIGHTS["duration"] * (df["duration_hours"] / max_duration)
    + SEVERITY_WEIGHTS["baseline"]
)
```

---

### 5. **Hardcoded Expected Values for Validation**
**File: Multiple files**

```python
# feature_engineering.py
EXPECTED_ROWS = 8173
EXPECTED_ORIGINAL_COLUMNS = 46
EXPECTED_OUTPUT_COLUMNS = 57

# train_closure_classifier.py
EXPECTED_ROWS = 8173

# train_severity_and_duration.py
EXPECTED_ROWS = 8173
EXPECTED_CAUSES = 17
```

**Problems:**
- If dataset changes, these need updating in multiple files
- Brittle validation that won't scale
- Duplicated constants across modules

**Recommended Fix:**
```python
# constants.py
class DatasetConstants:
    """Centralized dataset metadata."""
    EXPECTED_ROWS = 8173
    EXPECTED_ORIGINAL_COLUMNS = 46
    EXPECTED_OUTPUT_COLUMNS = 57
    EXPECTED_EVENT_CAUSES = 17

# Import everywhere
from constants import DatasetConstants
assert df.shape == (DatasetConstants.EXPECTED_ROWS, DatasetConstants.EXPECTED_OUTPUT_COLUMNS)
```

---

### 6. **Hardcoded Model Hyperparameters**
**File: `train_closure_classifier.py` (lines 142-150) & `train_severity_and_duration.py` (lines 130-137)**

```python
# train_closure_classifier.py
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
)

# train_severity_and_duration.py
severity_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
```

**Problems:**
- No way to experiment with hyperparameters
- Should be configurable, not hardcoded
- Test/train split (0.20) also hardcoded

**Recommended Fix:**
```python
# model_config.py
from dataclasses import dataclass

@dataclass
class ClassifierConfig:
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    random_state: int = 42
    test_size: float = 0.20

# train_closure_classifier.py
from model_config import ClassifierConfig

config = ClassifierConfig()
model = GradientBoostingClassifier(**vars(config))
```

---

### 7. **Hardcoded Geohashes**
**File: `build_road_graph.py` (lines 70-79)**

```python
raw_geohashes = [
    "tumh", "tumj", "tumc", "tumf", "tumk", "tume", "tumg", "tums",
    "tumm", "tumy", "tumz", "tukn", "tukq", "tukr", "tuks", "tukt",
    # ... more hardcoded values
]
```

**Problems:**
- Large list hardcoded in source
- Should be in external configuration or database
- No way to add new geohashes without code change

**Recommended Fix:**
```python
# geohashes.json or geohashes.csv
[
    "tumh", "tumj", "tumc", "tumf", "tumk", "tume", "tumg", "tums",
    ...
]

# build_road_graph.py
import json
from pathlib import Path

def load_geohashes(config_path: Path = Path("geohashes.json")) -> list[str]:
    with config_path.open() as f:
        return json.load(f)

raw_geohashes = load_geohashes()
```

---

## 🟠 CODE QUALITY & MAINTAINABILITY ISSUES

### 8. **No Logging Infrastructure**
**Severity: MEDIUM | All files**

**Issue:**
All scripts use `print()` for output. No structured logging.

```python
print(f"Created {OUTPUT_REPORT}")
print(f"Rows analyzed: {total_rows:,}")
```

**Problems:**
- Can't control log level
- No timestamps
- Hard to integrate with monitoring
- Difficult to debug in production

**Recommended Fix:**
```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/trafficpulse_{datetime.now().isoformat()}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Created {OUTPUT_REPORT}")
logger.info(f"Rows analyzed: {total_rows:,}")
```

---

### 9. **Missing Type Hints and Documentation**
**Severity: MEDIUM | Multiple files**

**Issue:**
Many functions lack type hints or docstrings:

```python
# build_road_graph.py
def neighbors(geohash: str) -> dict[str, str]:  # Good!
    """Return all 8 neighbours of *geohash*..."""  # Good!

# But many missing in other files
```

**Recommended Fix:**
```python
from typing import Optional, Dict, List, Tuple

def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Calculate binary classification metrics with NaN handling for subgroups.
    
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Model predictions.
    y_probability : np.ndarray
        Model prediction probabilities.
    
    Returns
    -------
    dict[str, float]
        Dictionary with 'precision', 'recall', 'f1', 'auc_roc' keys.
    
    Raises
    ------
    ValueError
        If input arrays have mismatched lengths.
    """
```

---

### 10. **Missing Requirements Versioning**
**Severity: MEDIUM | File: `requirements.txt`**

**Issue:**
```
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn>=1.5,<2
```

**Problems:**
- Version ranges too broad (numpy >= 1.26 is very old)
- No pinned versions for reproducibility
- May cause compatibility issues in the future

**Recommended Fix:**
```
# requirements.txt (with pinned versions for production)
pandas==2.2.0
numpy==1.26.0
scikit-learn==1.5.0
networkx==3.4
geohash2==1.1
streamlit==1.58.0
folium==0.20.0
streamlit-folium==0.27.0

# requirements-dev.txt (with looser constraints for development)
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn>=1.5,<2
pytest>=7.0
black>=23.0
flake8>=6.0
mypy>=1.0
```

---

### 11. **No Tests**
**Severity: HIGH | Entire repository**

**Issue:**
No unit tests, integration tests, or test files present.

**Recommended Fix:**
```python
# tests/test_feature_engineering.py
import pytest
import pandas as pd
from feature_engineering import validate_features, main

def test_validate_features_correct_shape():
    """Test that features have expected shape."""
    df = pd.read_csv("features_all_8173.csv")
    assert df.shape == (8173, 57)

def test_validate_features_no_nulls():
    """Test that engineered features have no null values."""
    df = pd.read_csv("features_all_8173.csv")
    engineered_cols = ["hour", "day_of_week", "is_weekend", "is_peak_hour"]
    assert not df[engineered_cols].isna().any().any()

def test_peak_hours_binary():
    """Test is_peak_hour contains only 0 and 1."""
    df = pd.read_csv("features_all_8173.csv")
    assert set(df["is_peak_hour"].unique()).issubset({0, 1})
```

---

### 12. **Missing .gitignore**
**Severity: MEDIUM | Repository root**

**Issue:**
No `.gitignore` file. This will commit:
- `__pycache__/` directories
- `.pkl` model files (large)
- CSV data files
- `.pyc` files
- `.env` with sensitive paths

**Recommended Fix:**
```
# .gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Models (too large)
*.pkl
*.pickle

# Data files
*.csv
data/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment variables
.env
.env.local

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
logs/
*.log
```

---

### 13. **Missing README Documentation**
**Severity: MEDIUM | Repository root**

**Issue:**
No README explaining:
- What this project does
- How to set it up
- How to run scripts
- Expected file formats

**Recommended Fix:** Create comprehensive README.md

---

### 14. **Validation Redundancy and Brittleness**
**Severity: MEDIUM | Multiple files**

**Issue:**
```python
# feature_engineering.py lines 46-52
assert df.shape == (EXPECTED_ROWS, EXPECTED_OUTPUT_COLUMNS)
assert int(planned_mask.sum()) == 467
assert int(unplanned_mask.sum()) == 7706
assert not df[ENGINEERED_COLUMNS].isna().any().any()

# These exact values repeated in train_closure_classifier.py
assert len(df) == EXPECTED_ROWS, f"Expected 8,173 rows, found {len(df):,}"
```

**Problems:**
- Assertions will crash production silently
- Should use logging + graceful error handling
- Hardcoded numbers make validation brittle

**Recommended Fix:**
```python
def validate_with_logging(df: pd.DataFrame, expected_rows: int) -> bool:
    """Validate with detailed logging instead of assertions."""
    issues = []
    
    if len(df) != expected_rows:
        issues.append(f"Row count mismatch: expected {expected_rows}, got {len(df)}")
    
    if df[ENGINEERED_COLUMNS].isna().any().any():
        null_cols = df[ENGINEERED_COLUMNS].columns[df[ENGINEERED_COLUMNS].isna().any()]
        issues.append(f"Found nulls in engineered columns: {null_cols.tolist()}")
    
    if issues:
        logger.error("Validation failed with issues:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("All validations passed")
    return True
```

---

## 🔵 MISSING FEATURES & IMPROVEMENTS

### 15. **No Configuration Management**
**Severity: MEDIUM**

Need a centralized config system instead of scattered hardcoded values.

**Recommended:**
```python
# config.py
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class TrafficPulseConfig:
    """Centralized configuration for all TrafficPulse modules."""
    
    # Paths
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    
    # Data
    expected_rows: int = 8173
    peak_hours: list = None
    
    # Model hyperparameters
    classifier_params: dict = None
    regressor_params: dict = None
    
    # Simulation parameters
    simulation_params: dict = None
    
    def __post_init__(self):
        if self.peak_hours is None:
            self.peak_hours = [8, 9, 10, 17, 18, 19]
        if self.classifier_params is None:
            self.classifier_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
            }
    
    @classmethod
    def from_json(cls, config_path: Path) -> "TrafficPulseConfig":
        """Load configuration from JSON file."""
        with open(config_path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)
```

---

### 16. **No Input Data Validation Schema**
**Severity: MEDIUM**

CSV files should have a schema definition.

**Recommended:**
```python
# schema.py
from dataclasses import dataclass

@dataclass
class ColumnSchema:
    name: str
    dtype: str
    required: bool = True
    nullable: bool = False

ASTRAM_SCHEMA = [
    ColumnSchema("id", "int64", required=True),
    ColumnSchema("event_type", "string", required=True),
    ColumnSchema("event_cause", "string", required=True),
    ColumnSchema("start_datetime", "datetime64[ns]", required=True),
    ColumnSchema("end_datetime", "datetime64[ns]", required=False, nullable=True),
    # ... more columns
]

def validate_schema(df: pd.DataFrame, schema: list) -> bool:
    """Validate dataframe against schema."""
    for col_def in schema:
        if col_def.required and col_def.name not in df.columns:
            raise ValueError(f"Missing required column: {col_def.name}")
        # ... more validation
```

---

### 17. **Missing Project Structure**
**Severity: MEDIUM**

Current structure is flat. Should be organized:

```
TrafficPulse/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── simulation_engine.py
│   │   └── training/
│   │       ├── train_classifier.py
│   │       └── train_regressor.py
│   └── data/
│       ├── __init__.py
│       ├── feature_engineering.py
│       └── validation.py
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   └── test_models.py
├── scripts/
│   ├── build_graph.py
│   ├── eda_analysis.py
│   └── run_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── logs/
├── requirements.txt
├── requirements-dev.txt
├── config.json
├── .env.example
├── .gitignore
├── setup.py
└── README.md
```

---

### 18. **No Data Version Control**
**Severity: HIGH**

No tracking of dataset versions or lineage.

**Recommended:**
```python
# data_versioning.py
import hashlib
import json
from pathlib import Path
from datetime import datetime

class DataVersion:
    """Track dataset versions and lineage."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.hash = self._compute_hash()
        self.timestamp = datetime.now().isoformat()
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of data file."""
        sha256_hash = hashlib.sha256()
        with open(self.data_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def save_metadata(self, output_path: Path) -> None:
        """Save version metadata to JSON."""
        metadata = {
            "file": str(self.data_path),
            "hash": self.hash,
            "timestamp": self.timestamp,
            "size_bytes": self.data_path.stat().st_size,
        }
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
```

---

### 19. **Exception Handling Issues**
**Severity: MEDIUM | File: `build_road_graph.py`**

```python
try:
    geohash2.decode_exactly(gh)
except Exception as exc:
    print(f"  [SKIP] invalid geohash '{gh}': {exc}")
```

**Problems:**
- Bare `except Exception` catches too much
- Should be specific exceptions
- Print-based error handling instead of logging

**Recommended Fix:**
```python
try:
    geohash2.decode_exactly(gh)
except (ValueError, TypeError) as exc:
    logger.warning(f"Skipping invalid geohash '{gh}': {exc}")
    continue
except Exception as exc:
    logger.error(f"Unexpected error processing geohash '{gh}': {exc}")
    raise
```

---

### 20. **No Model Versioning or Tracking**
**Severity: HIGH**

Models are just pickled without metadata about:
- Training date
- Data version used
- Hyperparameters
- Performance metrics
- Git commit hash

**Recommended:**
```python
# model_registry.py
import pickle
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

@dataclass
class ModelMetadata:
    name: str
    model_type: str
    training_date: str
    data_version: str
    hyperparameters: dict
    metrics: dict
    git_commit: str
    python_version: str
    scikit_learn_version: str

class ModelRegistry:
    """Manage model artifacts with metadata."""
    
    def save_model(
        self,
        model,
        metadata: ModelMetadata,
        output_dir: Path,
    ) -> None:
        """Save model and metadata together."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / f"{metadata.name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = output_dir / f"{metadata.name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        
        logger.info(f"Saved model and metadata to {output_dir}")
    
    def load_model(self, model_dir: Path, model_name: str):
        """Load model and verify metadata."""
        model_path = model_dir / f"{model_name}_model.pkl"
        metadata_path = model_dir / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded {metadata['name']} trained on {metadata['training_date']}")
        return model, metadata
```

---

## 📋 SUMMARY OF CHANGES NEEDED

| Priority | Issue | File(s) | Action |
|----------|-------|---------|--------|
| 🔴 CRITICAL | Hardcoded Windows paths | astra_eda.py, feature_engineering.py | Use env vars / config file |
| 🔴 CRITICAL | No error handling | train_*.py, feature_engineering.py | Add try-except and validation |
| 🔴 HIGH | Hardcoded hyperparameters | train_*.py | Move to config |
| 🔴 HIGH | No tests | All files | Add pytest suite |
| 🔴 HIGH | No data versioning | All files | Add DVC or metadata tracking |
| 🟡 MEDIUM | Hardcoded simulation params | simulation_engine.py | Make configurable |
| 🟡 MEDIUM | No logging | All files | Add logging infrastructure |
| 🟡 MEDIUM | Missing requirements pinning | requirements.txt | Pin versions |
| 🟡 MEDIUM | No project structure | Repository | Reorganize directories |
| 🟡 MEDIUM | No .gitignore | Repository root | Create .gitignore |
| 🟢 LOW | Missing documentation | README.md | Create comprehensive docs |
| 🟢 LOW | Sparse type hints | All files | Add type annotations |

---

## Next Steps

1. **Immediate (Week 1):**
   - Fix hardcoded file paths (move to environment config)
   - Add .gitignore file
   - Create requirements-dev.txt with pinned versions

2. **Short term (Week 2-3):**
   - Add logging infrastructure
   - Create config management system
   - Add basic input validation

3. **Medium term (Week 4-6):**
   - Add comprehensive unit tests
   - Implement model versioning/metadata tracking
   - Reorganize project structure
   - Add CI/CD pipeline

4. **Long term:**
   - Add data versioning (DVC)
   - Implement monitoring/alerting
   - Add API layer for model serving
