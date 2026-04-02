"""
features.py — Feature engineering pipeline

This module is the SINGLE SOURCE OF TRUTH for feature preprocessing.
It is imported identically during:
  - Training (churn_model/train.py)
  - Batch inference (churn_model/predict.py)
  - Model serving (the pyfunc wrapper uses this pipeline)

Keeping preprocessing in one place prevents train/serving skew — one of the
most common sources of production ML bugs.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_feature_pipeline(config: dict) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer from the shared config.

    The pipeline handles:
    - Numeric features: median imputation (for TotalCharges nulls) + standard scaling
    - Categorical features: constant imputation + one-hot encoding

    Parameters
    ----------
    config : dict
        Loaded from config.yml. Must have a 'feature_columns' key with
        'numeric' and 'categorical' sub-keys.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor. Fit it on training data, then transform
        both train and serve payloads with the same fitted instance.

    Example
    -------
    >>> import yaml
    >>> with open("src/utils/config.yml") as f:
    ...     cfg = yaml.safe_load(f)
    >>> preprocessor = build_feature_pipeline(cfg)
    >>> X_train_transformed = preprocessor.fit_transform(X_train)
    >>> X_test_transformed  = preprocessor.transform(X_test)
    """
    numeric_features = config["feature_columns"]["numeric"]
    categorical_features = config["feature_columns"]["categorical"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",   # Drop customerID and any other columns not in feature lists
    )

    return preprocessor


def prepare_dataframe(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Prepare a raw DataFrame for training or inference.

    - Drops the ID column
    - Separates features from the target label (if present)
    - Converts SeniorCitizen int → str for consistent categorical handling

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame (from Spark .toPandas() or a CSV read).
    config : dict
        Loaded from config.yml.

    Returns
    -------
    X : pd.DataFrame
        Feature columns only.
    y : pd.Series or None
        Target column if present in df, else None (inference mode).
    """
    df = df.copy()

    # SeniorCitizen is stored as int (0/1) but logically categorical
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    target_col = config.get("target_column", "Churn")
    id_col = config.get("id_column", "customerID")

    # Separate target
    y = None
    if target_col in df.columns:
        positive_class = config.get("positive_class", "Yes")
        y = (df[target_col] == positive_class).astype(int)

    # Drop non-feature columns
    drop_cols = [id_col, target_col] if target_col in df.columns else [id_col]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y


def run_data_quality_checks(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run lightweight data quality checks on raw input data.

    Returns a DataFrame with columns: check_name, passed, actual_value, details.
    Used in 06_production_checklist.ipynb and as the first task in the retrain job.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    config : dict
        Loaded from config.yml. Uses 'data_quality' section for thresholds.

    Returns
    -------
    pd.DataFrame with check results.
    """
    dq_cfg = config.get("data_quality", {})
    target_col = config.get("target_column", "Churn")
    positive_class = config.get("positive_class", "Yes")

    results = []

    def _check(name: str, passed: bool, actual, details: str = ""):
        results.append({
            "check_name": name,
            "passed": passed,
            "actual_value": str(actual),
            "details": details,
        })

    # 1. Row count
    n_rows = len(df)
    _check("row_count_positive", n_rows > 0, n_rows, "Dataset must have at least 1 row")

    # 2. Required columns present
    required_cols = (
        config["feature_columns"]["numeric"]
        + config["feature_columns"]["categorical"]
        + [target_col]
    )
    missing_cols = [c for c in required_cols if c not in df.columns]
    _check("required_columns_present", len(missing_cols) == 0, missing_cols,
           f"Missing: {missing_cols}" if missing_cols else "All required columns present")

    # 3. Numeric range checks
    for col, bounds in dq_cfg.items():
        if col not in df.columns or not isinstance(bounds, dict):
            continue
        if "min" in bounds and "max" in bounds:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            in_range = series.between(bounds["min"], bounds["max"]).all()
            actual_range = f"[{series.min():.2f}, {series.max():.2f}]"
            expected_range = f"[{bounds['min']}, {bounds['max']}]"
            _check(f"{col}_in_range", bool(in_range), actual_range,
                   f"Expected {expected_range}")

    # 4. Churn rate sanity check
    if target_col in df.columns:
        churn_rate = (df[target_col] == positive_class).mean()
        bounds = dq_cfg.get("expected_churn_rate", {"min": 0.05, "max": 0.70})
        in_range = bounds["min"] <= churn_rate <= bounds["max"]
        _check("churn_rate_in_range", in_range, f"{churn_rate:.1%}",
               f"Expected between {bounds['min']:.0%} and {bounds['max']:.0%}")

    # 5. Null rate check for critical columns
    for col in ["tenure", "MonthlyCharges"]:
        if col in df.columns:
            null_rate = df[col].isna().mean()
            _check(f"{col}_null_rate_low", null_rate < 0.05, f"{null_rate:.1%}",
                   "Null rate should be < 5%")

    return pd.DataFrame(results)
