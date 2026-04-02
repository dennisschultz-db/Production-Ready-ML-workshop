""" 
test_features.py — Unit tests for churn_model/features.py

These tests run without a Spark session (pure pandas/sklearn).
They validate the feature pipeline and data quality check logic.

Run with:
    pytest src/tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the wheels directory (parent of tests/) to sys.path so churn_model can be imported
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

from churn_model.features import (
    build_feature_pipeline,
    prepare_dataframe,
    run_data_quality_checks,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# From tests/ -> wheels/ -> bundle/ -> participant/ -> session1/config.yml
CONFIG_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "common", "config.yml")


@pytest.fixture
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_df():
    """Minimal valid Telco-like DataFrame for testing."""
    return pd.DataFrame({
        "customerID":       ["C001", "C002", "C003", "C004", "C005"],
        "gender":           ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen":    [0, 1, 0, 0, 1],
        "Partner":          ["Yes", "No", "No", "Yes", "No"],
        "Dependents":       ["No", "No", "Yes", "No", "Yes"],
        "tenure":           [1, 34, 2, 45, 8],
        "PhoneService":     ["No", "Yes", "Yes", "No", "Yes"],
        "MultipleLines":    ["No phone service", "No", "No", "No phone service", "Yes"],
        "InternetService":  ["DSL", "DSL", "DSL", "DSL", "Fiber optic"],
        "OnlineSecurity":   ["No", "Yes", "Yes", "Yes", "No"],
        "OnlineBackup":     ["Yes", "No", "Yes", "No", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes", "No"],
        "TechSupport":      ["No", "No", "No", "Yes", "No"],
        "StreamingTV":      ["No", "No", "No", "No", "No"],
        "StreamingMovies":  ["No", "No", "No", "No", "No"],
        "Contract":         ["Month-to-month", "One year", "Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod":    ["Electronic check", "Mailed check", "Mailed check", "Bank transfer (automatic)", "Electronic check"],
        "MonthlyCharges":   [29.85, 56.95, 53.85, 42.30, 70.70],
        "TotalCharges":     [29.85, 1889.5, 108.15, 1840.75, 151.65],
        "Churn":            ["No", "No", "Yes", "No", "Yes"],
    })


# ---------------------------------------------------------------------------
# Tests for build_feature_pipeline
# ---------------------------------------------------------------------------

class TestBuildFeaturePipeline:

    def test_returns_column_transformer(self, config):
        from sklearn.compose import ColumnTransformer
        preprocessor = build_feature_pipeline(config)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_numeric_and_categorical_transformers(self, config):
        preprocessor = build_feature_pipeline(config)
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names

    def test_fit_transform_shape(self, config, sample_df):
        preprocessor = build_feature_pipeline(config)
        X, y = prepare_dataframe(sample_df, config)
        X_transformed = preprocessor.fit_transform(X)
        # Should have more columns than input due to one-hot encoding
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > len(config["feature_columns"]["numeric"])

    def test_handles_null_totalcharges(self, config, sample_df):
        """TotalCharges can be null for new customers (tenure=0). Should not crash."""
        df_with_nulls = sample_df.copy()
        df_with_nulls.loc[0, "TotalCharges"] = None  # Simulate new customer

        preprocessor = build_feature_pipeline(config)
        X, y = prepare_dataframe(df_with_nulls, config)
        X_transformed = preprocessor.fit_transform(X)
        assert not np.isnan(X_transformed).any(), "NaN values should be imputed"

    def test_no_negative_values_after_numeric_transform(self, config, sample_df):
        """Scaled values can be negative (StandardScaler), but there should be no NaN."""
        preprocessor = build_feature_pipeline(config)
        X, y = prepare_dataframe(sample_df, config)
        X_transformed = preprocessor.fit_transform(X)
        assert not np.isnan(X_transformed).any()

    def test_unknown_categories_handled_at_transform_time(self, config, sample_df):
        """OneHotEncoder with handle_unknown='ignore' should not fail on unseen categories."""
        preprocessor = build_feature_pipeline(config)
        X_train, _ = prepare_dataframe(sample_df, config)
        preprocessor.fit(X_train)

        # Create a row with an unseen category value
        df_new = sample_df.head(1).copy()
        df_new["Contract"] = "Five year"  # Unseen value
        X_new, _ = prepare_dataframe(df_new, config)
        # Should not raise
        X_transformed = preprocessor.transform(X_new)
        assert X_transformed.shape[0] == 1


# ---------------------------------------------------------------------------
# Tests for prepare_dataframe
# ---------------------------------------------------------------------------

class TestPrepareDataframe:

    def test_drops_customer_id(self, config, sample_df):
        X, y = prepare_dataframe(sample_df, config)
        assert "customerID" not in X.columns

    def test_drops_churn_column(self, config, sample_df):
        X, y = prepare_dataframe(sample_df, config)
        assert "Churn" not in X.columns

    def test_returns_binary_target(self, config, sample_df):
        X, y = prepare_dataframe(sample_df, config)
        assert set(y.unique()).issubset({0, 1})

    def test_positive_class_is_yes(self, config, sample_df):
        """Rows with Churn='Yes' should map to y=1."""
        X, y = prepare_dataframe(sample_df, config)
        churn_indices = sample_df[sample_df["Churn"] == "Yes"].index
        assert all(y.loc[churn_indices] == 1)

    def test_senior_citizen_converted_to_str(self, config, sample_df):
        """SeniorCitizen is int in the dataset but should be treated as categorical."""
        X, y = prepare_dataframe(sample_df, config)
        assert X["SeniorCitizen"].dtype == object  # str in pandas

    def test_inference_mode_no_target(self, config, sample_df):
        """If Churn column is absent, y should be None (inference mode)."""
        df_no_label = sample_df.drop(columns=["Churn"])
        X, y = prepare_dataframe(df_no_label, config)
        assert y is None
        assert "customerID" not in X.columns


# ---------------------------------------------------------------------------
# Tests for run_data_quality_checks
# ---------------------------------------------------------------------------

class TestDataQualityChecks:

    def test_clean_data_all_pass(self, config, sample_df):
        results = run_data_quality_checks(sample_df, config)
        assert isinstance(results, pd.DataFrame)
        assert "check_name" in results.columns
        assert "passed" in results.columns
        failed = results[~results["passed"]]
        assert len(failed) == 0, f"Unexpected failures:\n{failed}"

    def test_negative_tenure_fails_range_check(self, config, sample_df):
        df_bad = sample_df.copy()
        df_bad["tenure"] = -1  # Invalid
        results = run_data_quality_checks(df_bad, config)
        tenure_check = results[results["check_name"] == "tenure_in_range"]
        assert len(tenure_check) == 1
        assert not tenure_check["passed"].iloc[0]

    def test_missing_column_fails_check(self, config, sample_df):
        df_missing = sample_df.drop(columns=["MonthlyCharges"])
        results = run_data_quality_checks(df_missing, config)
        col_check = results[results["check_name"] == "required_columns_present"]
        assert not col_check["passed"].iloc[0]

    def test_high_churn_rate_fails(self, config, sample_df):
        df_all_churn = sample_df.copy()
        df_all_churn["Churn"] = "Yes"  # 100% churn — unrealistic
        results = run_data_quality_checks(df_all_churn, config)
        churn_check = results[results["check_name"] == "churn_rate_in_range"]
        assert not churn_check["passed"].iloc[0]

    def test_returns_dataframe_structure(self, config, sample_df):
        results = run_data_quality_checks(sample_df, config)
        expected_cols = {"check_name", "passed", "actual_value", "details"}
        assert expected_cols.issubset(set(results.columns))
