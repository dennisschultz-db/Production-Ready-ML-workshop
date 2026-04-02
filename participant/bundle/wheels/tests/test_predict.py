"""
test_predict.py — Validation tests for churn_model/predict.py

Tests the prediction output schema and value constraints without
requiring a live Databricks workspace.

Run with:
    pytest src/tests/test_predict.py -v
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import os
import sys
from unittest.mock import patch, MagicMock

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the wheels directory (parent of tests/) to sys.path so churn_model can be imported
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

# From tests/ -> wheels/ -> bundle/ -> participant/ -> session1/config.yml
CONFIG_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "common", "config.yml")



@pytest.fixture
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_predictions():
    """Simulate what run_batch_inference returns."""
    return pd.DataFrame({
        "customerID":    ["C001", "C002", "C003"],
        "tenure":        [1, 34, 2],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges":  [29.85, 1889.5, 108.15],
        "Contract":      ["Month-to-month", "One year", "Month-to-month"],
        "Churn":         ["No", "No", "Yes"],
        "prediction":    ["No", "No", "Yes"],
        "prediction_prob": [0.12, 0.08, 0.81],
        "model_id":      ["workshop.jdoe.churn_classifier_v1"] * 3,
        "inference_ts":  ["2026-05-07T09:00:00"] * 3,
    })


class TestPredictionSchema:

    def test_required_output_columns_present(self, sample_predictions):
        required = {"prediction", "prediction_prob", "model_id", "inference_ts"}
        assert required.issubset(set(sample_predictions.columns))

    def test_prediction_values_are_valid(self, sample_predictions, config):
        positive_class = config.get("positive_class", "Yes")
        valid_values = {positive_class, "No"}
        invalid = sample_predictions[~sample_predictions["prediction"].isin(valid_values)]
        assert len(invalid) == 0, f"Invalid prediction values:\n{invalid['prediction']}"

    def test_prediction_prob_in_range(self, sample_predictions):
        probs = sample_predictions["prediction_prob"]
        assert probs.between(0.0, 1.0).all(), "Probabilities must be in [0, 1]"

    def test_model_id_not_null(self, sample_predictions):
        assert sample_predictions["model_id"].notna().all()
        assert (sample_predictions["model_id"] != "").all()

    def test_inference_ts_not_null(self, sample_predictions):
        assert sample_predictions["inference_ts"].notna().all()

    def test_high_prob_predicts_churn(self, sample_predictions, config):
        """Rows with prediction_prob > 0.5 should predict the positive class."""
        positive_class = config.get("positive_class", "Yes")
        high_prob = sample_predictions[sample_predictions["prediction_prob"] > 0.5]
        if len(high_prob) > 0:
            assert (high_prob["prediction"] == positive_class).all(), (
                "Rows with prob > 0.5 should predict churn"
            )

    def test_low_prob_predicts_no_churn(self, sample_predictions, config):
        """Rows with prediction_prob < 0.5 should predict the negative class."""
        positive_class = config.get("positive_class", "Yes")
        low_prob = sample_predictions[sample_predictions["prediction_prob"] < 0.5]
        if len(low_prob) > 0:
            assert (low_prob["prediction"] != positive_class).all(), (
                "Rows with prob < 0.5 should predict no churn"
            )


class TestDriftIndicators:
    """Tests that simulate detecting drift in prediction distributions."""

    def test_churn_rate_increase_is_detectable(self, config):
        """Verify we can detect when prediction churn rate shifts significantly."""
        positive_class = config.get("positive_class", "Yes")

        baseline_predictions = pd.DataFrame({
            "prediction": [positive_class] * 26 + ["No"] * 74,
            "prediction_prob": np.random.beta(2, 5, 100),
        })

        drifted_predictions = pd.DataFrame({
            "prediction": [positive_class] * 45 + ["No"] * 55,
            "prediction_prob": np.random.beta(4, 3, 100),
        })

        baseline_rate = (baseline_predictions["prediction"] == positive_class).mean()
        drifted_rate = (drifted_predictions["prediction"] == positive_class).mean()

        rate_change = abs(drifted_rate - baseline_rate)
        assert rate_change > 0.10, (
            f"Expected detectable rate change > 10%, got {rate_change:.1%}"
        )
