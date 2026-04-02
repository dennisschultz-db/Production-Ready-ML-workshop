"""
test_train.py — Smoke tests for churn_model/train.py

These tests mock the Spark session to run without a live cluster.
They validate that the training pipeline produces correct outputs
and that MLflow logging is called as expected.

Run with:
    pytest src/tests/test_train.py -v
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

from churn_model.train import _get_classifier, TrainResult

# From tests/ -> wheels/ -> bundle/ -> participant/ -> session1/config.yml
CONFIG_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "common", "config.yml")


@pytest.fixture
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_df():
    """100-row synthetic dataset for fast smoke tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customerID":       [f"C{i:04d}" for i in range(n)],
        "gender":           np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":    np.random.choice([0, 1], n),
        "Partner":          np.random.choice(["Yes", "No"], n),
        "Dependents":       np.random.choice(["Yes", "No"], n),
        "tenure":           np.random.randint(0, 72, n),
        "PhoneService":     np.random.choice(["Yes", "No"], n),
        "MultipleLines":    np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService":  np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":   np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":     np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":      np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":      np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies":  np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":         np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod":    np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges":   np.random.uniform(18, 120, n).round(2),
        "TotalCharges":     np.random.uniform(0, 8000, n).round(2),
        "Churn":            np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })


# ---------------------------------------------------------------------------
# Tests for _get_classifier
# ---------------------------------------------------------------------------

class TestGetClassifier:

    def test_logistic_regression(self, config):
        from sklearn.linear_model import LogisticRegression
        clf = _get_classifier("logistic_regression", config)
        assert isinstance(clf, LogisticRegression)

    def test_random_forest(self, config):
        from sklearn.ensemble import RandomForestClassifier
        clf = _get_classifier("random_forest", config)
        assert isinstance(clf, RandomForestClassifier)

    def test_gradient_boosted_trees(self, config):
        from sklearn.ensemble import GradientBoostingClassifier
        clf = _get_classifier("gradient_boosted_trees", config)
        assert isinstance(clf, GradientBoostingClassifier)

    def test_unknown_model_raises(self, config):
        with pytest.raises(ValueError, match="Unknown model_type"):
            _get_classifier("xgboost", config)


# ---------------------------------------------------------------------------
# Smoke test for run_training (mocked Spark + MLflow)
# ---------------------------------------------------------------------------

class TestRunTraining:
    """
    Tests for run_training() using the data= parameter to bypass Spark.
    This lets us run training tests locally without a Databricks cluster.
    """

    @pytest.mark.parametrize("model_type", [
        "logistic_regression",
        "random_forest",
        "gradient_boosted_trees",
    ])
    def test_training_returns_train_result(self, config, sample_df, model_type):
        """Each model type should complete without error and return a TrainResult."""
        with patch("mlflow.set_registry_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run") as mock_run_ctx, \
             patch("mlflow.log_param"), \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.log_text"), \
             patch("mlflow.sklearn.log_model"):

            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_id_123"
            mock_run.info.experiment_id = "exp_123"
            mock_run_ctx.return_value.__enter__.return_value = mock_run

            from churn_model.train import run_training
            result = run_training(
                catalog="workshop",
                schema="test_user",
                config=config,
                model_type=model_type,
                experiment_name="/Users/test/churn_experiment",
                data=sample_df,   # Bypass Spark — pass data directly
            )

            assert isinstance(result, TrainResult)
            assert result.run_id == "test_run_id_123"
            assert result.model_type == model_type
            assert "test_f1" in result.metrics
            assert "test_roc_auc" in result.metrics

    def test_metrics_are_in_valid_range(self, config, sample_df):
        """All classification metrics should be between 0 and 1."""
        with patch("mlflow.set_registry_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run") as mock_run_ctx, \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.log_text"), \
             patch("mlflow.sklearn.log_model"):

            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_456"
            mock_run.info.experiment_id = "exp_456"
            mock_run_ctx.return_value.__enter__.return_value = mock_run

            from churn_model.train import run_training
            result = run_training(
                catalog="workshop",
                schema="test_user",
                config=config,
                model_type="logistic_regression",
                experiment_name="/Users/test/churn_experiment",
                data=sample_df,   # Bypass Spark — pass data directly
            )

            for metric_name, metric_val in result.metrics.items():
                if metric_name not in ("n_train", "n_test"):
                    assert 0.0 <= metric_val <= 1.0, (
                        f"Metric '{metric_name}' = {metric_val} is out of [0, 1]"
                    )
