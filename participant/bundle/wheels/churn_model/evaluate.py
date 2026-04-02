"""
evaluate.py — Model evaluation, run selection, and deployment gate

Three main responsibilities:
  1. get_best_run()       — Query an MLflow experiment to find the best run
  2. evaluate_gate()      — Compare a challenger vs champion; decide whether to deploy
  3. run_data_quality_checks() — Re-exported from features.py for convenience

Used in:
  - src/session1/04_mlflow_experiment.ipynb  (get_best_run)
  - src/session2/03_deployment_gate.ipynb    (evaluate_gate)
  - src/jobs/03_gate_and_register.ipynb      (both)
"""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# Re-export for convenience so callers only need to import from evaluate
from churn_model.features import run_data_quality_checks  # noqa: F401


@dataclass
class GateResult:
    """Result returned by evaluate_gate()."""
    should_deploy: bool
    reason: str
    challenger_metrics: dict
    champion_metrics: dict
    metrics_delta: dict


def get_best_run(
    experiment_name: str,
    metric: str = "test_f1",
) -> tuple[str, str, dict]:
    """
    Return the run_id, model_type, and metrics for the best run in an experiment.

    Parameters
    ----------
    experiment_name : str
        Full MLflow experiment path, e.g. '/Users/jane.doe@company.com/churn_experiment'.
    metric : str
        The metric to maximise. Must be logged by run_training().

    Returns
    -------
    (run_id, model_type, metrics_dict)

    Example
    -------
    >>> run_id, model_type, metrics = get_best_run(
    ...     "/Users/you@company.com/churn_experiment", metric="test_f1"
    ... )
    >>> print(f"Best: {model_type}  F1={metrics['test_f1']:.4f}")
    """
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment not found: '{experiment_name}'. "
            "Have you run 04_mlflow_experiment.ipynb first?"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No finished runs found in experiment '{experiment_name}'."
        )

    best_run: Run = runs[0]
    metrics = best_run.data.metrics
    model_type = best_run.data.params.get("model_type", "unknown")

    return best_run.info.run_id, model_type, metrics


def evaluate_gate(
    challenger_run_id: str,
    champion_model_uri: str,
    test_data,           # pd.DataFrame — raw, with target column
    config: dict,
    threshold: float = 0.05,
) -> GateResult:
    """
    Deployment gate: should we replace the champion with the challenger?

    The gate passes (should_deploy=True) if:
      challenger_f1 >= champion_f1 - threshold

    That is, the challenger must not be meaningfully worse than the champion.
    It does NOT need to be strictly better — we allow deployment if performance
    is within the threshold, so retraining on fresh data always moves forward.

    Parameters
    ----------
    challenger_run_id : str
        MLflow run_id of the newly trained challenger model.
    champion_model_uri : str
        URI of the current champion model, e.g.
        'models:/workshop.john_doe.churn_classifier@champion'
    test_data : pd.DataFrame or Spark DataFrame
        Held-out test data including the target column.
        If Spark DataFrame, it will be converted via .toPandas().
    config : dict
        Loaded from config.yml.
    threshold : float
        Maximum allowed F1 drop from champion to challenger. Default 0.05 (5%).

    Returns
    -------
    GateResult

    Example
    -------
    >>> result = evaluate_gate(
    ...     challenger_run_id="abc123",
    ...     champion_model_uri="models:/workshop.jdoe.churn_classifier@champion",
    ...     test_data=test_df,
    ...     config=config,
    ...     threshold=0.05,
    ... )
    >>> if result.should_deploy:
    ...     print("Promote challenger to champion!")
    ... else:
    ...     print(f"Gate failed: {result.reason}")
    """
    from sklearn.metrics import f1_score, roc_auc_score, recall_score
    from churn_model.features import prepare_dataframe

    mlflow.set_registry_uri("databricks-uc")

    # Convert Spark DataFrame if needed
    if hasattr(test_data, "toPandas"):
        test_data = test_data.toPandas()

    X_test, y_test = prepare_dataframe(test_data, config)

    # Load challenger from MLflow run
    challenger_model = mlflow.sklearn.load_model(f"runs:/{challenger_run_id}/model")
    y_pred_c = challenger_model.predict(X_test)
    y_prob_c = challenger_model.predict_proba(X_test)[:, 1]
    challenger_metrics = {
        "f1":      f1_score(y_test, y_pred_c, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob_c),
        "recall":  recall_score(y_test, y_pred_c, zero_division=0),
    }

    # Load champion from UC registry
    try:
        champion_model = mlflow.sklearn.load_model(champion_model_uri)
        y_pred_ch = champion_model.predict(X_test)
        y_prob_ch = champion_model.predict_proba(X_test)[:, 1]
        champion_metrics = {
            "f1":      f1_score(y_test, y_pred_ch, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob_ch),
            "recall":  recall_score(y_test, y_pred_ch, zero_division=0),
        }
    except Exception as e:
        # No champion exists yet (first deployment) — always pass the gate
        print(f"No champion found ({e}). Treating as first deployment — gate passes.")
        champion_metrics = {"f1": 0.0, "roc_auc": 0.0, "recall": 0.0}

    metrics_delta = {
        k: challenger_metrics[k] - champion_metrics[k]
        for k in challenger_metrics
    }

    f1_drop = champion_metrics["f1"] - challenger_metrics["f1"]
    should_deploy = f1_drop <= threshold

    if should_deploy:
        reason = (
            f"Gate PASSED. Challenger F1={challenger_metrics['f1']:.4f} "
            f"vs Champion F1={champion_metrics['f1']:.4f} "
            f"(drop={f1_drop:.4f} ≤ threshold={threshold})."
        )
    else:
        reason = (
            f"Gate FAILED. Challenger F1={challenger_metrics['f1']:.4f} "
            f"vs Champion F1={champion_metrics['f1']:.4f} "
            f"(drop={f1_drop:.4f} > threshold={threshold}). "
            "Challenger will NOT be promoted."
        )

    print(reason)

    return GateResult(
        should_deploy=should_deploy,
        reason=reason,
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
        metrics_delta=metrics_delta,
    )
