"""
train.py — Model training with MLflow tracking

Trains Logistic Regression, Random Forest, and Gradient Boosted Trees
on the Telco churn dataset. Each model type runs as a separate MLflow run
within the same experiment, making them easy to compare in the UI.

Usage (from a notebook):
    from churn_model.train import run_training
    run_info = run_training(catalog, schema, config, model_type="random_forest")

Usage (from a Databricks job notebook via dbutils.widgets):
    See src/jobs/02_train_models.ipynb
"""

from __future__ import annotations

import os
import time
import yaml
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, accuracy_score,
    classification_report,
)
from dataclasses import dataclass
from typing import Optional

from churn_model.features import build_feature_pipeline, prepare_dataframe


@dataclass
class TrainResult:
    """Result returned by run_training()."""
    run_id: str
    model_type: str
    experiment_id: str
    metrics: dict
    model_uri: str


def _get_classifier(model_type: str, config: dict):
    """Return an unfitted classifier for the given model type."""
    cfg = config["models"].get(model_type, {})

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=cfg.get("C", [1.0])[0],
            max_iter=cfg.get("max_iter", [200])[0],
            solver=cfg.get("solver", ["lbfgs"])[0],
            class_weight=cfg.get("class_weight", ["balanced"])[0],
            random_state=config["training"]["random_state"],
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", [100])[0],
            max_depth=cfg.get("max_depth", [10])[0],
            min_samples_leaf=cfg.get("min_samples_leaf", [1])[0],
            class_weight=cfg.get("class_weight", ["balanced"])[0],
            random_state=config["training"]["random_state"],
            n_jobs=-1,
        )
    elif model_type == "gradient_boosted_trees":
        return GradientBoostingClassifier(
            n_estimators=cfg.get("n_estimators", [100])[0],
            learning_rate=cfg.get("learning_rate", [0.1])[0],
            max_depth=cfg.get("max_depth", [3])[0],
            subsample=cfg.get("subsample", [1.0])[0],
            random_state=config["training"]["random_state"],
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Choose from: logistic_regression, random_forest, gradient_boosted_trees"
        )


def run_training(
    catalog: str,
    schema: str,
    config: dict,
    model_type: str = "random_forest",
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    fe=None,
    training_set=None,
) -> TrainResult:
    """
    Train a single model type and log everything to MLflow.

    Parameters
    ----------
    catalog : str
        Unity Catalog catalog where the participant's schema lives.
    schema : str
        Participant's schema name (e.g. 'john_doe_example_com').
    config : dict
        Loaded from config.yml.
    model_type : str
        One of: 'logistic_regression', 'random_forest', 'gradient_boosted_trees'.
    experiment_name : str, optional
        MLflow experiment path. Defaults to /Users/{current_user}/churn_{schema}.
    run_name : str, optional
        Human-readable name for this MLflow run.
    data : pd.DataFrame, optional
        Pre-loaded DataFrame. Used for unit tests / local development without
        a Spark session.  Takes precedence over training_set if both are given.
    fe : FeatureEngineeringClient, optional
        Databricks Feature Engineering client. When provided, the model is logged
        with fe.log_model() so the serving endpoint auto-fetches features by key.
    training_set : TrainingSet, optional
        Result of fe.create_training_set(). When provided (and data is None),
        loads the training DataFrame via training_set.load_df().

    Returns
    -------
    TrainResult
        Contains run_id, model_uri, metrics, etc.
    """
    # -- Load data --
    if data is not None:
        df = data
    elif training_set is not None:
        df = training_set.load_df().toPandas()
    else:
        raise ValueError(
            "Either 'data' or 'training_set' must be provided. "
            "Pass data=your_dataframe for local/test use, or "
            "fe=fe_client and training_set=training_set for production use."
        )

    X, y = prepare_dataframe(df, config)

    # -- Split --
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    # -- Build pipeline --
    preprocessor = build_feature_pipeline(config)
    classifier = _get_classifier(model_type, config)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # -- Set MLflow experiment --
    if experiment_name is None:
        current_user = os.environ.get("USER", "unknown")
        try:
            from pyspark.sql import SparkSession
            active_spark = SparkSession.getActiveSession()
            if active_spark is not None:
                current_user = active_spark.sql("SELECT current_user()").first()[0]
        except Exception:
            pass
        experiment_name = f"/Users/{current_user}/churn_{schema}"

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_name)

    run_name = run_name or f"{model_type}_{int(time.time())}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log config parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", config["training"]["test_size"])
        mlflow.log_param("random_state", config["training"]["random_state"])

        # Log model hyperparameters
        clf_params = classifier.get_params()
        for k, v in clf_params.items():
            mlflow.log_param(f"clf_{k}", v)

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy":  accuracy_score(y_test, y_pred),
            "test_f1":        f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc":   roc_auc_score(y_test, y_prob),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall":    recall_score(y_test, y_pred, zero_division=0),
            "train_f1":       f1_score(
                y_train, pipeline.predict(X_train), zero_division=0
            ),
            "n_train":        len(X_train),
            "n_test":         len(X_test),
        }

        mlflow.log_metrics(metrics)

        # Log classification report as a text artifact
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
        mlflow.log_text(report, "classification_report.txt")

        # Log the model — use Feature Engineering client when available so the
        # serving endpoint knows how to auto-fetch features by lookup key.
        input_example = X_test.head(5)
        if fe is not None and training_set is not None:
            fe.log_model(
                model=pipeline,
                artifact_path="model",
                flavor=mlflow.sklearn,
                training_set=training_set,
                registered_model_name=None,  # Don't auto-register; done explicitly later
            )
        else:
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=input_example,
                registered_model_name=None,  # Don't auto-register; done explicitly later
            )

        model_uri = f"runs:/{run.info.run_id}/model"

        print(f"\n{'='*60}")
        print(f"  Model:     {model_type}")
        print(f"  Run ID:    {run.info.run_id}")
        print(f"  F1:        {metrics['test_f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['test_roc_auc']:.4f}")
        print(f"  Recall:    {metrics['test_recall']:.4f}")
        print(f"{'='*60}\n")

    return TrainResult(
        run_id=run.info.run_id,
        model_type=model_type,
        experiment_id=run.info.experiment_id,
        metrics=metrics,
        model_uri=model_uri,
    )


def run_all_models(
    catalog: str,
    schema: str,
    config: dict,
    experiment_name: Optional[str] = None,
    fe=None,
    training_set=None,
) -> list[TrainResult]:
    """
    Train all three model types in sequence and return their results.
    Used by the retrain job and the mlflow_experiment notebook.

    Parameters
    ----------
    fe : FeatureEngineeringClient, optional
        Databricks Feature Engineering client. Passed through to run_training().
    training_set : TrainingSet, optional
        Result of fe.create_training_set(). Passed through to run_training().

    Returns
    -------
    list[TrainResult]
        One result per model type, in the order defined in config['models'].
    """
    model_types = list(config["models"].keys())
    results = []
    for model_type in model_types:
        print(f"\nTraining: {model_type}")
        result = run_training(
            catalog=catalog,
            schema=schema,
            config=config,
            model_type=model_type,
            experiment_name=experiment_name,
            fe=fe,
            training_set=training_set,
        )
        results.append(result)
    return results
