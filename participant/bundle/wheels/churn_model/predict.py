"""
predict.py — Batch inference against the registered champion model

Loads the @champion model version from Unity Catalog, runs predictions
over a Spark DataFrame, and writes results (with original features) to
an inference log Delta table. This log is what Lakehouse Monitoring watches.

Usage (from a notebook):
    from churn_model.predict import run_batch_inference
    run_batch_inference(catalog, schema, config)

Usage (from a Databricks job):
    See src/jobs/04_batch_inference.ipynb
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
from typing import Optional


def run_batch_inference(
    catalog: str,
    schema: str,
    config: dict,
    source_table: Optional[str] = None,
    output_table: Optional[str] = None,
    model_alias: str = "champion",
    inference_timestamp: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the champion model and run batch predictions.

    Writes a Delta table with columns:
      - All original feature columns
      - prediction      (str: "Yes" / "No")
      - prediction_prob (float: probability of churn)
      - model_id        (str: registered model name + version)
      - inference_ts    (str: ISO timestamp)
      - Churn           (str: actual label, if present in source — for monitoring)

    Parameters
    ----------
    catalog : str
        Unity Catalog catalog name.
    schema : str
        Schema name where participant's model is registered.
    config : dict
        Loaded from config.yml.
    source_table : str, optional
        Fully qualified source table. Defaults to workshop.`00_shared`.telco_churn.
    output_table : str, optional
        Fully qualified output table. Defaults to {catalog}.{schema}.churn_inference_log.
    model_alias : str
        Model alias to load. Default 'champion'.
    inference_timestamp : str, optional
        ISO timestamp to tag predictions with. Defaults to now.

    Returns
    -------
    pd.DataFrame
        The inference results (same data written to output_table).
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession")

    # Defaults
    if source_table is None:
        source_table = "workshop.`00_shared`.telco_churn"
    if output_table is None:
        output_table = f"`{catalog}`.`{schema}`.churn_inference_log"
    if inference_timestamp is None:
        inference_timestamp = datetime.utcnow().isoformat()

    # Load data
    df_spark = spark.table(source_table)
    df = df_spark.toPandas()

    # Load model
    model_name = f"{catalog}.{schema}.{config['model_name']}"
    model_uri = f"models:/{model_name}@{model_alias}"

    mlflow.set_registry_uri("databricks-uc")
    model = mlflow.sklearn.load_model(model_uri)

    # Get model version for model_id tag
    from mlflow import MlflowClient
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, model_alias)
    model_id = f"{model_name}_v{mv.version}"

    # Prepare features
    from churn_model.features import prepare_dataframe
    X, _ = prepare_dataframe(df, config)

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Assemble output
    positive_class = config.get("positive_class", "Yes")
    negative_class = "No"
    df_out = df.copy()
    df_out["prediction"] = [positive_class if p == 1 else negative_class for p in predictions]
    df_out["prediction_prob"] = probabilities
    df_out["model_id"] = model_id
    df_out["inference_ts"] = inference_timestamp

    # Write to Delta
    df_out_spark = spark.createDataFrame(df_out)
    df_out_spark.write.mode("append").option("mergeSchema", "true").saveAsTable(output_table)

    print(f"Wrote {len(df_out)} predictions to {output_table}")
    print(f"Model: {model_id}")
    predicted_churn_rate = (df_out["prediction"] == positive_class).mean()
    print(f"Predicted churn rate: {predicted_churn_rate:.1%}")

    return df_out
