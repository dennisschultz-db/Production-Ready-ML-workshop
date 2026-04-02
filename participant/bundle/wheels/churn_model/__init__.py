"""
churn_model —  Production-Ready ML Systems Workshop
Core ML package for the Telco Customer Churn prediction model.

Modules:
  features  — Feature engineering pipeline (shared between training and serving)
  train     — Model training with MLflow tracking
  evaluate  — Model evaluation, run selection, and deployment gate
  predict   — Batch inference against registered models
"""

__version__ = "0.1.0"
