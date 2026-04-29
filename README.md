# Production-Ready ML Systems Workshop


|               |                                                                                               |
| --------------- | :-------------------------------------------------------------------------------------------- |
| **Format:**   | Two 2-hour instructor-led sessions with hands-on exercises                                    |
| **Dataset:**  | IBM Telco Customer Churn — binary classification, ~7K rows                                   |
| **Platform:** | Databricks (serverless notebooks, Unity Catalog, MLflow, Model Serving, Lakehouse Monitoring) |

---

## Scenario

Participants take a customer churn prediction model from a messy notebook all the way through to a fully monitored production deployment. In Session 1 they refactor a prototype into a testable Python package, run MLflow experiments to compare Logistic Regression, Random Forest, and Gradient Boosted Trees, register the best model in Unity Catalog with `@champion` and `@serving` aliases, deploy it as a rate-limited REST endpoint via AI Gateway, and use a Foundation Model API to generate a human-readable explanation of each prediction — with the shared prompt versioned in the MLflow Prompt Registry. In Session 2 the focus shifts to automation and resilience: participants deploy a CI/CD pipeline using Declarative Automation Bundles, inject realistic data drift, create a Lakehouse Monitor to detect it, wire an alert, trigger an automated retraining pipeline, and walk through an incident runbook — diagnosing drift, rolling back to the previous champion, and re-triggering retraining.

---

## Session 1 — Building Production-Grade ML (~2 hrs)

> **Pre-session:** Participants run `00_setup_participant.ipynb` before the formal session begins to create their personal schema and verify data access.

| # | Module | Duration | Artifacts Created |
|---|---|---|---|
| 01 | `01_data_exploration` | 15 min | — (Genie Space query, visualizations) |
| 02 | `02_messy_notebook` | 5 min *(instructor demo)* | — |
| 03 | `03_refactored_pipeline` | 20 min | 1 MLflow run, trained sklearn pipeline |
| 04 | `04_mlflow_experiment` | 25 min | Feature Store table `churn_features`, 3 MLflow runs, experiment |
| 05 | `05_register_and_serve` | 25 min | UC model versions `@champion` + `@serving`, serving endpoint |
| 06 | `06_llm_explanation` | 25 min | Personal prompt version in shared Prompt Registry |
| 07 | `07_production_checklist` | 12 min | — (pytest + data quality check results) |
|    | Total | 127 min | |

**What participants leave with:**
- How to structure ML code as a testable, reusable Python package rather than a monolithic notebook
- How to use MLflow to track experiments, compare model types, and maintain lineage from raw data to registered model
- How Unity Catalog Model Registry works: versioning, dual aliases (`@champion` for batch scoring, `@serving` for the endpoint), and governance
- How to expose a model as a governed REST endpoint with AI Gateway rate limiting, inference logging, and an LLM-generated explanation pulled from a versioned Prompt Registry

### Session 1 error-prone areas

| Risk | Where | Mitigation |
|---|---|---|
| REST API cell runs before endpoint is ready | `05` — endpoint creation is async (~5-10 min spin-up) | Instruct participants to check Serving UI for "Ready" status before running the final REST cell |
| `best_run_id` widget not populated | `05` — reads from prior task values or manual widget input | If running standalone (not via job), participants must copy the run ID from `04`'s output |
| LLM gateway cold start | `06` — first call may take 30+ sec | Instructor starts the `01_keep_endpoint_warm notebook` before the workshop begins.  Don't for get to turn it off after the workshop! |
| pip install adds cluster variance | `03`, `04`, `05`, `06` | Pre-installing the `churn_model` wheel in the cluster init script would eliminate this, but the current design is a teaching moment about options. |

---

## Session 2 — MLOps in Production (~2 hrs)

> Session 2 opens with `07_production_checklist` from Session 1 as a 10-minute bridge: "here's what we built — here's what's still missing."

| # | Module | Duration | Artifacts Created |
|---|---|---|---|
| — | `07_production_checklist` *(bridge)* | 10 min | — |
| 01 | `01_cicd_overview` | 13 min | Deployed bundle, running retrain job (async) |
| 02 | `02_simulate_drift` | 20 min | `telco_churn_new_training_data`, `churn_inference_log` |
| 03 | `03_monitoring_setup` | 15 min | Lakehouse Monitor, `_profile_metrics` + `_drift_metrics` tables (async refresh) |
| 04 | `04_drift_alerts` | 12 min | DBSQL alert on PSI threshold |
| 05 | `05_trigger_retrain` | 10 min | Retrain job run (async, 10-15 min pipeline) |
| 06 | `06_ab_canary_setup` | 15 min | — (reads live endpoint state; UI-driven traffic split) |
| 07 | `07_incident_runbook` | 22 min *(bonus)* | Rolled-back endpoint config, retrain job run |
| 08 | `08_client_deployment` | 18 min *(discussion)* | — |
|    | Total | 125 min | |

**What participants leave with:**
- How to automate the full retraining lifecycle with Databricks Jobs: task dependencies, deployment gates that block underperforming models
- How to manage risk during model updates using canary deployments and UI-driven traffic splits
- How Lakehouse Monitoring detects data drift using PSI, and how to wire it to automated DBSQL alerts
- How to run an end-to-end incident response: detect → diagnose → rollback → retrain using Databricks-native tooling

### Session 2 error-prone areas

| Risk | Where | Mitigation |
|---|---|---|
| `06_ab_canary_setup` reads stale endpoint state | Retrain job from `05` takes 10-15 min; `06` may open before it finishes | Add note: if only one model version is visible, wait 1-2 min and re-run the inspect cells |
| `04_drift_alerts` runs before monitor refresh completes | Monitor refresh from `03` takes 5-10 min | Notebook warns participants; they can proceed and re-run the query cell once refresh is done |
| Bundle deploy fails if DABs in Workspace not enabled | `01_cicd_overview` | Verify the feature is enabled in the workspace before the workshop |
| Alert subscription UI path changes between DBR versions | `04_drift_alerts` — legacy SQL alerts UI | Test the subscription flow in the target workspace before the session |

---

## Repository Structure

```
participant/
├── session1/
│   ├── 00_setup_participant.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_messy_notebook.ipynb         # instructor demo — do not run
│   ├── 03_refactored_pipeline.ipynb
│   ├── 04_mlflow_experiment.ipynb
│   ├── 05_register_and_serve.ipynb
│   ├── 06_llm_explanation.ipynb
│   └── 07_production_checklist.ipynb   # also opens Session 2 as a bridge
├── session2/
│   ├── 01_cicd_overview.ipynb
│   ├── 02_simulate_drift.ipynb
│   ├── 03_monitoring_setup.ipynb
│   ├── 04_drift_alerts.ipynb
│   ├── 05_trigger_retrain.ipynb
│   ├── 06_ab_canary_setup.ipynb
│   ├── 07_incident_runbook.ipynb       # bonus — run if time allows
│   └── 08_client_deployment.ipynb      # discussion / leave-behind
├── utils/
│   ├── config.ipynb                    # sets catalog, schema, safe_username, genie_space_id
│   └── resources/                      # diagrams embedded in notebooks (PNG)
└── bundle/
    ├── databricks.yml                  # DAB root — schema auto-derived from current_user()
    ├── resources/retrain_job.yml       # 5-task retrain pipeline definition
    ├── wheels/                         # churn_model package source + tests
    ├── jobs/                           # job task notebooks (00–04)
    └── .github/workflows/             # GitHub Actions CI/CD definitions
instructor/
├── scripts/00_instructor_setup.ipynb   # one-time workspace setup (workspace admin required)
└── artifacts/                          # Telco-Customer-Churn.csv + pre-built wheel
```

---

## Instructor Setup

Run `instructor/scripts/00_instructor_setup.ipynb` (requires workspace admin) before the workshop. The notebook is broken into numbered steps — run them in order:

| Step | What it does |
|---|---|
| 1–2 | Set catalog, create `00_shared` schema |
| 3–3b | Create UC Volumes, copy CSV and wheel |
| 4 | Load `telco_churn` Delta table (7,043 rows) |
| 5 | Create `telco_churn_holdout` (stratified 15% sample) |
| 6 | Grant permissions to `account users` |
| 7 | Create `foundation_model_with_gateway` serving endpoint with AI Gateway guardrails |
| 8 | Register `churn_explainer` prompt `@production` in shared Prompt Registry |
| 9 | Create shared Genie Space on `telco_churn`; prints `space_id` |

After Step 9: paste the printed `space_id` into `participant/utils/config.ipynb` to replace the `genie_space_id` placeholder.

---

## Participant Permissions Required

| Permission | Resource | Purpose |
|---|---|---|
| `USE CATALOG` | `workshop` | Access the catalog |
| `BROWSE` | `workshop` | See catalog in Explorer |
| `CREATE SCHEMA` | `workshop` | Participants create `workshop.<username>` |
| `USE SCHEMA` | `workshop.00_shared` | Access shared data |
| `SELECT` | `workshop.00_shared.telco_churn` | Read training data |
| `SELECT` | `workshop.00_shared.telco_churn_holdout` | Read holdout data for deployment gate |
| `READ VOLUME` | `workshop.00_shared.raw_files` | Access raw CSV |
| `READ VOLUME` | `workshop.00_shared.wheels` | Install `churn_model` wheel |
| `CAN_QUERY` | `foundation_model_with_gateway` endpoint | Call LLM from `06_llm_explanation` |

All granted via `GRANT ... TO 'account users'` in the instructor setup notebook.

---

## Dataset

**Source:** IBM Telco Customer Churn — [IBM GitHub](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

- 7,043 customer records, 21 features + 1 binary target (Churn: Yes/No)
- ~26% churn rate — moderately imbalanced
- Features: demographics (gender, SeniorCitizen, Partner, Dependents), account (tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges), services (PhoneService, MultipleLines, InternetService, and 6 add-on services)
- Known quirks handled in the setup notebook:
  - `SeniorCitizen` is stored as `0`/`1` int; all other binary features are `"Yes"`/`"No"` strings
  - 11 rows have an empty string for `TotalCharges` (new customers with `tenure = 0`)

---

## Data Architecture

```
╔══ UNITY CATALOG: workshop.00_shared (shared · read-only) ══════════════════════════╗
║  telco_churn  (7,043 rows)              telco_churn_holdout  (~1,056 rows)          ║
║  training source for all sessions       deployment gate evaluation only             ║
╚══════════════════════╤═════════════════════════════════════╤══════════════════════╝
                       │ spark.table()                        │ (gate only)
╔══ SESSION 1 ═════════╪══════════════════════════════════════╪══════════════════════╗
║                      ▼                                      │                      ║
║  churn_model  (Python package · installed from UC Volume)   │                      ║
║  features.py · train.py · evaluate.py                       │                      ║
║         │                                                   │                      ║
║         │ fe.create_training_set() + run_all_models()       │                      ║
║         ▼                                                   │                      ║
║  Feature Store: workshop.<schema>.churn_features            │                      ║
║  MLflow Experiment: /Users/<user>/churn_<schema>            │                      ║
║    LogisticRegression · RandomForest · GradientBoostedTrees │                      ║
║         │                                                   │                      ║
║         │ mlflow.register_model()                           │                      ║
║         ▼                                                   ▼                      ║
║  UC Model Registry: workshop.<schema>.churn_classifier                             ║
║    @champion  →  pyfunc (fe.score_batch)                                           ║
║    @serving   →  sklearn pipeline (endpoint)                                       ║
║         │                                                                          ║
║         │ w.serving_endpoints.create()  [async — ~5-10 min spin-up]               ║
║         ▼                                                                          ║
║  Model Serving + AI Gateway: churn_<schema>                                        ║
║    rate limit: 60 req/min/user · inference logging enabled                         ║
║         │                                                                          ║
║         ├──► REST prediction                                                       ║
║         └──► Foundation Model API (Llama 3.3 70B) → churn explanation             ║
║                    Prompt: workshop.00_shared.churn_explainer@production           ║
╚════════════════════════════════════════════════════════════════════════════════════╝

╔══ SESSION 2 ═══════════════════════════════════════════════════════════════════════╗
║                                                                                    ║
║  DAB: bundle/databricks.yml  →  workshop_retrain_churn (Lakeflow Job)             ║
║    feature_engineering → ingest_and_validate → train_models                       ║
║                       → gate_and_register (F1 gate ±5%)                           ║
║                       → update_endpoint (90/10 canary)                            ║
║    schema auto-derived from current_user() — no manual variable override needed   ║
║                                                                                    ║
║  02_simulate_drift:                                                                ║
║    telco_churn_new_training_data  (tenure shift · price increase · label flip)     ║
║    churn_inference_log            (simulated predictions with ground truth)        ║
║         │                                                                          ║
║         ▼                                                                          ║
║  Lakehouse Monitor on churn_inference_log                                          ║
║    _profile_metrics · _drift_metrics  (PSI per feature · async refresh)           ║
║         │                                                                          ║
║         │ PSI > 0.2                                                                ║
║         ▼                                                                          ║
║  DBSQL Alert → notification → Incident Runbook                                     ║
║    detect → diagnose → rollback @champion → re-trigger retrain                    ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```
