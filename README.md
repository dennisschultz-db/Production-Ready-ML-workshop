# Production-Ready ML Systems Workshop


|               |                                                                                               |
| --------------- | :---------------------------------------------------------------------------------------------- |
| **Format:**   | Two 2-hour instructor-led sessions with hands-on exercises                                    |
| **Dataset:**  | IBM Telco Customer Churn — binary classification, ~7K rows                                   |
| **Platform:** | Databricks (serverless notebooks, Unity Catalog, MLflow, Model Serving, Lakehouse Monitoring) |

---

## Scenario

Participants will take a customer churn prediction model from a messy notebook all the way through to a fully monitored production deployment. They will refactor a prototype into a testable Python package, run MLflow experiments to compare Logistic Regression, Random Forest, and Gradient Boosted Trees, register the best model in Unity Catalog with a `@champion` alias, deploy it as a rate-limited REST endpoint via AI Gateway, and query it live — including a Foundation Model API call that generates a human-readable explanation of the prediction. In Session 2, the focus shifts to automation and resilience: participants trigger an automated retraining pipeline, observe a deployment gate block a underperforming model, configure a canary traffic split, inject realistic data drift, create a Lakehouse Monitor to detect it, set up an alert, and then walk through an end-to-end incident runbook — diagnosing the drift, rolling back to the previous champion, and re-triggering retraining.

## Workshop Outline

### Session 1 — Building Production-Grade ML (~2 hrs)


| Step              | What they do                                                               | What they learn                                                  |
| ------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| EDA               | Explore Telco dataset                                                      | Class imbalance, null handling, feature distributions            |
| Code quality      | Compare messy notebook vs refactored`churn_model` package                  | Why notebooks fail at scale; modular ML code                     |
| Train models      | Run LR, RF, and GBT experiments                                            | MLflow autolog: params, metrics, artifacts                       |
| Compare & select  | Navigate MLflow experiment UI                                              | Pick best model by F1; understand run lineage                    |
| Register & deploy | Register to Unity Catalog, assign`@champion` alias, deploy endpoint        | Model Registry, aliases, Model Serving, AI Gateway rate limiting |
| Serve & explain   | Query endpoint via REST, call Foundation Model API for a churn explanation | Live prediction, LLM augmentation                                |
| Reality check     | Run pytest + data quality checks, review scorecard                         | 5 things done, 5 still missing → sets up Session 2              |

**What participants leave with:**

* How to structure ML code as a testable, reusable Python package instead of a monolithic notebook — and why it matters when models go to production
* How to use MLflow to track experiments across model types, compare runs objectively, and maintain a full lineage from raw data to registered model
* How Unity Catalog Model Registry works: versioning, aliases like `@champion`, and the governance layer that separates model development from deployment
* How to expose a model as a governed REST endpoint using Model Serving and AI Gateway — including rate limiting, inference logging, and augmenting predictions with an LLM explanation

### Session 2 — MLOps in Production (~2 hrs)


| Step         | What they do                                                    | What they learn                                                                   |
| -------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| CI/CD        | Review GitHub Actions + DABs job graph                          | How retraining pipelines are structured                                           |
| Retrain      | Trigger retraining job via SDK                                  | Automated training pipelines, task graphs                                         |
| Gate         | Observe deployment gate pass/fail scenarios                     | Quality thresholds block bad model promotions                                     |
| A/B & canary | Configure 90/10 then 50/50 traffic split                        | Safe rollouts without downtime                                                    |
| Inject drift | Run drift simulation (tenure shift, price increase, label flip) | How real-world distribution shift looks                                           |
| Monitor      | Create Lakehouse Monitor, trigger refresh                       | Population Stability Index (PSI) drift detection, profile vs drift metrics tables |
| Alert        | Create a DBSQL alert on drift metrics                           | Automated notification when drift exceeds threshold                               |
| Incident     | Detect drift → diagnose → rollback → re-trigger retrain      | End-to-end incident response runbook                                              |

**What participants leave with:**

* How to automate the full retraining lifecycle with Databricks Jobs: scheduled runs, task dependencies, deployment gates that block underperforming models from reaching production
* How to manage risk during model updates using canary deployments and A/B traffic splits — rolling forward safely and rolling back when something goes wrong
* How Lakehouse Monitoring detects data drift using PSI, and how to wire it to automated DBSQL alerts so degradation triggers a response before users notice
* How to run an end-to-end incident response: detect → diagnose → rollback → retrain — using only Databricks-native tooling throughout

---

## Repository Structure

```
databricks.yml                        # Databricks Asset Bundle root
resources/                            # DABs resource definitions
├── training_job.yml
├── retrain_job.yml
└── monitoring_refresh_job.yml
participant/
├── session1/                         # Session 1 notebooks (00–06)
├── session2/                         # Session 2 notebooks (01–08)
├── common/config.yml                 # Shared config: catalog, features, hyperparams
├── utils/config.ipynb                # Config loader
└── asset_bundle/
    ├── databricks.yml
    ├── wheels/                       # churn_model package + tests
    └── .github/workflows/            # CI/CD pipeline definitions
instructor/
├── scripts/00_instructor_setup.ipynb # One-time workspace setup (instructor only)
└── artifacts/                        # Source data and pre-built wheel
```

## Instructor setup

1. Ensure the `platform-workshop` catalog exists in the target workspace
2. Create Git Folder at root of SHARED workspace folder.
3. Run `instructor/scripts/00_instructor_setup.ipynb` (requires WORKSPACE ADMIN)
   - Creates shared schema and Volumes
   - Load the Telco dataset
   - Build and upload the `churn_model` wheel
   - Grant permissions to `account users`
4. Copy `SETUP-Production-ready-ML-workshop` to `/Workspace/shared/` in the workspace

## Participant permissions required


| Permission      | Resource                                 | Purpose                                                |
| ----------------- | ------------------------------------------ | -------------------------------------------------------- |
| `USE CATALOG`   | `workshop`                               | Access the catalog                                     |
| `BROWSE`        | `workshop`                               | See catalog in Explorer                                |
| `CREATE SCHEMA` | `workshop`                               | Participants create their own workshop.username schema |
| `USE SCHEMA`    | `workshop.00_shared`                     | Access shared data                                     |
| `SELECT`        | `workshop.00_shared.telco_churn`         | Read training data                                     |
| `SELECT`        | `workshop.00_shared.telco_churn_holdout` | Read holdout data for evaluation                       |
| `READ VOLUME`   | `workshop.00_shared.raw_files`           | Access raw CSV                                         |

All granted via `GRANT ... TO 'account users'` in the instructor setup notebook.

### Participant setup (run at the start of the workshop)

Each participant opens `SETUP-Production-ready-ML-workshop.ipynb` from `/Workspace/shared` and runs all cells. This creates their personal schema and workspace artifacts.

---

## Customer Churn Corpus

Source: IBM's GitHub repo for their Cloud Pak for Data tutorials: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The IBM Telco Customer Churn dataset is a synthetic but realistic dataset originally published by IBM as a Watson Analytics sample. It's one of the most widely used demo datasets in the ML community — you'll find it in Databricks tutorials, Kaggle notebooks, scikit-learn examples, and countless blog posts.

- 7,043 customer records, 21 features + 1 binary target (Churn: Yes/No)
- ~26% churn rate — moderately imbalanced
- Features cover three areas:
  - Demographics: gender, SeniorCitizen, Partner, Dependents
  - Account: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges
  - Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Known quirks (both of which we handle in the admin setup notebook):
  - SeniorCitizen is 0/1 int while every other binary feature is "Yes"/"No" string
  - 11 rows have an empty string for TotalCharges (customers with tenure = 0 who haven't been billed yet)

---

## Data Architecture

╔══ UNITY CATALOG: workshop.00_shared (shared · read-only for all participants) ════════╗
║  ┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐  ║
║  │  telco_churn                         │   │  telco_churn_holdout                 │  ║
║  │  7,043 rows · 21 features            │   │  ~1,056 rows · stratified 15%        │  ║
║  │  Churn: Yes/No  (26% positive)       │   │  not used in any training exercise   │  ║
║  └──────────────────┬───────────────────┘   └──────────────────┬───────────────────┘  ║
╚═════════════════════╪══════════════════════════════════════════╪══════════════════════╝
│ spark.table()                            │ (deployment gate only)
╔══ SESSION 1: BUILD ═╪══════════════════════════════════════════╪═══════════════════════╗
║                     ▼                                        ...                       ║
║  ┌──────────────────────────────────────────────────────────────────────────────────┐  ║
║  │  churn_model  (Python package · installed from UC Volume)                        │  ║
║  │  features.py  ·  train.py  ·  evaluate.py  ·  data_quality.py                    │  ║
║  └──────────────────────────────────────┬──────────────────────────────────────────-┘  ║
║                                         │ run_training(data=df)                        ║
║                                         ▼                                              ║
║  ┌──────────────────────────────────────────────────────────────────────────────────┐  ║
║  │  MLflow Experiment   workshop.<schema>.churn_experiment                          │  ║
║  │                                                                                  │  ║
║  │  ┌──────────────────┐   ┌───────────────────────┐   ┌──────────────────────┐     │  ║
║  │  │  LogisticReg     │   │  RandomForest         │   │ GradientBoostedTrees │     │  ║
║  │  │  F1 ≈ 0.55       │   │  F1 ≈ 0.62            │   │ F1 ≈ 0.65  ← best    │     │  ║
║  │  └──────────────────┘   └───────────────────────┘   └──────────────────────┘     │  ║
║  │  logs: params · metrics · confusion matrix · feature importance · model artifact │  ║
║  └──────────────────────────────────────┬──────────────────────────────────────────-┘  ║
║                                         │ mlflow.register_model()                      ║
║                                         ▼                                              ║
║  ┌──────────────────────────────────────────────────────────────────────────────────┐  ║
║  │  UC Model Registry   workshop.<schema>.churn_classifier                          │  ║
║  │  v1 · v2 · v3 · ...        @champion ──► points to best version                  │  ║
║  └──────────────────────────────────────┬──────────────────────────────────────────-┘  ║
║                                         │ w.serving_endpoints.create_and_wait()        ║
║                                         ▼                                              ║
║  ┌──────────────────────────────────────────────────────────────────────────────────┐  ║
║  │  Model Serving + AI Gateway   churn-<schema>                                     │  ║
║  │  rate limit: 60 req / min / user  ·  inference logging enabled                   │  ║
║  │                                                                                  │  ║
║  │   POST /serving-endpoints/churn-<schema>/invocations  ───────────────────────►   │  ║
║  │                         │                                                        │  ║
║  │              ┌──────────┴────────────────────────────────────┐                   │  ║
║  │              ▼                                               ▼                   │  ║
║  │   churn_payload                              Foundation Model API                │  ║
║  │   AI Gateway inference log table             Llama 3.3 70B Instruct              │  ║
║  │   request · response · model_id              → human-readable churn explanation  │  ║
║  └──────────────────────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════════════════════════════╝

╔══ SESSION 2: OPERATE ══════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐    ║
║  │  Databricks Jobs (DABs)   workshop_retrain_churn                               │    ║
║  │                                                                                │    ║
║  │   GitHub push / schedule                                                       │    ║
║  │          │                                                                     │    ║
║  │          ▼                                                                     │    ║
║  │   [train_models] ──► [deployment_gate] ──────────────► [update_champion]       │    ║
║  │                              │              pass             │                 │    ║
║  │                         F1 < threshold?              @champion alias           │    ║
║  │                         block · notify               updated in UC             │    ║
║  └────────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                        ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐    ║
║  │  Traffic Control   churn-<schema>                                              │    ║
║  │                                                                                │    ║
║  │  canary:   10% ──► @challenger        90% ──► @champion                        │    ║
║  │  A/B test: 50% ──► @challenger        50% ──► @champion                        │    ║
║  └────────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                        ║
║  05_simulate_drift.ipynb  (tenure shift · MonthlyCharges +$10 · label flip)            ║
║         │                                                                              ║
║         ▼                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐    ║
║  │  churn_inference_log        (workshop.<schema>)                                │    ║
║  │  feature columns · prediction · model_id · inference_ts · Churn (ground truth) │    ║
║  └─────────────────────────────────────┬──────────────────────────────────────────┘    ║
║                                        │                                               ║
║   churn_monitor_baseline  (view) ──────┤  compare feature distributions                ║
║   telco_churn with typed columns       │                                               ║
║                                        ▼                                               ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐    ║
║  │  Lakehouse Monitor   on churn_inference_log                                    │    ║
║  │  InferenceLog · daily granularity · PSI per feature                            │    ║
║  │                                                                                │    ║
║  │  ┌─────────────────────────────────┐  ┌──────────────────────────────────┐     │    ║
║  │  │  _profile_metrics               │  │  _drift_metrics                  │     │    ║
║  │  │  prediction distribution        │  │  PSI · tenure         ≈ 0.28     │     │    ║
║  │  │  per-feature statistics         │  │  PSI · MonthlyCharges ≈ 0.12     │     │    ║
║  │  └─────────────────────────────────┘  └──────────────────┬───────────────┘     │    ║
║  └──────────────────────────────────────────────────────────┼─────────────────────┘    ║
║                                                              │ PSI > 0.2 threshold     ║
║                                                              ▼                         ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐    ║
║  │  DBSQL Alert  →  notification  →  Incident Runbook                             │    ║
║  │  diagnose drift  →  rollback @champion  →  re-trigger retrain job              │    ║
║  └────────────────────────────────────────────────────────────────────────────────┘    ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
