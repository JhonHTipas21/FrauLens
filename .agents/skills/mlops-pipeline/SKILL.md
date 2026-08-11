---
name: fraudlens-mlops-pipeline
description: Retraining, validating, and releasing the FraudLens hybrid model via Git and GitHub Actions
---

# FraudLens MLOps Pipeline Skill

This skill provides guidelines and operational procedures for retraining, validating, and releasing the FraudLens model.

## Model Training & Dataset Handling
- Core Pipeline: Ingested via `src/data_loader.py` and trained via `src/train.py`.
- Dataset Path: Controlled via the `FRAUDLENS_DATA_PATH` environment variable.
  - If `FRAUDLENS_DATA_PATH` is set, the pipeline uses that file (e.g., in CI/CD, pointing to the 2,000-row fixture `tests/fixtures/creditcard_sample.csv`).
  - If unset, the pipeline downloads the full 284k Kaggle Credit Card dataset (~150MB uncompressed) to `data/creditcard.csv`.
- Training Output: Saves the serialised artifacts to `models/fraud_model.joblib` and training parameters/curve metrics to `models/metadata.json`.

## Temporal Splitting & Leakage Prevention
- Evaluation must **always** use a temporal split (`src.utils.temporal_split`) rather than a random split.
- The first 80% of chronological transactions (sorted by `Time` ascending) are used for training, and the last 20% for testing.
- Feature scaling must be fit strictly on the train set and applied to the test set to prevent future-data leakage.

## Model Validation & Threshold Enforcement
- Executed via `scripts/ci_validate.py`.
- Calculates four core metrics at a default threshold of 0.5:
  - `Precision`
  - `Recall`
  - `F1 Score`
  - `AUC-PR`
- Validation fails (exit code 1) if any metric drops below the thresholds specified in the environment:
  - `MIN_AUC_PR` (production target: 0.70)
  - `MIN_F1` (production target: 0.65)
  - `MIN_RECALL` (production target: 0.60)
  *Note: For CI runs using the 2,000-row fixture, these thresholds are set to 0.00 to prevent false build failures due to small sample size limitations.*

## Step Summary Comparison
- The validation script compares the candidate model's metrics with `models/current_metrics.json`.
- It writes a markdown comparison table directly to `$GITHUB_STEP_SUMMARY` for developer review.

## Lineage Model Card
- Generates a versioned markdown file `models/model_card_{version}.md` detailing:
  - UTC Date, Commit SHA, and dataset MD5 hash.
  - Kaggle dataset source & ODbL license.
  - Feature names and XGBoost hyperparameters.
  - Split method, threshold, metrics, and confusion matrix.
  - Clear risk notes regarding data leakage, concept drift, and a **mandatory disclaimer** that this is an experimental project and not validated for real financial transactions.

## GitHub Action Workflow Actions
- Action file: `.github/workflows/model-validation.yml`.
- Triggers: manual `workflow_dispatch` or `pull_request` on core ML source changes.
- Permissions: Must include `contents: write` to allow the automated generation of GitHub Releases and tag creations upon manual triggers.
