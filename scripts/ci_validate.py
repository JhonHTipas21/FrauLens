import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_loader import load_dataset
from src.utils import temporal_split, load_model_artifacts, calculate_metrics
from src.explainer import SHAPExplainer

def get_dataset_hash(df: pd.DataFrame) -> str:
    """Calculates an MD5 hash representing the dataset's state."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def format_markdown_table(current: dict, new: dict) -> str:
    """Formats the metrics comparison table for the GitHub Step Summary."""
    return f"""
| Metric | Current Model | Candidate Model | Status |
|--------|---------------|-----------|--------|
| Precision | {current.get('precision', 0):.4f} | {new.get('precision'):.4f} | {'✅' if new.get('precision') >= current.get('precision', 0) else '⚠️'} |
| Recall | {current.get('recall', 0):.4f} | {new.get('recall'):.4f} | {'✅' if new.get('recall') >= current.get('recall', 0) else '⚠️'} |
| F1 Score | {current.get('f1_score', 0):.4f} | {new.get('f1'):.4f} | {'✅' if new.get('f1') >= current.get('f1_score', 0) else '⚠️'} |
| AUC-PR | {current.get('auc_pr', 0):.4f} | {new.get('auc_pr'):.4f} | {'✅' if new.get('auc_pr') >= current.get('auc_pr', 0) else '⚠️'} |
"""

def main():
    print("--- Starting MLOps Model Validation ---")
    
    # 1. Load test data directly from the standard load logic
    print("Loading dataset for evaluation...")
    df = load_dataset()
    dataset_hash = get_dataset_hash(df)
    
    # Must use temporal split to avoid data leakage
    _, test_df = temporal_split(df, time_col='Time', test_ratio=0.2)
    
    # 2. Load candidate model artifacts
    model_path = project_root / "models" / "fraud_model.joblib"
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}. Run training first.")
        sys.exit(1)
    
    print("Loading candidate artifacts...")
    artifacts = load_model_artifacts(model_path)
    preprocessor = artifacts['preprocessor']
    classifier = artifacts['classifier']
    feature_names = artifacts['feature_names']
    
    # 3. Preprocess and evaluate
    print("Evaluating metrics on temporal test set...")
    X_test, y_test = preprocessor.transform(test_df)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_prob, threshold=0.5)
    
    new_metrics = {
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
        "auc_pr": metrics['auc_pr']
    }
    
    print("\nValidation Metrics Calculated:")
    for k, v in new_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # 4. Enforce strict MLOps thresholds from env variables
    min_auc_pr = float(os.environ.get('MIN_AUC_PR', 0.70))
    min_f1 = float(os.environ.get('MIN_F1', 0.65))
    min_recall = float(os.environ.get('MIN_RECALL', 0.60))
    
    failed_thresholds = []
    if new_metrics['auc_pr'] < min_auc_pr:
        failed_thresholds.append(f"AUC-PR {new_metrics['auc_pr']:.4f} is below minimum {min_auc_pr}")
    if new_metrics['f1'] < min_f1:
        failed_thresholds.append(f"F1 Score {new_metrics['f1']:.4f} is below minimum {min_f1}")
    if new_metrics['recall'] < min_recall:
        failed_thresholds.append(f"Recall {new_metrics['recall']:.4f} is below minimum {min_recall}")
        
    # 5. Compare with baseline (Current Model)
    current_metrics_path = project_root / "models" / "current_metrics.json"
    current_metrics = {}
    if current_metrics_path.exists():
        with open(current_metrics_path, 'r') as f:
            current_metrics = json.load(f).get("metrics", {})
            
    summary_table = format_markdown_table(current_metrics, new_metrics)
    
    # Write to GitHub Step Summary if running in CI
    step_summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if step_summary_path:
        with open(step_summary_path, 'a') as f:
            f.write("## FraudLens MLOps Validation Report\n\n")
            if failed_thresholds:
                f.write("### ❌ Validation Failed\n")
                f.write("The candidate model failed to meet the following criteria:\n")
                for err in failed_thresholds:
                    f.write(f"- **{err}**\n")
            else:
                f.write("### ✅ Validation Passed\n")
                f.write("The candidate model meets all performance thresholds.\n")
            f.write(summary_table)
            
    # 6. Extract Top 5 SHAP Features globally
    print("\nComputing global SHAP feature importance...")
    # Using a 200 sample to speed up CI runtime while keeping relative global importance accurate
    sample_size = min(200, len(X_test))
    np.random.seed(42)
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[indices]
    
    top_5_features = []
    try:
        explainer = SHAPExplainer(classifier)
        global_shap = explainer.explain_global(X_sample, feature_names)
        top_5_features = list(global_shap.items())[:5]
        print("SHAP global features computed successfully.")
    except Exception as e:
        print(f"SHAP explanation not supported or failed for this model architecture: {e}")
    
    # 7. Generate versioned Model Card
    gh_run_id = os.environ.get('GITHUB_RUN_ID')
    version = gh_run_id if gh_run_id else datetime.utcnow().strftime("%Y%m%d%H%M%S")
    commit_sha = os.environ.get('GITHUB_SHA', 'Unknown (local)')
    
    model_card_path = project_root / "models" / f"model_card_{version}.md"
    
    # Extract hyperparams safely if XGBoost
    hyperparams = {}
    if hasattr(classifier.classifier, 'get_params'):
        hyperparams = classifier.classifier.get_params()
    safe_params = {k: v for k, v in hyperparams.items() if isinstance(v, (int, float, str, bool))}
        
    model_card_content = f"""# Model Card - FraudLens (v{version})

> [!WARNING]
> **Not Validated for Financial Decisions**: This model is an experimental portfolio project. It is **NOT** validated, audited, or intended for use in real-world financial decision-making or production payment blocking systems.

## Model Details
- **Validation Date (UTC)**: {datetime.utcnow().isoformat()}
- **Commit SHA**: `{commit_sha}`
- **Version**: {version}
- **Architecture**: Hybrid (Isolation Forest unsupervised anomaly detection + XGBoost supervised classifier)

## Dataset
- **Origin**: Kaggle Credit Card Fraud Detection Dataset
- **License**: Open Data Commons Open Database License (ODbL)
- **Dataset Hash (MD5)**: `{dataset_hash}`
- **Split Method**: Temporal Split (Time column), 80/20 ratio. Never randomized to prevent data leakage.

## Performance Metrics (Temporal Test Split)
- **Threshold**: 0.5
- **Precision**: {new_metrics['precision']:.4f}
- **Recall**: {new_metrics['recall']:.4f}
- **F1 Score**: {new_metrics['f1']:.4f}
- **AUC-PR**: {new_metrics['auc_pr']:.4f}

### Confusion Matrix (Threshold 0.5)
- **True Negatives**: {metrics['confusion_matrix']['tn']}
- **False Positives**: {metrics['confusion_matrix']['fp']}
- **False Negatives**: {metrics['confusion_matrix']['fn']}
- **True Positives**: {metrics['confusion_matrix']['tp']}

## Features
- **Features Used**: {', '.join(feature_names)}

## Top 5 SHAP Global Features
"""
    if top_5_features:
        model_card_content += "The most influential features in predicting fraud across the validation sample:\n"
        for rank, (feat, val) in enumerate(top_5_features, 1):
            model_card_content += f"{rank}. **{feat}**: {val:.4f}\n"
    else:
        model_card_content += "SHAP explanations are not supported by the current underlying architecture.\n"

    model_card_content += "\n## Phase 2 Hyperparameters\n"
    model_card_content += "```json\n"
    model_card_content += json.dumps(safe_params, indent=2)
    model_card_content += "\n```\n"

    model_card_content += """
## Limitations & Risks
- **Data Leakage Risk**: Mitigated strictly via temporal splitting (past predicts future). However, feature scaling is applied post-split to ensure zero leakage.
- **Concept Drift**: Financial fraud evolves rapidly. This model trained on historical static data will degrade over time without continuous retraining.
- **Imbalance Handling**: Uses scale_pos_weight for XGBoost and anomaly scores. May over-flag normal transactions if contamination rate is miscalibrated.
"""

    with open(model_card_path, 'w') as f:
        f.write(model_card_content)
        
    print(f"\nModel card written to {model_card_path.name}")

    if failed_thresholds:
        print("\n❌ Pipeline Halt: Model validation failed due to threshold breaches.")
        sys.exit(1)
        
    print("\n✅ MLOps Pipeline checks passed successfully. Model ready for release.")

if __name__ == "__main__":
    main()
