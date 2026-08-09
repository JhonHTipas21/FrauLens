import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_loader import load_dataset
from src.anomaly_detector import IsolationForestDetector
from src.classifier import FraudClassifier
from src.utils import (
    temporal_split, 
    DataPreprocessor, 
    calculate_metrics, 
    save_model_artifacts
)

def train_pipeline() -> None:
    """Executes the complete end-to-end model training pipeline.
    
    This fits the preprocessor, fits the unsupervised anomaly detector, trains
    the supervised classifier, computes threshold-based performance profiles,
    and serializes the models and metadata to disk.
    """
    print("--- Starting FraudLens Training Pipeline ---")
    
    # 1. Load data
    df = load_dataset()
    
    # 2. Temporal split
    train_df, test_df = temporal_split(df, time_col='Time', test_ratio=0.2)
    
    # 3. Preprocess
    print("Preprocessing transaction data...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df, target_col='Class')
    
    X_train, y_train = preprocessor.transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    assert y_train is not None
    assert y_test is not None
    
    # Save feature names for explanation mapping later
    feature_names = preprocessor.feature_cols
    
    # 4. Initialize Isolation Forest detector (Phase 1)
    contamination = float(y_train.mean())
    print(f"Initializing Anomaly Detector (contamination rate={contamination:.6f})...")
    detector = IsolationForestDetector(contamination=contamination, random_state=42)
    
    # 5. Initialize and Fit Hybrid Fraud Classifier (Phase 2)
    print("Training Hybrid Classifier (XGBoost + Anomaly Score)...")
    # Calculate scale_pos_weight for XGBoost to handle extreme imbalance
    num_neg = len(y_train) - sum(y_train)
    num_pos = sum(y_train)
    scale_pos_weight = float(num_neg / num_pos)
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=150,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    hybrid_classifier = FraudClassifier(anomaly_detector=detector, classifier_model=xgb_model)
    hybrid_classifier.fit(X_train, y_train)
    
    # 6. Evaluate model on test set
    print("Evaluating model performance on temporal test set...")
    y_prob = hybrid_classifier.predict_proba(X_test)[:, 1]
    
    # Calculate metrics at standard threshold 0.5
    standard_metrics = calculate_metrics(y_test, y_prob, threshold=0.5)
    
    print("\n--- Model Performance at Threshold 0.5 ---")
    print(f"Precision: {standard_metrics['precision']:.4f}")
    print(f"Recall:    {standard_metrics['recall']:.4f}")
    print(f"F1-Score:  {standard_metrics['f1']:.4f}")
    print(f"AUC-PR:    {standard_metrics['auc_pr']:.4f}")
    print("Confusion Matrix:")
    print(f"  True Negatives (Legit): {standard_metrics['confusion_matrix']['tn']}")
    print(f"  False Positives (Falsely flagged): {standard_metrics['confusion_matrix']['fp']}")
    print(f"  False Negatives (Missed Fraud):    {standard_metrics['confusion_matrix']['fn']}")
    print(f"  True Positives (Detected Fraud):   {standard_metrics['confusion_matrix']['tp']}")
    
    # 7. Generate Precision-Recall trade-off curve data for the dashboard
    print("Generating decision threshold trade-off curve...")
    threshold_curve = []
    # Test thresholds from 0.01 to 0.99
    for th in np.linspace(0.01, 0.99, 99):
        m = calculate_metrics(y_test, y_prob, threshold=float(th))
        threshold_curve.append({
            "threshold": float(th),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "fp": int(m["confusion_matrix"]["fp"]),
            "fn": int(m["confusion_matrix"]["fn"]),
            "tp": int(m["confusion_matrix"]["tp"])
        })
        
    # Find optimal threshold by F1-score
    best_th_idx = np.argmax([tc["f1"] for tc in threshold_curve])
    best_th_info = threshold_curve[best_th_idx]
    print(f"Optimal threshold based on F1-score: {best_th_info['threshold']:.2f} (F1: {best_th_info['f1']:.4f}, Precision: {best_th_info['precision']:.4f}, Recall: {best_th_info['recall']:.4f})")
    
    # 8. Serialize and Save Model & Metadata
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    artifacts = {
        "preprocessor": preprocessor,
        "classifier": hybrid_classifier,
        "feature_names": feature_names,
    }
    
    save_model_artifacts(artifacts, models_dir / "fraud_model.joblib")
    
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "standard_metrics": standard_metrics,
        "best_threshold_f1": best_th_info,
        "threshold_curve": threshold_curve
    }
    
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Saved training metadata to {models_dir / 'metadata.json'}")
    print("Training pipeline finished successfully.")

if __name__ == "__main__":
    train_pipeline()
