import argparse
import sys
from pathlib import Path

from typing import Dict, Any, List
from src.utils import load_model_artifacts

def load_and_validate_model(model_path_str: str) -> Dict[str, Any]:
    """Loads and validates the serialized model artifacts.
    
    Args:
        model_path_str: Path to the serialized .joblib file.
        
    Returns:
        Dictionary of loaded model artifacts.
    """
    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path.resolve()}", file=sys.stderr)
        print("Please train the model first by running: python3 src/train.py", file=sys.stderr)
        sys.exit(1)
        
    try:
        artifacts = load_model_artifacts(model_path)
    except Exception as e:
        print(f"Error loading model artifacts: {e}", file=sys.stderr)
        sys.exit(1)
        
    required_keys = ["preprocessor", "classifier", "feature_names"]
    for key in required_keys:
        if key not in artifacts:
            print(f"Error: Missing required component '{key}' in model artifacts.", file=sys.stderr)
            sys.exit(1)
            
    return artifacts

import json
import pandas as pd
from src.explainer import SHAPExplainer

def predict_single(features: List[float], artifacts: Dict[str, Any]) -> None:
    """Audits a single transaction and prints SHAP explainability results in JSON format.
    
    Args:
        features: List of 30 numerical transaction values.
        artifacts: Dictionary containing preprocessor and model components.
    """
    preprocessor = artifacts["preprocessor"]
    classifier = artifacts["classifier"]
    feature_names = artifacts["feature_names"]
    
    if len(features) != 30:
        print(f"Error: Expected exactly 30 feature values (Time, V1-V28, Amount). Received {len(features)}.", file=sys.stderr)
        sys.exit(1)
        
    # Construct DataFrame with proper headers
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df_input = pd.DataFrame([features], columns=columns)
    
    # Preprocess feature values
    X, _ = preprocessor.transform(df_input)
    
    # Run predictions
    prob = float(classifier.predict_proba(X)[0, 1])
    is_fraud = prob >= 0.50
    
    # Assign Risk Tier
    if prob >= 0.70:
        risk_tier = "High"
    elif prob >= 0.20:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
        
    # Compute SHAP explanation
    explainer = SHAPExplainer(classifier)
    explanation = explainer.explain_instance(X[0], feature_names)
    
    # Format features contribution, sorted by absolute impact descending
    features_attribution = []
    for f in explanation.features:
        features_attribution.append({
            "feature": f.feature_name,
            "value": float(f.value),
            "contribution": float(f.contribution)
        })
    features_attribution.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    
    # Package output json
    result = {
        "transaction_summary": {
            "fraud_probability": prob,
            "classification_decision": "Fraud" if is_fraud else "Legitimate",
            "risk_tier": risk_tier,
            "base_value": float(explanation.base_value)
        },
        "explanations": features_attribution
    }
    
    print(json.dumps(result, indent=4))

def main() -> None:
    """Entry point for the FraudLens CLI."""
    parser = argparse.ArgumentParser(
        description="FraudLens Command Line Interface for auditing financial transactions."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/fraud_model.joblib",
        help="Path to the serialized model artifacts (.joblib)."
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Modes of operation")
    
    # Subparser for single-instance prediction
    single_parser = subparsers.add_parser("single", help="Audit a single transaction.")
    single_parser.add_argument(
        "--features", 
        type=float, 
        nargs="+", 
        required=True,
        help="List of transaction features in order: Time, V1-V28, Amount (30 values)."
    )
    
    # Subparser for batch-instance prediction
    batch_parser = subparsers.add_parser("batch", help="Batch process transactions from CSV.")
    batch_parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to the input CSV file containing transactions."
    )
    batch_parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path to save the resulting prediction CSV file."
    )
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
        
    # Load and validate artifacts
    artifacts = load_and_validate_model(args.model)
    
    if args.mode == "single":
        predict_single(args.features, artifacts)

if __name__ == "__main__":
    main()
