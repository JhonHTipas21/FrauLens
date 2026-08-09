import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.utils import load_model_artifacts
from src.explainer import SHAPExplainer


def load_and_validate_model(model_path_str: str) -> Dict[str, Any]:
    """Loads and validates the serialized model artifacts.

    Args:
        model_path_str: Path to the serialized .joblib file.

    Returns:
        Dictionary of loaded model artifacts.

    Raises:
        SystemExit: If the file is not found or a required key is missing.
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


def _assign_risk_tier(prob: float) -> str:
    """Maps a fraud probability to a named risk tier.

    Args:
        prob: Fraud probability between 0.0 and 1.0.

    Returns:
        Risk tier string: 'High', 'Medium', or 'Low'.
    """
    if prob >= 0.70:
        return "High"
    elif prob >= 0.20:
        return "Medium"
    return "Low"


def predict_single(features: List[float], artifacts: Dict[str, Any]) -> None:
    """Audits a single transaction and prints SHAP explainability results in JSON format.

    Args:
        features: List of 30 numerical transaction values (Time, V1-V28, Amount).
        artifacts: Dictionary containing preprocessor and model components.

    Raises:
        SystemExit: If features list does not contain exactly 30 values.
    """
    preprocessor = artifacts["preprocessor"]
    classifier = artifacts["classifier"]
    feature_names = artifacts["feature_names"]

    if len(features) != 30:
        print(
            f"Error: Expected exactly 30 feature values (Time, V1-V28, Amount). "
            f"Received {len(features)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Construct DataFrame with proper column headers
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df_input = pd.DataFrame([features], columns=columns)

    # Scale and transform feature values
    X, _ = preprocessor.transform(df_input)

    # Run predictions
    prob = float(classifier.predict_proba(X)[0, 1])
    is_fraud = prob >= 0.50
    risk_tier = _assign_risk_tier(prob)

    # Compute SHAP explanation for interpretability
    explainer = SHAPExplainer(classifier)
    explanation = explainer.explain_instance(X[0], feature_names)

    sorted_features = sorted(
        explanation.features,
        key=lambda f: abs(f.contribution),
        reverse=True,
    )
    features_attribution = [
        {
            "feature": f.feature_name,
            "value": float(f.value),
            "contribution": float(f.contribution),
        }
        for f in sorted_features
    ]

    result = {
        "transaction_summary": {
            "fraud_probability": prob,
            "classification_decision": "Fraud" if is_fraud else "Legitimate",
            "risk_tier": risk_tier,
            "base_value": float(explanation.base_value),
        },
        "explanations": features_attribution,
    }

    print(json.dumps(result, indent=4))


def predict_batch(input_path: str, output_path: str, artifacts: Dict[str, Any]) -> None:
    """Batch-processes a CSV file of transactions and saves an annotated output CSV.

    For each transaction the function appends three columns:
      - fraud_probability: the predicted positive class probability.
      - classification_decision: 'Fraud' or 'Legitimate'.
      - risk_tier: 'High', 'Medium', or 'Low'.

    Args:
        input_path: Path to the input CSV file containing raw transactions.
        output_path: Path where the annotated results CSV will be written.
        artifacts: Dictionary containing preprocessor and model components.

    Raises:
        SystemExit: If the input file does not exist or cannot be parsed.
    """
    preprocessor = artifacts["preprocessor"]
    classifier = artifacts["classifier"]

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file.resolve()}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} transactions from {input_path}. Running predictions...")

    # Preprocess — drop Class column if present so we score unknown transactions
    df_features = df.drop(columns=["Class"], errors="ignore")
    X, _ = preprocessor.transform(df_features)

    probs = classifier.predict_proba(X)[:, 1]

    df["fraud_probability"] = probs
    df["classification_decision"] = ["Fraud" if p >= 0.50 else "Legitimate" for p in probs]
    df["risk_tier"] = [_assign_risk_tier(float(p)) for p in probs]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    n_fraud = (df["classification_decision"] == "Fraud").sum()
    print(f"Batch audit complete. {n_fraud}/{len(df)} transactions flagged as Fraud.")
    print(f"Annotated results saved to: {output_file.resolve()}")


def main() -> None:
    """Entry point for the FraudLens CLI."""
    parser = argparse.ArgumentParser(
        description="FraudLens Command Line Interface for auditing financial transactions."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/fraud_model.joblib",
        help="Path to the serialized model artifacts (.joblib).",
    )

    subparsers = parser.add_subparsers(dest="mode", help="Modes of operation")

    # Subparser: single-instance audit
    single_parser = subparsers.add_parser("single", help="Audit a single transaction.")
    single_parser.add_argument(
        "--features",
        type=float,
        nargs="+",
        required=True,
        help="List of transaction features in order: Time, V1-V28, Amount (30 values).",
    )

    # Subparser: batch audit from CSV
    batch_parser = subparsers.add_parser("batch", help="Batch process transactions from CSV.")
    batch_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing transactions.",
    )
    batch_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the annotated results CSV file.",
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Load and validate serialized artifacts
    artifacts = load_and_validate_model(args.model)

    if args.mode == "single":
        predict_single(args.features, artifacts)
    elif args.mode == "batch":
        predict_batch(args.input, args.output, artifacts)


if __name__ == "__main__":
    main()
