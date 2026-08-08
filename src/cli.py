import argparse
import sys
from pathlib import Path

from typing import Dict, Any
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
    print(f"Loaded model artifacts. Mode: {args.mode}")

if __name__ == "__main__":
    main()
