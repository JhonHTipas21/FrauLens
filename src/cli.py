import argparse
import sys
from pathlib import Path

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
        
    print(f"Executing FraudLens CLI in mode: {args.mode}")

if __name__ == "__main__":
    main()
