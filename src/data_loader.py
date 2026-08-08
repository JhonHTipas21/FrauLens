import os
import zipfile
import requests
from pathlib import Path

DATA_URL = "https://github.com/jbrownlee/Datasets/raw/master/creditcard.csv.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "creditcard.csv.zip"
CSV_PATH = DATA_DIR / "creditcard.csv"

def download_dataset(url: str = DATA_URL, dest_path: Path = ZIP_PATH) -> None:
    """Downloads the credit card fraud dataset zip file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    with open(dest_path, "wb") as f:
        downloaded = 0
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded += len(data)
            if total_size:
                percent = (downloaded / total_size) * 100
                print(f"Downloaded: {downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB ({percent:.1f}%)", end="\r")
            else:
                print(f"Downloaded: {downloaded / (1024*1024):.2f}MB", end="\r")
    print("\nDownload completed successfully.")

def extract_zip(zip_path: Path = ZIP_PATH, extract_to: Path = DATA_DIR) -> None:
    """Extracts the zip file contents to the destination folder."""
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

import pandas as pd

def get_or_download_dataset() -> Path:
    """Ensures the dataset is downloaded and extracted, returning the CSV path."""
    if CSV_PATH.exists():
        print(f"Dataset already exists at {CSV_PATH.resolve()}")
        return CSV_PATH
    
    if not ZIP_PATH.exists():
        download_dataset()
        
    extract_zip()
    
    # Clean up the zip file to save space
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print("Cleaned up temporary zip file.")
        
    return CSV_PATH

def load_dataset() -> pd.DataFrame:
    """Ensures the dataset is downloaded and extracted, and returns it as a pandas DataFrame with proper headers."""
    csv_path = get_or_download_dataset()
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    print(f"Loading dataset from {csv_path}...")
    return pd.read_csv(csv_path, names=columns)

if __name__ == "__main__":
    load_dataset()

