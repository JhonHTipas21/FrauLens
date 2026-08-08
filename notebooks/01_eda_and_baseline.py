# %% [markdown]
# # FraudLens — Exploratory Data Analysis & Baseline Model
# This notebook/script conducts the initial exploration of the Kaggle Credit Card Fraud Detection dataset
# and establishes an unsupervised baseline using Isolation Forest.

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_loader import get_or_download_dataset
from src.anomaly_detector import IsolationForestDetector
from src.utils import temporal_split, DataPreprocessor, calculate_metrics

# %%
# 1. Load Dataset
print("Loading dataset...")
csv_path = get_or_download_dataset()
df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")

# %%
# 2. Basic Statistics and Imbalance Analysis
print("\n--- Basic Statistics ---")
print(df.describe())

missing_values = df.isnull().sum().sum()
print(f"\nMissing values in dataset: {missing_values}")

class_counts = df['Class'].value_counts()
class_percentages = df['Class'].value_counts(normalize=True) * 100
print(f"\nClass Distribution:")
print(f"Normal (0): {class_counts[0]} ({class_percentages[0]:.4f}%)")
print(f"Fraud (1):  {class_counts[1]} ({class_percentages[1]:.4f}%)")

# %%
# 3. Create Reports Directory
figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# %%
# 4. Visualization: Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='Set2')
plt.title('Transaction Class Distribution')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Count')
plt.yscale('log')  # Log scale since class imbalance is extreme
plt.tight_layout()
plt.savefig(figures_dir / 'class_distribution.png', dpi=300)
plt.close()
print("Saved class distribution plot to reports/figures/class_distribution.png")

# %%
# 5. Visualization: Amount Distribution
plt.figure(figsize=(10, 5))
# Use log(Amount + 1) for visualization due to large range and skew
df_viz = df.copy()
df_viz['Log_Amount'] = np.log1p(df_viz['Amount'])
sns.kdeplot(data=df_viz, x='Log_Amount', hue='Class', common_norm=False, fill=True, palette='Set1', alpha=0.5)
plt.title('Log-scaled Transaction Amount Distribution')
plt.xlabel('Log(Amount + 1)')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(figures_dir / 'amount_distribution.png', dpi=300)
plt.close()
print("Saved transaction amount distribution plot to reports/figures/amount_distribution.png")

# %%
# 6. Visualization: Time Distribution
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x='Time', hue='Class', common_norm=False, fill=True, palette='Set1', alpha=0.5)
plt.title('Transaction Time Distribution')
plt.xlabel('Time (seconds elapsed since first transaction)')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(figures_dir / 'time_distribution.png', dpi=300)
plt.close()
print("Saved transaction time distribution plot to reports/figures/time_distribution.png")

# %%
# 7. Correlation Analysis (Sample of features)
plt.figure(figsize=(12, 10))
# Calculate correlation on a sample of V features + Time, Amount, Class
cols_to_corr = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 10)]
corr_matrix = df[cols_to_corr].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Selected Features')
plt.tight_layout()
plt.savefig(figures_dir / 'correlation_matrix.png', dpi=300)
plt.close()
print("Saved correlation matrix to reports/figures/correlation_matrix.png")

# %%
# 8. Unsupervised Baseline: Isolation Forest
print("\n--- Running Isolation Forest Baseline ---")

# Step 1: Temporal Split
train_df, test_df = temporal_split(df, time_col='Time', test_ratio=0.2)

# Step 2: Preprocess Features
preprocessor = DataPreprocessor()
preprocessor.fit(train_df, target_col='Class')

X_train, y_train = preprocessor.transform(train_df)
X_test, y_test = preprocessor.transform(test_df)

# Step 3: Train Isolation Forest (We set contamination to the expected fraud rate in train set)
contamination_rate = y_train.mean()
print(f"Training Isolation Forest with contamination rate: {contamination_rate:.6f}")

detector = IsolationForestDetector(contamination=contamination_rate, random_state=42)
detector.fit(X_train)

# Step 4: Evaluate on Test Set
y_pred = detector.predict_anomaly(X_test)
y_scores = detector.predict_anomaly_score(X_test)

# Min-max scale the anomaly score to [0, 1] range for metric calculation
# Isolation Forest decision scores can be transformed to probabilities/confidence levels
min_score = y_scores.min()
max_score = y_scores.max()
y_scores_normalized = (y_scores - min_score) / (max_score - min_score) if max_score > min_score else y_scores

# Calculate metrics
metrics = calculate_metrics(y_test, y_scores_normalized, threshold=0.5)

print("\n--- Baseline Isolation Forest Metrics on Test Set ---")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1']:.4f}")
print(f"AUC-PR:    {metrics['auc_pr']:.4f}")
print("Confusion Matrix:")
print(f"  True Negatives (Legit): {metrics['confusion_matrix']['tn']}")
print(f"  False Positives (Falsely flagged): {metrics['confusion_matrix']['fp']}")
print(f"  False Negatives (Missed Fraud):    {metrics['confusion_matrix']['fn']}")
print(f"  True Positives (Detected Fraud):   {metrics['confusion_matrix']['tp']}")

# Summary comments
print("\nEDA and baseline analysis completed successfully.")
