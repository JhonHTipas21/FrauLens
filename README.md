# FraudLens - Explainable Fraud Detection System

FraudLens is an advanced financial transaction fraud detection system. It employs a two-phase ML pipeline—combining unsupervised anomaly detection (**Isolation Forest**) with supervised classification (**XGBoost**)—and features a decoupled explainability layer (**SHAP**) visualized on a premium **Streamlit** dashboard designed for risk analysts.

This project goes beyond giving a simple black-box classification score. It separates detection and presentation concerns, allowing auditors to inspect **exactly why** a transaction was flagged as high-risk, including the contribution of the unsupervised anomaly score itself.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [SOLID Principles Applied](#solid-principles-applied)
3. [Model Evaluation & Business Impact](#model-evaluation--business-impact)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Running the Application](#running-the-application)
7. [Testing Suite](#testing-suite)
8. [Automated CI/CD Pipeline](#automated-cicd-pipeline)

---

## System Architecture

The pipeline consists of four decoupled layers following a pattern validated in industrial production systems:

```
[CSV/Stream de transacciones] 
           ↓
    [Preprocesamiento]       # Scaler for Time & Amount
           ↓
   [Isolation Forest]        # Phase 1: Unsupervised Anomaly Score
           ↓
[XGBoost Hybrid Classifier]  # Phase 2: Supervised Class (Ingests features + anomaly score)
           ↓
  [SHAP TreeExplainer]       # Explainer Layer: Feature attribution (log-odds)
           ↓
  [Streamlit Dashboard]      # UI: Risk Tiers, cost trade-off, and Plotly SHAP graphs
```

1.  **Preprocessing Layer**: Standardizes raw variables (`Amount` and `Time`) while retaining pre-existing features.
2.  **Phase 1 (Unsupervised Anomaly Detection)**: An `Isolation Forest` model isolates rare transactions. Because fraud is extremely rare (<0.2%), unsupervised partitioning captures outliers regardless of labelling.
3.  **Phase 2 (Supervised Classification)**: An `XGBoost` classifier is trained using both the original transaction features and the continuous `anomaly_score` generated in Phase 1 as an additional feature.
4.  **Decoupled Explainer Layer**: `SHAP TreeExplainer` calculates exact Shapley additive feature attributions. It returns a generic `Explanation` format, ensuring the dashboard remains engine-agnostic.
5.  **Audit Dashboard**: A custom Streamlit application displaying transactional risk tiers (Low/Medium/High), model trade-offs, global importances, and interactive Plotly waterfalls of SHAP values.

---

## SOLID Principles Applied

The codebase is engineered strictly around SOLID design patterns:

*   **S (Single Responsibility Principle)**: All modules are fully decoupled: `data_loader.py` handles ingestion; `anomaly_detector.py` defines outliers; `classifier.py` runs classification; `explainer.py` generates attributions; and `app.py` runs the UI.
*   **O (Open/Closed Principle)**: The hybrid `FraudClassifier` accepts *any* scikit-learn compatible estimator and *any* class implementing `BaseAnomalyDetector`. You can swap Isolation Forest with an Autoencoder without modifying the training script or classifier.
*   **L (Liskov Substitution Principle)**: `IsolationForestDetector` implements `BaseAnomalyDetector`. Any other detector can be substituted without breaking the downstream classification logic.
*   **I (Interface Segregation Principle)**: Interfaces defined in `interfaces.py` are minimal and focused. The anomaly detection contract (`BaseAnomalyDetector`) is kept independent of explainability (`BaseExplainer`).
*   **D (Dependency Inversion Principle)**: The Streamlit dashboard depends entirely on high-level abstractions (`Explanation` and `FeatureExplanation` data structures), not on SHAP. If we want to switch the explainability engine to LIME, the UI code requires zero modification.

---

## Model Evaluation & Business Impact

### 1. Verification & Metrics (Temporal Split)
Fraud models must be evaluated chronologically, as fraud behavior shifts over time. We split the Kaggle Credit Card dataset temporally using the `Time` feature (first 80% of transactions for training, last 20% for testing).

| Model Evaluation (Test Set) | Precision | Recall (Sensibilidad) | F1-Score | AUC-PR |
| :--- | :---: | :---: | :---: | :---: |
| **Isolation Forest (Baseline)** | 7.37% | 51.02% | 0.1289 | 0.0481 |
| **FraudLens (Hybrid XGBoost @ 0.50)** | 63.03% | 76.53% | 0.6912 | 0.7708 |
| **FraudLens (Hybrid XGBoost @ 0.96)** | **93.33%** | **71.43%** | **0.8092** | **0.7708** |

*   **Baseline Outliers**: Flagging raw anomalies yields high False Positives (628 FPs vs 50 TPs), making manual audit unsustainable.
*   **FraudLens Hybrid Pipeline**: Drastically reduces False Positives down to **44** (at threshold 0.50) while increasing True Positives to **75**, resulting in a massive boost in AUC-PR (**0.7708**).
*   **F1-Optimal Umbral (0.96)**: Optimizes detection to **93.33% Precision** and **71.43% Recall**, identifying the majority of fraud cases with almost zero false alarms.

### 2. Business Impact Optimization
The Streamlit dashboard includes an **interactive cost-minimizer simulator**:
*   **False Negatives (Omissions)**: High cost ($250 average cost of chargebacks/stolen funds).
*   **False Positives (False Alarms)**: Low cost ($10 cost for support audits, verification SMS, or card re-issues).
*   The simulator aggregates these costs in real-time, charting the global operational loss across different decision thresholds and identifying the exact threshold that minimizes cash leakage.

---

## Project Structure

```
FrauLens/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI Configuration
├── data/                        # Dataset storage folder (Git ignored)
├── reports/
│   └── figures/                 # EDA and descriptive figures
├── src/
│   ├── __init__.py
│   ├── interfaces.py            # SOLID DIP contracts & dataclasses
│   ├── data_loader.py           # Ingests and processes Kaggle raw data
│   ├── anomaly_detector.py      # Unsupervised Isolation Forest implementation
│   ├── classifier.py            # Supervised XGBoost hybrid classifier
│   ├── explainer.py             # SHAP TreeExplainer wrapper
│   ├── utils.py                 # Evaluation metrics and serialization helpers
│   └── train.py                 # ML training pipeline script
├── dashboard/
│   ├── __init__.py
│   └── app.py                   # Streamlit audit UI
├── tests/
│   ├── __init__.py
│   ├── test_interfaces.py       # Contract tests
│   └── test_pipeline.py         # Integration & Preprocessing tests
├── notebooks/
│   └── 01_eda_and_baseline.py   # Exploratory Notebook script
├── requirements.txt             # Project library requirements
├── .gitignore                   # Exclusions list
└── README.md                    # Technical documentation
```

---

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/JhonHTipas21/FrauLens.git
    cd FrauLens
    ```

2.  **Create and activate a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: macOS users may need to run `brew install libomp` to install the OpenMP runtime required by XGBoost.*

4.  **Download dataset & Train models**:
    ```bash
    python3 src/train.py
    ```
    This script downloads the Kaggle Credit Card dataset (~150MB uncompressed), splits it temporally, scales features, fits both model phases, evaluates threshold metrics, and serializes the assets inside `models/`.

---

## Running the Application

Launch the Streamlit audit dashboard locally:
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

*   **Rendimiento Tab**: Simulate business savings by moving the decision threshold slider and updating cost parameter metrics.
*   **Auditoría Tab**: Inspect transaction rows. Select a suspicious ID and inspect the Plotly horizontal bar chart containing its SHAP attribution factors, including how much the unsupervised anomaly detector influenced the risk assessment.
*   **Global Tab**: View the overall model feature importances calculated with mean absolute SHAP values.

---

## 🧪 Testing Suite

We maintain unit and integration tests using `pytest` to ensure pipeline consistency:
```bash
python3 -m pytest tests/
```
The suite verifies:
*   Chronology of the temporal split.
*   Abstract contracts of `BaseAnomalyDetector` and `BaseExplainer`.
*   Correct dimensionality and scaling of `DataPreprocessor`.
*   Unsupervised output of the `IsolationForestDetector`.
*   SHAP explanation packaging and sorting.

---

## 🚀 Automated CI/CD Pipeline

We utilize **GitHub Actions** for CI/CD checks:
*   Runs automatically on every `push` or `pull_request` to `main`.
*   Sets up Python 3.9, installs dependencies, and runs `flake8` for linting.
*   Runs the `pytest` suite to guarantee code safety before merging.
