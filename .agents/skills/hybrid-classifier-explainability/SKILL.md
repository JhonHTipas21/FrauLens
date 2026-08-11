---
name: fraudlens-hybrid-classifier-explainability
description: Implementation guidelines for the FraudLens double-phase hybrid classifier (Isolation Forest + XGBoost) and SHAP tree explainability layer
---

# FraudLens Hybrid Classifier & Explainability Skill

This skill documents the design patterns, architecture, and explainability layer of FraudLens.

## Hybrid Classification Architecture (Double-Phase)
- **Decoupled Interfaces**: Interface definitions are stored in `src/interfaces.py`. The hybrid pipeline follows the Dependency Inversion Principle (DIP) and Open/Closed Principle (OCP) by wrapping any outlier detector in `BaseAnomalyDetector` and any classifier in standard scikit-learn interfaces.
- **Phase 1 (Unsupervised Anomaly Detection)**: 
  - Managed by `src/anomaly_detector.py` via `IsolationForestDetector`.
  - Fits an Isolation Forest model to capture rare transaction outliers (unsupervised).
  - Continuous anomaly score is inverted from scikit-learn's default (higher score = more anomalous).
- **Phase 2 (Supervised Classification)**:
  - Managed by `src/classifier.py` via `FraudClassifier`.
  - The feature space is augmented by appending the anomaly score from Phase 1 as the final column.
  - A supervised `XGBClassifier` is trained on this augmented feature set.
  - Inbalanced dataset is addressed via the `scale_pos_weight` hyperparameter calculated dynamically from training label counts.

## Decoupled Explainability Layer (SHAP)
- **Engine-Agnostic Output**: Visualizations and interfaces depend on the abstract `Explanation` and `FeatureExplanation` dataclasses defined in `src/interfaces.py`, ensuring that switching from SHAP to another engine (e.g. LIME) requires zero dashboard changes.
- **SHAP tree explainer wrapper**: Managed by `src/explainer.py` via `SHAPExplainer`.
- **Instance Explanation (`explain_instance`)**:
  - Predicts the anomaly score of the individual transaction, augments the feature space, and evaluates SHAP values using `shap.TreeExplainer` on the positive (fraud) class.
  - Returns a unified `Explanation` with base values (log-odds), prediction probabilities, and detailed feature attributions including the contribution of `anomaly_score`.
- **Global Explanation (`explain_global`)**:
  - Calculates global feature importances as the mean absolute SHAP value for each feature over a representative background sample (default sample size: 200).
  - Returns a sorted list mapping feature names to their absolute global importance score.

## Formatting & Code Quality
- Ensure all custom models inherit from `BaseAnomalyDetector` or `BaseExplainer` and conform strictly to the defined signatures.
- Mypy configurations in `mypy.ini` must be satisfied. Type annotations on preprocessor transformations and dataclass structures must be strictly checked before deployment.
