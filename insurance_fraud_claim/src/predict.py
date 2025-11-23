"""
predict.py
Loads trained models and generates predictions (fraud scores) 
for new claim data.
"""

import joblib
import pandas as pd
import numpy as np



# LOAD MODELS

def load_models(prefix: str):
    """
    Loads IsolationForest, RandomForestClassifier, and feature list.
    """
    iso = joblib.load(f"{prefix}_iso.joblib")
    clf = joblib.load(f"{prefix}_clf.joblib")
    feature_cols = joblib.load(f"{prefix}_features.joblib")

    print(f"Models loaded from prefix '{prefix}'")
    return iso, clf, feature_cols



# MAKE PREDICTIONS

def predict_from_models(df_features: pd.DataFrame, iso, clf, feature_cols):
    """
    df_features → DataFrame that already contains all feature columns.
    Returns:
        anomaly_score (0–1)
        classifier_probability (0–1)
    """

    # Ensure only model-required features
    X = df_features[feature_cols].fillna(0)

    # IsolationForest → anomaly score
    raw = iso.decision_function(X)
    anomaly_score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    # Classifier → probability score
    try:
        clf_proba = clf.predict_proba(X)[:, 1]
    except:
        clf_proba = np.zeros(len(X))

    return anomaly_score, clf_proba
